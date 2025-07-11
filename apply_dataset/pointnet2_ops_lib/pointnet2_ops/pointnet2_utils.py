import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *

# 优先尝试导入C++/CUDA扩展模块, 如果失败则JIT编译
try:
    import pointnet2_ops._ext as _ext
except ImportError:
    from torch.utils.cpp_extension import load
    import glob
    import os.path as osp
    import os

    warnings.warn("无法直接加载pointnet2_ops的C++扩展, 正在JIT编译。")

    _ext_src_root = osp.join(osp.dirname(__file__), "_ext-src")
    _ext_sources = glob.glob(osp.join(_ext_src_root, "src", "*.cpp")) + glob.glob(
        osp.join(_ext_src_root, "src", "*.cu")
    )
    _ext_headers = glob.glob(osp.join(_ext_src_root, "include", "*"))

    os.environ["TORCH_CUDA_ARCH_LIST"] = "3.7+PTX;5.0;6.0;6.1;6.2;7.0;7.5"
    _ext = load(
        "_ext",
        sources=_ext_sources,
        extra_include_paths=[osp.join(_ext_src_root, "include")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-Xfatbin", "-compress-all"],
        with_cuda=True,
    )

# ===========================
# 1. 最远点采样(Furthest Point Sampling)
# ===========================
class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        """
        使用迭代的最远点采样方法, 选择一组具有最大最小距离的npoint个特征点
        xyz: (B, N, 3) 输入点云
        npoint: int 采样点数
        返回: (B, npoint) 采样点的索引
        """
        out = _ext.furthest_point_sampling(xyz, npoint)
        ctx.mark_non_differentiable(out)  # 采样操作不可微
        return out

    @staticmethod
    def backward(ctx, grad_out):
        # 采样操作不可微, 直接返回空
        return ()

furthest_point_sample = FurthestPointSampling.apply

# ===========================
# 2. 特征收集操作(Gather Operation)
# ===========================
class GatherOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        根据索引从特征张量中收集指定的特征
        features: (B, C, N)
        idx: (B, npoint)
        返回: (B, C, npoint)
        """
        ctx.save_for_backward(idx, features)
        return _ext.gather_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        idx, features = ctx.saved_tensors
        N = features.size(2)
        grad_features = _ext.gather_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, None

gather_operation = GatherOperation.apply

# ===========================
# 3. 三邻域查找(ThreeNN)
# ===========================
class ThreeNN(Function):
    @staticmethod
    def forward(ctx, unknown, known):
        """
        查找unknown中每个点在known中的三个最近邻点
        unknown: (B, n, 3)
        known: (B, m, 3)
        返回: dist (B, n, 3) 距离, idx (B, n, 3) 索引
        """
        dist2, idx = _ext.three_nn(unknown, known)
        dist = torch.sqrt(dist2)
        ctx.mark_non_differentiable(dist, idx)
        return dist, idx

    @staticmethod
    def backward(ctx, grad_dist, grad_idx):
        # 该操作不可微
        return ()

three_nn = ThreeNN.apply

# ===========================
# 4. 三点插值(ThreeInterpolate)
# ===========================
class ThreeInterpolate(Function):
    @staticmethod
    def forward(ctx, features, idx, weight):
        """
        对三个特征点进行加权线性插值
        features: (B, c, m)
        idx: (B, n, 3)
        weight: (B, n, 3)
        返回: (B, c, n)
        """
        ctx.save_for_backward(idx, weight, features)
        return _ext.three_interpolate(features, idx, weight)

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播, 计算特征的梯度
        grad_out: (B, c, n)
        返回: grad_features, None, None
        """
        idx, weight, features = ctx.saved_tensors
        m = features.size(2)
        grad_features = _ext.three_interpolate_grad(
            grad_out.contiguous(), idx, weight, m
        )
        return grad_features, torch.zeros_like(idx), torch.zeros_like(weight)

three_interpolate = ThreeInterpolate.apply

# ===========================
# 5. 分组操作(Grouping Operation)
# ===========================
class GroupingOperation(Function):
    @staticmethod
    def forward(ctx, features, idx):
        """
        根据索引对特征进行分组
        features: (B, C, N)
        idx: (B, npoint, nsample)
        返回: (B, C, npoint, nsample)
        """
        ctx.save_for_backward(idx, features)
        return _ext.group_points(features, idx)

    @staticmethod
    def backward(ctx, grad_out):
        """
        反向传播, 计算特征的梯度
        grad_out: (B, C, npoint, nsample)
        返回: grad_features, None
        """
        idx, features = ctx.saved_tensors
        N = features.size(2)
        grad_features = _ext.group_points_grad(grad_out.contiguous(), idx, N)
        return grad_features, torch.zeros_like(idx)

grouping_operation = GroupingOperation.apply

# ===========================
# 6. 球查询操作(Ball Query)
# ===========================
class BallQuery(Function):
    @staticmethod
    def forward(ctx, radius, nsample, xyz, new_xyz):
        """
        球查询操作, 在指定半径内查找邻域点
        radius: float 球半径
        nsample: int 每个球内最多采样的点数
        xyz: (B, N, 3) 所有点
        new_xyz: (B, npoint, 3) 球心
        返回: (B, npoint, nsample) 邻域点索引
        """
        output = _ext.ball_query(new_xyz, xyz, radius, nsample)
        ctx.mark_non_differentiable(output)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        # 球查询不可微
        return ()

ball_query = BallQuery.apply

# ===========================
# 7. 分组与特征拼接(QueryAndGroup)
# ===========================
class QueryAndGroup(nn.Module):
    r"""
    使用球查询(ball query)方式进行分组
    radius: 球半径
    nsample: 每个球内最多采样的点数
    use_xyz: 是否将xyz坐标拼接到特征中
    """

    def __init__(self, radius, nsample, use_xyz=True):
        super(QueryAndGroup, self).__init__()
        self.radius, self.nsample, self.use_xyz = radius, nsample, use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        xyz: (B, N, 3) 所有点的三维坐标
        new_xyz: (B, npoint, 3) 球心坐标
        features: (B, C, N) 点的特征描述
        返回: (B, 3 + C, npoint, nsample) 分组后的特征张量
        """
        idx = ball_query(self.radius, self.nsample, xyz, new_xyz)
        xyz_trans = xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        grouped_xyz -= new_xyz.transpose(1, 2).unsqueeze(-1)  # 归一化坐标

        if features is not None:
            grouped_features = grouping_operation(features, idx)
            if self.use_xyz:
                # 拼接坐标和特征
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, C + 3, npoint, nsample)
            else:
                new_features = grouped_features
        else:
            # 只用坐标作为特征
            assert (
                self.use_xyz
            ), "不能既没有特征又不使用xyz作为特征！"
            new_features = grouped_xyz

        return new_features

# ===========================
# 8. 全局分组(GroupAll)
# ===========================
class GroupAll(nn.Module):
    r"""
    对所有点进行分组(全局分组)
    use_xyz: 是否将xyz坐标拼接到特征中
    """

    def __init__(self, use_xyz=True):
        super(GroupAll, self).__init__()
        self.use_xyz = use_xyz

    def forward(self, xyz, new_xyz, features=None):
        """
        xyz: (B, N, 3) 所有点的三维坐标
        new_xyz: 忽略
        features: (B, C, N) 点的特征描述
        返回: (B, C + 3, 1, N) 分组后的特征张量
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)  # (B, 3, 1, N)
        if features is not None:
            grouped_features = features.unsqueeze(2)     # (B, C, 1, N)
            if self.use_xyz:
                new_features = torch.cat(
                    [grouped_xyz, grouped_features], dim=1
                )  # (B, 3 + C, 1, N)
            else:
                new_features = grouped_features
        else:
            new_features = grouped_xyz

        return new_features
