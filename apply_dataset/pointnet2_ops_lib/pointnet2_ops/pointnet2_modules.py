from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    # 构建共享的多层感知机(MLP), 可选是否使用批归一化
    # mlp_spec: 每层的通道数列表, 例如[64,128,256]
    # bn: 是否使用BatchNorm
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))  # inplace激活

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    """
    PointNet++ 的集合抽象(SA)模块基类, 支持采样和分组操作。
    """
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None      # 采样点数
        self.groupers = None    # 分组操作器(如球查询)
        self.mlps = None        # 每个分组对应的MLP

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        前向传播, 进行采样、分组、特征聚合。

        参数说明
        ----------
        xyz : torch.Tensor
            (B, N, 3) 代表特征点的三维坐标
        features : torch.Tensor
            (B, C, N) 代表特征点的特征描述

        返回值
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) 新采样点的三维坐标
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) 新采样点的特征描述
        """

        new_features_list = []

        # xyz: (B, N, 3) -> (B, 3, N)
        xyz_flipped = xyz.transpose(1, 2).contiguous()
        # 采样npoint个点作为新的中心点
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            # 分组操作, 得到每个中心点的邻域特征 (B, C, npoint, nsample)
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )

            # 经过MLP处理 (B, mlp[-1], npoint, nsample)
            new_features = self.mlps[i](new_features)
            # 在每个分组内做最大池化, 得到每个中心点的特征 (B, mlp[-1], npoint, 1)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )
            # 去掉最后一维 (B, mlp[-1], npoint)
            new_features = new_features.squeeze(-1)

            new_features_list.append(new_features)

        # 多尺度特征拼接 (B, sum(mlp[-1]), npoint)
        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""PointNet多尺度分组的集合抽象层(MSG)

    支持对每个采样点用不同半径、不同采样数的球查询, 提取多尺度特征。
    参数说明
    ----------
    npoint : int
        采样点数量
    radii : list of float32
        每个分组的半径列表
    nsamples : list of int32
        每个球查询的采样点数
    mlps : list of list of int32
        每个尺度下MLP的结构
    bn : bool
        是否使用批归一化
    use_xyz : bool
        是否将xyz坐标拼接到特征中
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            # 构建分组操作器(球查询或全局分组)
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3  # 输入特征加上xyz

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class PointnetSAModule(PointnetSAModuleMSG):
    r"""PointNet集合抽象层(单尺度分组)

    只用一个球查询和一个MLP, 适合简单场景。
    参数说明
    ----------
    npoint : int
        采样点数量
    radius : float
        球查询半径
    nsample : int
        球查询内采样点数
    mlp : list
        MLP结构
    bn : bool
        是否使用批归一化
    use_xyz : bool
        是否将xyz坐标拼接到特征中
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""特征传播模块(Feature Propagation), 用于插值上采样, 将已知点集的特征传播到未知点集。

    常用于PointNet++的解码阶段, 实现点云特征的逐层细化。
    参数说明
    ----------
    mlp : list
        PointNet模块参数
    bn : bool
        是否使用批归一化
    """

    def __init__(self, mlp, bn=True):
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        r"""
        前向传播, 插值传播特征。

        参数说明
        ----------
        unknown : torch.Tensor
            (B, n, 3) 未知点的三维坐标(需要插值的点)
        known : torch.Tensor
            (B, m, 3) 已知点的三维坐标(有特征的点)
        unknow_feats : torch.Tensor
            (B, C1, n) 未知点的特征(可选)
        known_feats : torch.Tensor
            (B, C2, m) 已知点的特征

        返回值
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) 未知点的新特征
        """

        if known is not None:
            # 三邻域插值, 计算距离和索引
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)  # 防止除零
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            # 根据权重插值已知点的特征
            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            # 如果没有已知点, 直接扩展特征
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            # 拼接未知点自身特征和插值特征
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)  # (B, C, n, 1)
        new_features = self.mlp(new_features)      # 经过MLP
        return new_features.squeeze(-1)            # (B, mlp[-1], n)
