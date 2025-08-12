import h5py
import numpy as np
import matplotlib
# 尝试使用TkAgg后端，如果不可用则使用Agg后端
try:
    matplotlib.use('TkAgg')
except ImportError:
    try:
        matplotlib.use('Qt5Agg')
    except ImportError:
        matplotlib.use('Agg')
        print("警告: 使用Agg后端，可能无法显示图形窗口")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import argparse
import open3d as o3d

# 读取H5文件
def read_h5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        points = f['points'][:]
        suction_or = f['suction_or'][:]
        suction_seal_scores = f['suction_seal_scores'][:]
        suction_wrench_scores = f['suction_wrench_scores'][:]
        suction_feasibility_scores = f['suction_feasibility_scores'][:]
        individual_object_size_lable = f['individual_object_size_lable'][:]
    return points, suction_or, suction_seal_scores, suction_wrench_scores, suction_feasibility_scores, individual_object_size_lable

def create_coordinate_frame(size=0.1):
    """
    创建坐标轴框架
    """
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=size, origin=[0, 0, 0])
    return coordinate_frame

def visualize_point_cloud_with_o3d(points, normals=None, normal_num=100, show_axis=True, scores=None, show_heatmap=False):
    """
    使用Open3D可视化点云
    红色箭头表示X轴
    绿色箭头表示Y轴
    蓝色箭头表示Z轴
    """
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 添加热力图颜色（如果提供分数）
    if show_heatmap and scores is not None:
        # 归一化分数到[0,1]范围
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        
        # 创建颜色映射：蓝色(低分) -> 绿色(中分) -> 红色(高分)
        colors = np.zeros((len(points), 3))
        for i, score in enumerate(scores_norm):
            if score < 0.5:
                # 蓝色到绿色的渐变
                colors[i] = [0, 2*score, 1-2*score]
            else:
                # 绿色到红色的渐变
                colors[i] = [2*(score-0.5), 1-2*(score-0.5), 0]
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print(f"已应用热力图颜色映射，分数范围: {scores.min():.4f} - {scores.max():.4f}")
    else:
        # 如果不显示热力图，使用默认颜色
        default_color = np.array([[0.5, 0.5, 0.5]] * len(points))  # 灰色
        pcd.colors = o3d.utility.Vector3dVector(default_color)
    
    # 创建可视化几何体列表
    geometries = [pcd]
    
    # 添加法向量线条（如果提供）
    if normals is not None:
        # 只显示前normal_num个法向量
        arrow_points = []
        lines = []
        colors_normal = []
        scale = 0.1  # 法向量长度
        
        for i in range(min(normal_num, len(points))):
            start = points[i]
            end = points[i] + normals[i] * scale
            arrow_points.append(start)
            arrow_points.append(end)
            lines.append([2*i, 2*i+1])
            colors_normal.append([1, 1, 0])  # 黄色法向量，更容易区分
            
        if arrow_points:
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(np.array(arrow_points))
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors_normal)
            geometries.append(line_set)
    
    # 添加坐标轴
    if show_axis:
        coordinate_frame = create_coordinate_frame()
        geometries.append(coordinate_frame)
    
    # 可视化 - WSL环境优化的可视化方式
    try:
        # 检测是否在WSL环境中
        import os
        is_wsl = "microsoft" in os.uname().release.lower() or "WSL" in os.environ.get("WSL_DISTRO_NAME", "")
        
        if is_wsl:
            print("检测到WSL环境，使用离屏渲染...")
            # 在WSL环境中使用离屏渲染
            vis = o3d.visualization.Visualizer()
            vis.create_window(visible=False)  # 创建不可见窗口
            for geom in geometries:
                vis.add_geometry(geom)
            
            # 设置视角和渲染选项
            ctr = vis.get_view_control()
            ctr.set_zoom(0.8)
            
            # 离屏渲染并保存图像
            vis.poll_events()
            vis.update_renderer()
            image = vis.capture_screen_float_buffer(False)
            plt.figure(figsize=(12, 9))
            plt.imshow(np.asarray(image))
            plt.title("Point Cloud Visualization (Rendered)")
            plt.axis('off')
            plt.show()
            vis.destroy_window()
        else:
            # 非WSL环境，尝试标准可视化
            o3d.visualization.draw_geometries(geometries, 
                                            window_name="Point Cloud Visualization",
                                            width=800, height=600)
    except Exception as e:
        print(f"Open3D可视化失败: {e}")
        try:
            # 备用方案：使用可视化器
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Point Cloud Visualization", width=800, height=600)
            for geom in geometries:
                vis.add_geometry(geom)
            vis.run()
            vis.destroy_window()
        except Exception as e2:
            print(f"Open3D可视化器也失败: {e2}")
            print("自动切换到matplotlib可视化...")
            # 提取点云数据用于matplotlib可视化
            if len(geometries) > 0 and hasattr(geometries[0], 'points'):
                points_for_mpl = np.asarray(geometries[0].points)
                visualize_point_cloud_with_matplotlib(points_for_mpl, normals, normal_num)
            else:
                print("无法提取点云数据进行matplotlib可视化")

def visualize_point_cloud_with_matplotlib(points, normals=None, normal_num=100, scores=None, show_heatmap=False):
    """
    使用Matplotlib可视化点云
    """
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 绘制点云
    if show_heatmap and scores is not None:
        # 使用热力图颜色
        scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                           c=scores, cmap='coolwarm', s=4, alpha=0.8)
        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Score Value', rotation=270, labelpad=15)
        print(f"已应用热力图颜色映射，分数范围: {scores.min():.4f} - {scores.max():.4f}")
    else:
        # 使用默认颜色
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, c='gray', alpha=0.8)
    
    # 添加法向量
    if normals is not None:
        # 只显示前normal_num个法向量
        normal_points = points[:normal_num]
        normal_vectors = normals[:normal_num]
        
        # 绘制法向量（按一定比例缩放）
        scale = 0.1  # 法向量长度
        for i, (point, normal) in enumerate(zip(normal_points, normal_vectors)):
            end_point = point + normal * scale
            ax.plot([point[0], end_point[0]], 
                   [point[1], end_point[1]], 
                   [point[2], end_point[2]], 
                   color='yellow', linewidth=2.0)  # 黄色法向量，更容易区分
    
    # 添加坐标轴
    # 绘制坐标轴箭头
    max_range = np.array([points[:, 0].max()-points[:, 0].min(), 
                          points[:, 1].max()-points[:, 1].min(),
                          points[:, 2].max()-points[:, 2].min()]).max() / 2.0
    
    mid_x = (points[:, 0].max()+points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max()+points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max()+points[:, 2].min()) * 0.5
    
    # 绘制坐标轴
    axis_length = max_range * 0.5
    ax.quiver(mid_x, mid_y, mid_z, axis_length, 0, 0, color='red', arrow_length_ratio=0.1)
    ax.quiver(mid_x, mid_y, mid_z, 0, axis_length, 0, color='green', arrow_length_ratio=0.1)
    ax.quiver(mid_x, mid_y, mid_z, 0, 0, axis_length, color='blue', arrow_length_ratio=0.1)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Point Cloud Visualization with Heatmap' if show_heatmap else 'Point Cloud Visualization')
    
    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='G:/Diffusion_Suction_DataSet/train', help='数据集目录')
    parser.add_argument('--clycle_id', type=int, default=0, help='循环编号')
    parser.add_argument('--scene_id', type=int, default=30, help='场景编号')
    parser.add_argument('--vision_normal_num', type=int, default=10, help='可视化向量个数')
    parser.add_argument('--method', type=str, default='matplotlib', choices=['o3d', 'matplotlib'], help='可视化方法')
    parser.add_argument('--show_axis', type=bool, default=True, help='是否显示坐标轴')
    parser.add_argument('--vision_score_type', type=str, default='suction_seal_score', choices=['suction_score','suction_seal_score','suction_wrench_score','suction_feasibility_score', 'individual_object_size_lable'], help='可视化分数类型')
    parser.add_argument('--show_heatmap', type=bool, default=True, help='是否显示热力图图')
    args = parser.parse_args()
    
    h5_file_path = os.path.join(args.data_dir, 'cycle_' + str(args.clycle_id).zfill(4), str(args.scene_id).zfill(3) + '.h5')
    print(f'打开文件: {h5_file_path}')
    if not os.path.exists(h5_file_path):
        print(f"文件 {h5_file_path} 不存在，请检查路径。")
        return
    point_cloud, normals, suction_seal_scores, suction_wrench_scores, suction_feasibility_scores, individual_object_size_lable = read_h5_file(h5_file_path)
    
    # 剔除重复点
    unique_points, unique_indices = np.unique(point_cloud, axis=0, return_index=True)
    if len(unique_points) < point_cloud.shape[0]:
        point_cloud = unique_points
        normals = normals[unique_indices]
        suction_seal_scores = suction_seal_scores[unique_indices]
        suction_wrench_scores = suction_wrench_scores[unique_indices]
        suction_feasibility_scores = suction_feasibility_scores[unique_indices]
        individual_object_size_lable = individual_object_size_lable[unique_indices]
        
        if not point_cloud.flags['C_CONTIGUOUS']:
            point_cloud = np.ascontiguousarray(point_cloud)
            normals = np.ascontiguousarray(normals)
            suction_seal_scores = np.ascontiguousarray(suction_seal_scores)
            suction_wrench_scores = np.ascontiguousarray(suction_wrench_scores)
            suction_feasibility_scores = np.ascontiguousarray(suction_feasibility_scores)
            individual_object_size_lable = np.ascontiguousarray(individual_object_size_lable)
    
    suction_score = suction_seal_scores * suction_wrench_scores * suction_feasibility_scores * individual_object_size_lable
    
    # 按照suction_score将点云排序
    if args.vision_score_type =='suction_score':
        sorted_indices = np.argsort(suction_score)[::-1]
    elif args.vision_score_type =='suction_seal_score':
        sorted_indices = np.argsort(suction_seal_scores)[::-1]
    elif args.vision_score_type =='suction_wrench_score':
        sorted_indices = np.argsort(suction_wrench_scores)[::-1]
    elif args.vision_score_type =='suction_feasibility_score':
        sorted_indices = np.argsort(suction_feasibility_scores)[::-1]
    elif args.vision_score_type == 'individual_object_size_lable':
        sorted_indices = np.argsort(individual_object_size_lable)[::-1]
    else:
        print(f"可视化分数类型 {args.vision_score_type} 不支持。")
        return
    # 应用排序到所有数据
    point_cloud = point_cloud[sorted_indices]
    normals = normals[sorted_indices] if normals is not None else None
    suction_score = suction_score[sorted_indices]
    suction_seal_scores = suction_seal_scores[sorted_indices]
    suction_wrench_scores = suction_wrench_scores[sorted_indices]
    suction_feasibility_scores = suction_feasibility_scores[sorted_indices]
    individual_object_size_lable = individual_object_size_lable[sorted_indices]
    
    # 取前args.vision_normal_num个向量
    vision_normals = normals[:args.vision_normal_num] if normals is not None else None
    
    # 获取当前可视化分数类型对应的分数
    score_map = {
        'suction_score': suction_score,
        'suction_seal_score': suction_seal_scores[sorted_indices],
        'suction_wrench_score': suction_wrench_scores[sorted_indices],
        'suction_feasibility_score': suction_feasibility_scores[sorted_indices],
        'individual_object_size_lable': individual_object_size_lable[sorted_indices]
    }
    current_scores = score_map.get(args.vision_score_type, suction_score)
    
    # 可视化分数直方图和点云
    print(f"点云形状: {point_cloud.shape}")
    
    # 先显示直方图窗口
    if args.show_heatmap:
        try:
            # 创建直方图
            plt.figure("Score Histogram", figsize=(10, 6))
            plt.hist(current_scores, bins=100, color='royalblue', alpha=0.7, edgecolor='black')
            plt.title(f"Histogram of {args.vision_score_type}")
            plt.xlabel(f"Score Value of {args.vision_score_type}")
            plt.ylabel("Frequency")
            plt.grid(True, alpha=0.3)
            
            # 添加统计信息
            mean_score = np.mean(current_scores)
            std_score = np.std(current_scores)
            plt.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.4f}')
            plt.legend()
            
            print(f"分数统计: 均值={mean_score:.4f}, 标准差={std_score:.4f}, 最小值={current_scores.min():.4f}, 最大值={current_scores.max():.4f}")
            
            # 保存图片并显示
            # plt.savefig('score_histogram.png', dpi=150, bbox_inches='tight')
            # print("直方图已保存为 score_histogram.png")
            plt.show(block=False)
            
        except Exception as e:
            print(f"直方图显示失败: {e}")
            print("将继续显示点云...")
    
    # 再显示点云窗口
    if args.method == 'o3d':
        print("使用Open3D可视化点云...")
        try:
            visualize_point_cloud_with_o3d(point_cloud, vision_normals, args.vision_normal_num, args.show_axis, current_scores, args.show_heatmap)
        except Exception as e:
            print(f"Open3D可视化失败: {e}")
            print("尝试使用matplotlib可视化...")
            visualize_point_cloud_with_matplotlib(point_cloud, vision_normals, args.vision_normal_num, current_scores, args.show_heatmap)
    else:
        print("使用Matplotlib可视化点云...")
        visualize_point_cloud_with_matplotlib(point_cloud, vision_normals, args.vision_normal_num, current_scores, args.show_heatmap)
    
    # 保持直方图窗口打开
    if args.show_heatmap:
        try:
            input("按回车键关闭所有窗口...")
            plt.close('all')
        except:
            pass

if __name__ == "__main__":
    main()