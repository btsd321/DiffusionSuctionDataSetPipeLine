import OpenEXR
import Imath
import numpy as np
import matplotlib.pyplot as plt
import os

def read_exr_to_numpy(exr_path):
    exr_file = OpenEXR.InputFile(exr_path)
    dw = exr_file.header()['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    pt = Imath.PixelType(Imath.PixelType.FLOAT)
    channels = ['R', 'G', 'B']
    img = np.zeros((height, width, 3), dtype=np.float32)
    for i, c in enumerate(channels):
        img[:, :, i] = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32).reshape(height, width)
    return img

# 分割图路径
seg_dir = r'D:/Project/DiffusionSuctionDataSetPipeLine/Data/Diffusion_Suction_DataSet/segment_images/cycle_0000/030'
# 找到EXR文件
exr_files = [f for f in os.listdir(seg_dir) if f.endswith('.exr')]
if not exr_files:
    print("未找到EXR分割图文件")
else:
    exr_path = os.path.join(seg_dir, exr_files[0])
    img = read_exr_to_numpy(exr_path)
    print("R通道最大/最小值:", img[:,:,0].max(), img[:,:,0].min())
    print("G通道最大/最小值:", img[:,:,1].max(), img[:,:,1].min())
    print("B通道最大/最小值:", img[:,:,2].max(), img[:,:,2].min())
    # 可视化G通道
    plt.imshow(img[:,:,1], cmap='viridis')
    plt.title('G Channel (Gradient)')
    plt.colorbar()
    plt.show()
    # 可视化RGB
    plt.imshow(img[:,:,:])
    plt.title('Segment Image RGB')
    plt.show()