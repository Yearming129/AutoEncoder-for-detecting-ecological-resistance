import os
import numpy as np
import rasterio
from rasterio.enums import Compression

# 输入文件路径：指定目录下所有.tif文件
input_folder = r'C:\Users\MR\Desktop\files\Landscape Disturb'
input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

# 输出文件夹路径
output_folder = r'C:\Users\MR\Desktop\files\disturb'
os.makedirs(output_folder, exist_ok=True)


def normalize(data, nodata_value):
    """归一化函数，将数据归一化到 [0, 1] 范围"""
    data[data == nodata_value] = np.nan  # 将 NoData 值设置为 NaN
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    print(f"最大值为：{data_max}, 最小值为“{data_min}”")

    # 避免除以零
    if data_max - data_min == 0:
        return np.zeros_like(data, dtype=np.float32)

    normalized_data = (data - data_min) / (data_max - data_min)
    normalized_data[np.isnan(normalized_data)] = 0  # 将 NaN 值赋值为 0
    return normalized_data


def process_file(input_file, output_folder):
    """读取、归一化并保存影像"""
    with rasterio.open(input_file) as src:
        # 读取数据
        data = src.read(1).astype(np.float32)  # 只读取第一个波段
        nodata_value = src.nodata  # 获取 NoData 值
        # 获取元数据
        meta = src.meta.copy()

        # 进行归一化
        norm_data = normalize(data, nodata_value)

        # 更新元数据
        meta.update({
            'dtype': 'float32',
            'compress': 'lzw',
            'nodata': np.nan  # 更新 NoData 值为 NaN
        })

        # 构造输出文件路径
        output_file = os.path.join(output_folder, os.path.basename(input_file))

        # 保存归一化后的影像
        with rasterio.open(output_file, 'w', **meta) as dst:
            dst.write(norm_data, 1)


for input_file in input_files:
    process_file(input_file, output_folder)

print("归一化完成，文件已保存到:", output_folder)