import numpy as np
from scipy.io import loadmat

# 加载数据文件
data = loadmat('dataSingleTracks.mat')

# 定义计算平均值的函数
def calculate_average(data):
    """
    计算每个样本的平均值
    参数:
    - data: 从 .mat 文件中加载的数据
    返回:
    - averages: 包含每个样本平均值的字典
    """
    averages = {}
    for sample_name, sample_data in data.items():
        # 跳过文件头部信息
        if sample_name in ['__header__', '__version__', '__globals__']:
            continue
        
        # 将数据转换为一维数组，并计算平均值
        signal = sample_data.flatten()  # 将数据转换为一维
        avg_value = np.mean(signal)  # 计算平均值
        averages[sample_name] = avg_value

    return averages

# 调用函数并输出结果
sample_averages = calculate_average(data)
for sample, avg in sample_averages.items():
    print(f"Sample {sample}: Average Value = {avg:.2f}")
