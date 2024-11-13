import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 加载数据
data = loadmat('dataSingleTracks.mat')

# 定义傅里叶变换函数
def calculate_fft(signal, sample_rate=128000):
    """
    对输入的时间域信号进行傅里叶变换，返回频率和对应的幅值。
    
    参数:
    - signal: 时间域信号数据（如电压）
    - sample_rate: 采样频率 (128kHz)
    
    返回:
    - frequencies: 频率
    - magnitudes: 对应频率的幅值
    """
    # 傅里叶变换
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)  # 幅值
    fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]  # 取一半频率值
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft_magnitude)]
    return frequencies, fft_magnitude

# 定义特征提取函数
def extract_features_from_samples(data, sample_rate=128000):
    """
    对每个样本进行傅里叶变换并提取特征。
    
    参数:
    - data: 样本数据字典
    - sample_rate: 采样频率
    
    返回:
    - features: 包含每个样本的特征的字典
    """
    features = {}
    for sample_name, sample_data in data.items():
        if sample_name in ['__header__', '__version__', '__globals__']:
            continue  # 跳过非数据项

        # 对单个样本信号进行傅里叶变换
        signal = sample_data.flatten()  # 将数据转换为一维
        freqs, magnitudes = calculate_fft(signal, sample_rate)

        # 提取特征：平均幅值、最大幅值、平均频率、最大频率
        avg_magnitude = np.mean(magnitudes)
        max_magnitude = np.max(magnitudes)
        avg_frequency = np.sum(freqs * magnitudes) / np.sum(magnitudes)  # 频率加权平均值
        max_frequency = np.max(freqs)  # 频率范围中的最大频率

        # 存储特征
        features[sample_name] = {
            'avg_magnitude': avg_magnitude,
            'max_magnitude': max_magnitude,
            'avg_frequency': avg_frequency,
            'max_frequency': max_frequency
        }

        # 可视化某些样本的频谱图（选几个样本画图）
        if sample_name in ['A3', 'B10']:  # 可以换成任何你想看的样本
            plt.figure()
            plt.plot(freqs, magnitudes)
            plt.title(f'Frequency Spectrum of Sample {sample_name}')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.show()

    return features

# 运行特征提取函数并显示结果
sample_features = extract_features_from_samples(data)
for sample, feature in sample_features.items():
    print(f"Sample {sample}: "
          f"Average Frequency = {feature['avg_frequency']:.2f} Hz, "
          f"Max Frequency = {feature['max_frequency']:.2f} Hz")
