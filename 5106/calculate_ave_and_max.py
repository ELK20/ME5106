import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

# 加载数据文件
data = loadmat('dataSingleTracks.mat')

# 定义傅里叶变换和特征提取函数
def calculate_fft_features(signal, sample_rate=128000):
    """
    对信号进行傅里叶变换并计算频域上的平均频率和最大频率幅值
    参数:
    - signal: 时间域信号数据
    - sample_rate: 采样率 (Hz)
    返回:
    - frequencies: 频率数组
    - magnitudes: 幅值数组
    - avg_freq: 平均频率
    - max_magnitude_freq: 最大幅值对应的频率
    - max_magnitude: 最大幅值
    """
    # 计算傅里叶变换
    fft_result = np.fft.fft(signal)
    magnitudes = np.abs(fft_result)[:len(fft_result) // 2]
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(magnitudes)]

    # 计算最大幅值及其对应频率
    max_magnitude = np.max(magnitudes)
    max_magnitude_freq = frequencies[np.argmax(magnitudes)]

    # 计算平均频率 (频率乘以幅值的加权平均)
    avg_freq = np.sum(frequencies * magnitudes) / np.sum(magnitudes)

    return frequencies, magnitudes, avg_freq, max_magnitude_freq, max_magnitude

# 提取每个样本的特征
def extract_frequency_features(data, sample_rate=128000):
    features = {}
    for sample_name, sample_data in data.items():
        if sample_name in ['__header__', '__version__', '__globals__']:
            continue
        
        signal = sample_data.flatten()  # 将数据转换为一维
        frequencies, magnitudes, avg_freq, max_magnitude_freq, max_magnitude = calculate_fft_features(signal, sample_rate)

        # 存储频域特征
        features[sample_name] = {
            'avg_frequency': avg_freq,
            'max_magnitude_frequency': max_magnitude_freq,
            'max_magnitude': max_magnitude
        }
        
        # 可视化样本的频谱图
        plt.figure()
        plt.plot(frequencies, magnitudes)
        plt.title(f'Frequency Spectrum of Sample {sample_name}')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.show()

    return features

# 运行特征提取并打印结果
frequency_features = extract_frequency_features(data)
for sample, feature in frequency_features.items():
    print(f"Sample {sample}: Average Frequency = {feature['avg_frequency']:.2f} Hz, Max Frequency = {feature['max_magnitude_frequency']:.2f} Hz, Max Magnitude = {feature['max_magnitude']:.2f}")
    