import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# 加载去噪后的声音数据文件
data = loadmat('dataDenoise.mat')

# 获取数据中的信号
# 假设你的信号存储在data的某个特定变量中，例如 'signal'
# 请根据文件内容修改 'signal_variable' 为实际变量名称
signal = data['singletracks_denoise'].flatten()  # 将信号转换为一维数组

# 设置采样率
sample_rate = 128000  # 根据实验设置的采样率，128kHz为例

# 定义傅里叶变换函数
def calculate_fft(signal, sample_rate=128000):
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)  # 获取幅值
    fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]  # 只取正频率部分
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)[:len(fft_magnitude)]
    return frequencies, fft_magnitude

# 计算信号的频谱
frequencies, magnitudes = calculate_fft(signal, sample_rate)

# 绘制频谱图
plt.figure(figsize=(10, 6))
plt.plot(frequencies, magnitudes)
plt.title('Frequency Spectrum of Signal')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.grid(True)
plt.show()
