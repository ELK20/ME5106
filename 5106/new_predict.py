import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.io import loadmat
import random

# 加载数据文件
data = loadmat('dataSingleTracks.mat')
laser_parameters = {
    'A1': {'Laser Power': 350, 'Scan Speed': 1200},
    'A2': {'Laser Power': 400, 'Scan Speed': 600},
    'A3': {'Laser Power': 250, 'Scan Speed': 600},
    'A4': {'Laser Power': 250, 'Scan Speed': 1000},
    'A5': {'Laser Power': 400, 'Scan Speed': 1000},
    'A7': {'Laser Power': 350, 'Scan Speed': 700},
    'A8': {'Laser Power': 200, 'Scan Speed': 500},
    'A9': {'Laser Power': 350, 'Scan Speed': 500},
    'A10': {'Laser Power': 200, 'Scan Speed': 400},
    'A11': {'Laser Power': 200, 'Scan Speed': 700},
    'A12': {'Laser Power': 250, 'Scan Speed': 500},
    'A13': {'Laser Power': 400, 'Scan Speed': 1200},
    'B1': {'Laser Power': 200, 'Scan Speed': 200},
    'B2': {'Laser Power': 300, 'Scan Speed': 1000},
    'B3': {'Laser Power': 150, 'Scan Speed': 600},
    'B4': {'Laser Power': 400, 'Scan Speed': 800},
    'B5': {'Laser Power': 150, 'Scan Speed': 500},
    'B6': {'Laser Power': 300, 'Scan Speed': 800},
    'B7': {'Laser Power': 150, 'Scan Speed': 400},
    'B8': {'Laser Power': 300, 'Scan Speed': 700},
    'B9': {'Laser Power': 400, 'Scan Speed': 1600},
    'B11': {'Laser Power': 150, 'Scan Speed': 200},
    'B13': {'Laser Power': 400, 'Scan Speed': 2000},
    'A6': {},  # A6 需要预测
    'B10': {},  # B10 需要预测
    'B12': {'Scan Speed': 500}  # B12 的 Scan Speed 已知
}

# 定义频段范围  
frequency_bands = [(0, 8000), (8000, 24500), (24500, 65000), (55000, 60000)]

# 定义傅里叶变换和特征提取函数
def calculate_fft(signal, sample_rate=128000):
    fft_result = np.fft.fft(signal)
    fft_magnitude = np.abs(fft_result)
    fft_magnitude = fft_magnitude[:len(fft_magnitude) // 2]
    frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)[:len(fft_magnitude)]
    return frequencies, fft_magnitude

# 计算频谱熵
def calculate_spectral_entropy(magnitudes):
    power_spectrum = magnitudes ** 2  # 功率谱
    power_spectrum /= np.sum(power_spectrum)  # 归一化
    spectral_entropy = -np.sum(power_spectrum * np.log2(power_spectrum + 1e-12))  # 避免log2(0)
    return spectral_entropy

# 计算峰值因子（Crest Factor）
def calculate_crest_factor(signal):
    peak_value = np.max(np.abs(signal))  # 峰值
    rms_value = np.sqrt(np.mean(signal ** 2))  # 均方根值
    crest_factor = peak_value / rms_value  # 计算峰值因子
    return crest_factor

# 提取幅值特征并计算频段的峰值因子
def extract_amplitude_features(frequencies, magnitudes, bands):
    features = []
    for i, (low, high) in enumerate(bands):
        band_indices = np.where((frequencies >= low) & (frequencies < high))
        band_mags = magnitudes[band_indices]
        
        avg_magnitude = np.mean(band_mags)  # 计算平均幅值
        max_magnitude = np.max(band_mags)   # 计算最大幅值
        features.extend([avg_magnitude, max_magnitude])

        # 仅对低频和中频计算峰值因子
        if i == 0 or i == 1:
            crest_factor = calculate_crest_factor(band_mags)
            features.append(crest_factor)
    return features

# 创建数据集
def create_dataset(data, laser_parameters, frequency_bands, sample_rate=128000):
    samples = []
    targets = []
    for sample_name, params in laser_parameters.items():
        signal = data.get(sample_name, None)
        if signal is not None:
            signal = signal.flatten()
            freqs, mags = calculate_fft(signal, sample_rate)
            
            # 提取频段幅值特征和峰值因子
            amplitude_features = extract_amplitude_features(freqs, mags, frequency_bands)
            
            # 计算频谱熵和整体峰值因子
            spectral_entropy = calculate_spectral_entropy(mags)
            crest_factor = calculate_crest_factor(signal)
            
            # 整合所有特征
            features = amplitude_features + [spectral_entropy, crest_factor]
            samples.append(features)
            
            # 仅在参数已知时添加标签
            if 'Laser Power' in params and 'Scan Speed' in params:
                targets.append([params['Laser Power'], params['Scan Speed']])
    
    return np.array(samples, dtype=np.float32), np.array(targets, dtype=np.float32)

# 生成训练和测试数据集
X, y = create_dataset(data, laser_parameters, frequency_bands)
indices = np.arange(len(y))
random.shuffle(indices)
train_indices = indices[:20]
X_train, y_train = X[train_indices], y[train_indices]

# 定义神经网络结构
class LaserProcessingNet(nn.Module):
    def __init__(self):
        super(LaserProcessingNet, self).__init__()
        self.hidden1 = nn.Linear(12, 64)  # 12个输入特征
        self.hidden2 = nn.Linear(64, 32)
        self.hidden3 = nn.Linear(32, 16)
        self.output = nn.Linear(16, 2)  # 输出激光功率和扫描速度
    
    def forward(self, x):
        x = torch.relu(self.hidden1(x))
        x = torch.relu(self.hidden2(x))
        x = torch.relu(self.hidden3(x))
        x = self.output(x)
        return x

# 创建模型、定义损失函数和优化器
model = LaserProcessingNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 转换训练数据为Tensor
X_train_tensor = torch.tensor(X_train)
y_train_tensor = torch.tensor(y_train)

# 训练模型
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 定义预测函数，针对 A6、B10 和 B12 进行预测
def predict_specific_samples(data, sample_names, frequency_bands):
    model.eval()
    predictions = {}
    with torch.no_grad():
        for sample_name in sample_names:
            signal = data.get(sample_name, None)
            if signal is not None:
                signal = signal.flatten()
                freqs, mags = calculate_fft(signal)
                
                # 提取特征
                features = extract_amplitude_features(freqs, mags, frequency_bands)
                spectral_entropy = calculate_spectral_entropy(mags)
                crest_factor = calculate_crest_factor(signal)
                features = features + [spectral_entropy, crest_factor]
                
                # 转换为Tensor并预测
                features_tensor = torch.tensor([features], dtype=torch.float32)
                prediction = model(features_tensor).numpy().flatten()
                
                if sample_name == 'B12':
                    # B12的Scan Speed已知，为500
                    predictions[sample_name] = {'Laser Power': prediction[0], 'Scan Speed': 500}
                else:
                    predictions[sample_name] = {'Laser Power': prediction[0], 'Scan Speed': prediction[1]}
    return predictions

# 对 A6、B10 和 B12 进行预测
sample_names = ['A6', 'B10', 'B12']
predictions = predict_specific_samples(data, sample_names, frequency_bands)

# 输出预测结果
print("\nA6、B10和B12的预测结果：")
for sample_name, prediction in predictions.items():
    print(f"{sample_name}: Laser Power = {prediction['Laser Power']:.2f}, Scan Speed = {prediction['Scan Speed']:.2f}")