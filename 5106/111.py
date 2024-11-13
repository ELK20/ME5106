import scipy.io

# 加载文件
data = scipy.io.loadmat('dataDenoise.mat')

# 打印所有变量名
print("Variables in .mat file:", data.keys())
