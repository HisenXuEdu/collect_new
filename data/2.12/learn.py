import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv

import numpy as np
import scipy.signal as signal
from scipy.signal import resample
from sklearn.linear_model import LinearRegression

def filter_data(emg_data, f, butterworth_order = 4, btype = 'lowpass'):
    #力并没有进行滤波，因为后面窗口内的取均值作为真值
    f_sampling = 2000
    nyquist = f_sampling/2
    if isinstance(f, int):
        fc = f/nyquist
    else:
        fc = list(f)
        for i in range(len(f)):
            fc[i] = fc[i]/nyquist
            
    b,a = signal.butter(butterworth_order, fc, btype=btype)
    transpose = emg_data.T.copy()
    
    for i in range(len(transpose)):
        transpose[i] = (signal.lfilter(b, a, transpose[i]))
    
    return pd.DataFrame(transpose.T)

def rectify_data(emg_data):
    return abs(emg_data)


path = "data/2.12/zuizhong/"

csv_pose = path + "pose.csv"
csv_force = path + "force.csv"
csv_emg = path + "emg.csv"
csv_pose_data = pd.read_csv(csv_pose)
csv_force_data = pd.read_csv(csv_force)
csv_emg_data = pd.read_csv(csv_emg)

emg_signal = pd.DataFrame(csv_emg_data)
# emg_signal = emg_signal.iloc[0:6000]
emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
emg_signal = rectify_data(emg_signal)
emg_signal = emg_signal.rolling(200).mean()
emg_signal = emg_signal.dropna()
print(emg_signal.shape)
num_samples = int(len(emg_signal) * (100 / 2000))
emg_signal = resample(emg_signal, num_samples)
emg_signal = pd.DataFrame(emg_signal)
# 每200个窗口取均值


force_signal = pd.DataFrame(csv_force_data)
pose_signal = pd.DataFrame(csv_pose_data)
num_samples = int(len(pose_signal) * (100 / 125))
pose_signal = resample(emg_signal, num_samples)

print(pose_signal.shape)
print(force_signal.shape)
print(emg_signal.shape)

# 全取2000-3000部分的数据
pose_signal = pd.DataFrame(pose_signal).iloc[1000:3000]
force_signal = force_signal.iloc[1000:3000]
emg_signal = emg_signal.iloc[1000:3000]


# 进行线性拟合



# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
c1 = emg_signal.iloc[:,[0]]
c2 = emg_signal.iloc[:,[2]]
plt.plot(c1, label='emg1')
plt.plot(c2, label='emg3')
plt.xlabel('time(ms)')
plt.ylabel('emg')
plt.show()

# Y = force_signal.iloc[:,[0]]
# Y = StandardScaler().fit_transform(Y)
# Y = force_signal[['4']]
# Y = StandardScaler().fit_transform(Y)
# Y = pd.DataFrame(Y)
# Y = Y[0]
# print(Y)
Y = force_signal['5']
# Y = StandardScaler().fit_transform(Y)
X = emg_signal.iloc[:,[0,1,2,3]]
model = LinearRegression()
model.fit(X, Y)

a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
e = model.intercept_  # e 是截距

# 打印拟合结果
print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

# 预测 y 值
y_pred = model.predict(X)

# 输出拟合的预测值
print("预测的 y 值:", y_pred)

# 绘制y_pred和Y的图像
plt.figure(figsize=(20, 3))
plt.plot(y_pred, label='y_pred')
plt.plot(Y, label='Y')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.show()

# import numpy as np
# from sklearn.linear_model import LinearRegression
# import pandas as pd

# # 示例数据
# data = {
#     'x1': [1, 2, 3, 4, 5],
#     'x2': [5, 4, 3, 2, 1],
#     'x3': [2, 3, 4, 5, 6],
#     'x4': [6, 7, 8, 9, 10],
#     'y': [12, 14, 16, 18, 20]
# }

# # 创建 DataFrame
# df = pd.DataFrame(data)

# # 提取特征 x1, x2, x3, x4
# X = df[['x1', 'x2', 'x3', 'x4']]
# print(X)
# print(X.shape)

# # 提取目标变量 y
# y = df['y']

# # 创建并训练线性回归模型
# model = LinearRegression()
# model.fit(X, y)

# # 获取拟合的参数（a, b, c, d）和截距 e
# a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
# e = model.intercept_  # e 是截距

# # 打印拟合结果
# print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

# # 预测 y 值
# y_pred = model.predict(X)

# # 输出拟合的预测值
# print("预测的 y 值:", y_pred)