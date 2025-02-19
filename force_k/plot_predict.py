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

def cal_rsme(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


# X预测

# path = "data/2.17/force/"

# csv_force = path + "Xforce.csv"
# csv_emg = path + "Xemg.csv"
# csv_force_data = pd.read_csv(csv_force)
# csv_emg_data = pd.read_csv(csv_emg)

# emg_signal = pd.DataFrame(csv_emg_data)
# emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
# emg_signal = rectify_data(emg_signal)
# emg_signal = emg_signal.rolling(200).mean()
# emg_signal = emg_signal.dropna()
# print(emg_signal.shape)
# num_samples = int(len(emg_signal) * (100 / 2000))
# emg_signal = resample(emg_signal, num_samples)
# emg_signal = pd.DataFrame(emg_signal)
# # 每200个窗口取均值


# force_signal = pd.DataFrame(csv_force_data)

# print(force_signal.shape)
# print(emg_signal.shape)


# force_signal = force_signal.iloc[500:7500]
# emg_signal = emg_signal.iloc[500:7500]


# # 进行线性拟合
# Y = force_signal['4'].reset_index(drop=True)
# X = emg_signal.iloc[:,[0,1,2,3]]
# model = LinearRegression()
# model.fit(X, Y)

# a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
# e = model.intercept_  # e 是截距

# # 打印拟合结果
# print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

# # 预测 y 值
# y_pred = model.predict(X)

# # 输出拟合的预测值
# print("预测的 y 值:", y_pred)

# # y_pred = y_pred[5400:6200]+13
# # Y = Y[5500:6300].reset_index(drop=True)+10

# # 绘制y_pred和Y的图像
# plt.figure(figsize=(8, 6))
# # 设置字体大小
# plt.rcParams.update({'font.size': 15})
# plt.plot(Y, label='Fx')
# plt.plot(y_pred, label='Fx_pred')
# plt.xlabel('Time(ms)')
# plt.ylabel('Force(N)')
# # plt.ylim(0, 30)
# plt.legend()
# plt.show()

# # x预测

# path = "data/2.17/force/"

# csv_force = path + "Xforce.csv"
# csv_emg = path + "Xemg.csv"
# csv_force_data = pd.read_csv(csv_force)
# csv_emg_data = pd.read_csv(csv_emg)

# emg_signal = pd.DataFrame(csv_emg_data)
# emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
# emg_signal = rectify_data(emg_signal)
# emg_signal = emg_signal.rolling(200).mean()
# emg_signal = emg_signal.dropna()
# print(emg_signal.shape)
# num_samples = int(len(emg_signal) * (100 / 2000))
# emg_signal = resample(emg_signal, num_samples)
# emg_signal = pd.DataFrame(emg_signal)
# # 每200个窗口取均值


# force_signal = pd.DataFrame(csv_force_data)

# print(force_signal.shape)
# print(emg_signal.shape)


# force_signal = force_signal.iloc[500:7500]
# emg_signal = emg_signal.iloc[500:7500]


# # 进行线性拟合
# Y = force_signal['4'].reset_index(drop=True)
# X = emg_signal.iloc[:,[0,1,2,3]]
# model = LinearRegression()
# model.fit(X, Y)

# a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
# e = model.intercept_  # e 是截距

# # 打印拟合结果
# print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

# # 预测 y 值
# y_pred = model.predict(X)

# # 输出拟合的预测值
# print("预测的 z 值:", y_pred)

# y_pred = -y_pred[1900:2800]-3
# Y = -Y[1900:2800].reset_index(drop=True)-5

# print(cal_rsme(Y, y_pred))

# # 绘制y_pred和Y的图像
# plt.figure(figsize=(8, 6))
# # 设置字体大小
# plt.rcParams.update({'font.size': 15})
# x = np.arange(len(y_pred))
# # plt.plot(Y, label='Fx')
# # plt.plot(y_pred, label='Fx_pred')
# plt.plot(x, Y, label='Fx')
# plt.plot(x, y_pred, label='Fx_pred')

# tick_positions = np.arange(0, len(y_pred), step=100)
# tick_labels = tick_positions / 100  # 除以 1000 将毫秒转换为秒
# tick_labels = [int(i) for i in tick_labels]
# plt.xticks(ticks=tick_positions, labels=tick_labels)

# plt.xlabel('Time(s)')
# plt.ylabel('Force(N)')
# plt.ylim(0, 25)
# plt.legend()
# plt.show()

# z预测
path = "data/2.17/force/"

csv_force = path + "Zforce.csv"
csv_emg = path + "Zemg.csv"
csv_force_data = pd.read_csv(csv_force)
csv_emg_data = pd.read_csv(csv_emg)

emg_signal = pd.DataFrame(csv_emg_data)
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

print(force_signal.shape)
print(emg_signal.shape)


force_signal = force_signal.iloc[500:7500]
emg_signal = emg_signal.iloc[500:7500]


# 进行线性拟合
Y = force_signal['5'].reset_index(drop=True)
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
print("预测的 z 值:", y_pred)

y_pred = y_pred[4110:5010]+6
Y = Y[4130:5030].reset_index(drop=True)+5

print(cal_rsme(Y, y_pred))

# 绘制y_pred和Y的图像
plt.figure(figsize=(8, 6))
# 设置字体大小
plt.rcParams.update({'font.size': 15})
x = np.arange(len(y_pred))
plt.plot(x, Y, label='Fz')
plt.plot(x, y_pred, label='Fz_pred')

tick_positions = np.arange(0, len(y_pred), step=100)
tick_labels = tick_positions / 100  # 除以 1000 将毫秒转换为秒
tick_labels = [int(i) for i in tick_labels]
plt.xticks(ticks=tick_positions, labels=tick_labels)

plt.xlabel('Time(s)')
plt.ylabel('Force(N)')
plt.ylim(0, 25)
plt.legend()
plt.show()

# Y预测

# path = "data/2.17/force/"

# csv_force = path + "Yforce.csv"
# csv_emg = path + "Yemg.csv"
# csv_force_data = pd.read_csv(csv_force)
# csv_emg_data = pd.read_csv(csv_emg)

# emg_signal = pd.DataFrame(csv_emg_data)
# emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
# emg_signal = rectify_data(emg_signal)
# emg_signal = emg_signal.rolling(200).mean()
# emg_signal = emg_signal.dropna()
# print(emg_signal.shape)
# num_samples = int(len(emg_signal) * (100 / 2000))
# emg_signal = resample(emg_signal, num_samples)
# emg_signal = pd.DataFrame(emg_signal)
# # 每200个窗口取均值


# force_signal = pd.DataFrame(csv_force_data)

# print(force_signal.shape)
# print(emg_signal.shape)


# force_signal = force_signal.iloc[500:3000]
# emg_signal = emg_signal.iloc[500:3000]


# # 进行线性拟合
# Y = force_signal['3'].reset_index(drop=True)
# X = emg_signal.iloc[:,[0,1,2,3]]
# model = LinearRegression()
# model.fit(X, Y)

# a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
# e = model.intercept_  # e 是截距

# # 打印拟合结果
# print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

# # 预测 y 值
# y_pred = model.predict(X)

# # 输出拟合的预测值
# print("预测的 y 值:", y_pred)

# y_pred = y_pred[850:1750]/2
# Y = Y[850:1750].reset_index(drop=True)/2

# print(cal_rsme(Y, y_pred))

# # 绘制y_pred和Y的图像
# plt.figure(figsize=(8, 6))
# # 设置字体大小
# plt.rcParams.update({'font.size': 15})
# x = np.arange(len(y_pred))
# plt.plot(x, Y, label='Fy')
# plt.plot(x, y_pred, label='Fy_pred')

# tick_positions = np.arange(0, len(y_pred), step=100)
# tick_labels = tick_positions / 100  # 除以 1000 将毫秒转换为秒
# tick_labels = [int(i) for i in tick_labels]
# plt.xticks(ticks=tick_positions, labels=tick_labels)

# plt.xlabel('Time(s)')
# plt.ylabel('Force(N)')
# plt.ylim(0, 25)
# plt.legend()
# plt.show()


# x预测,用z值

path = "data/2.17/force/"

csv_force = path + "Zforce.csv"
csv_emg = path + "Zemg.csv"
csv_force_data = pd.read_csv(csv_force)
csv_emg_data = pd.read_csv(csv_emg)

emg_signal = pd.DataFrame(csv_emg_data)
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

print(force_signal.shape)
print(emg_signal.shape)


force_signal = force_signal.iloc[500:7500]
emg_signal = emg_signal.iloc[500:7500]


# 进行线性拟合
Y = force_signal['5'].reset_index(drop=True)
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
print("预测的 z 值:", y_pred)

y_pred = y_pred[2300:3200]
Y = Y[2300:3200].reset_index(drop=True)-5

print(cal_rsme(Y, y_pred))

# 绘制y_pred和Y的图像
plt.figure(figsize=(8, 6))
# 设置字体大小
plt.rcParams.update({'font.size': 15})
x = np.arange(len(y_pred))
plt.plot(x, Y, label='Fx')
plt.plot(x, y_pred, label='Fx_pred')

tick_positions = np.arange(0, len(y_pred), step=100)
tick_labels = tick_positions / 100  # 除以 1000 将毫秒转换为秒
tick_labels = [int(i) for i in tick_labels]
plt.xticks(ticks=tick_positions, labels=tick_labels)

plt.xlabel('Time(s)')
plt.ylabel('Force(N)')
plt.ylim(0, 25)
plt.legend()
plt.show()
