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


path = "data/2.17/force/"

csv_force = path + "Zforce.csv"
csv_emg = path + "Zemg.csv"
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

print(force_signal.shape)
print(emg_signal.shape)


force_signal = force_signal.iloc[500:7500]
emg_signal = emg_signal.iloc[500:7500]

# emg_signal第一列乘100， 第二列乘1000，求和
emg_signal = emg_signal * [372434.07012185, 1220167.8851617225, 46693.71400183649, 701693.5158285548]
emg_signal = emg_signal.sum(axis=1)
emg_signal = pd.DataFrame(emg_signal)
# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(emg_signal, label='emg')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()
plt.show()
