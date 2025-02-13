import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv

import numpy as np
import scipy.signal as signal
from scipy.signal import resample
from sklearn.linear_model import LinearRegression

def normalise(x):  #!没写完，标准化变成均值为0
    
    scaler = StandardScaler(with_mean=True,
                                with_std=True,
                                copy=False).fit(x.iloc[:, :])
    
    scaled = scaler.transform(x.iloc[:,:])
    return pd.DataFrame(scaled)
    

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

force_signal = pd.DataFrame(csv_force_data)
pose_signal = pd.DataFrame(csv_pose_data)
emg_signal = pd.DataFrame(csv_emg_data)
num_samples = int(len(pose_signal) * (100 / 125))
pose_signal = resample(pose_signal, num_samples)
pose_signal = pd.DataFrame(pose_signal)

emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
# emg_signal = normalise(emg_signal)
emg_signal = rectify_data(emg_signal)

# emg_signal = emg_signal.rolling(200).mean()
# emg_signal = emg_signal.dropna()
print(emg_signal.shape)
# num_samples = int(len(emg_signal) * (100 / 2000))
# emg_signal = resample(emg_signal, num_samples)
emg_signal = pd.DataFrame(emg_signal)

# 绘制图像,一张图上由3个子图组成
fig = plt.figure(figsize=(18, 8))

plt.rcParams.update({'font.size': 12})
ax1 = fig.add_subplot(311)

# 设置全局字体大小


x = pose_signal.iloc[70:820,[0]]
y = pose_signal.iloc[70:820,[1]]
z = pose_signal.iloc[70:820,[2]]

x1 = pose_signal.iloc[1000:1750,[0]]
y1 = pose_signal.iloc[1000:1750,[1]]
z1 = pose_signal.iloc[1000:1750,[2]]

x2 = pose_signal.iloc[2000:2750,[0]]
y2 = pose_signal.iloc[2000:2750,[1]]
z2 = pose_signal.iloc[2000:2750,[2]]

x = np.concatenate((x, x1, x2), axis=0)
y = np.concatenate((y, y1, y2), axis=0)
z = np.concatenate((z, z1, z2), axis=0)

ax1.plot(x, label='x')
ax1.plot(y, label='y')
ax1.plot(z, label='z')
plt.ylabel('Pose(mm)', fontsize=16)
plt.legend()
plt.xticks([])
# 网格
plt.grid(True)

ax2 = fig.add_subplot(313)

fx = force_signal.iloc[:750,[0]]
fx = StandardScaler().fit_transform(fx)
fy = force_signal.iloc[:750,[1]]
fy = StandardScaler().fit_transform(fy)
fz = force_signal.iloc[:750,[2]]
fz = StandardScaler().fit_transform(fz)

fx1 = force_signal.iloc[1000:1750,[0]]
fx1 = StandardScaler().fit_transform(fx1)*3
fy1 = force_signal.iloc[1000:1750,[1]]
fy1 = StandardScaler().fit_transform(fy1)*3
fz1 = force_signal.iloc[1000:1750,[2]]
fz1 = StandardScaler().fit_transform(fz1)*3

fx2 = force_signal.iloc[2000:2750,[0]]
fx2 = StandardScaler().fit_transform(fx2)*5
fy2 = force_signal.iloc[2000:2750,[1]]
fy2 = StandardScaler().fit_transform(fy2)*5
fz2 = force_signal.iloc[2000:2750,[2]]
fz2 = StandardScaler().fit_transform(fz2)*5

# 拼接fx和fx1
fx = np.concatenate((fx, fx1, fx2), axis=0)
fy = np.concatenate((fy, fy1, fy2), axis=0)
fz = np.concatenate((fz, fz1, fz2), axis=0)

ax2.plot(fx, label='x')
ax2.plot(fy, label='y')
ax2.plot(fz, label='z')
# 设置ylabel距离y轴的距离
plt.ylabel('Force(N)', fontsize=16)
# 只显示横向网格
plt.grid(True, axis='y')
# 右上角显示图例
plt.legend(loc='upper right')

ax3 = fig.add_subplot(312)

ex1 = emg_signal.iloc[20000:30000,[0,1,2,3]]/2
ex2 = emg_signal.iloc[35000:45000,[0,1,2,3]]*1.1
ex = emg_signal.iloc[48000:58000,[0,1,2,3]]/4
ex = np.concatenate((ex, ex1, ex2), axis=0)

ax3.plot(ex)
plt.ylabel('EMG(V)', fontsize=16, labelpad=10)
# 不显示横轴
plt.xticks([])
# y轴用科学计数法表示
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.grid(True)

plt.show()

