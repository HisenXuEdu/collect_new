import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv

import numpy as np
import scipy.signal as signal
from scipy.signal import resample


def save_data(dataframe, file_name):
    pass
    # dataframe.to_csv(file_name, index=False)



path = "data/2.12/zuizhong/"

csv_file = path + "POSE1739367202590.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 510
end = 4470
x = csv_df.iloc[:,[0]]
x = x - x.iloc[500]
x = x[start:end]
y = csv_df.iloc[:,[1]]
y = y - y.iloc[500]
y = y[start:end]
z = csv_df.iloc[:,[2]]
z = z - z.iloc[500]
z = z[start:end]
plt.figure(figsize=(20, 3))
plt.plot(x, label='x')
plt.plot(y, label='y')
plt.plot(z, label='z')
plt.xlabel('time(ms)')
plt.ylabel('pose(mm)')
z_p = z
# 将x，y，z保存为csv文件
save_data(pd.concat([x, y, z], axis=1), path + 'pose.csv')

csv_file = path + "FORCE1739367195226.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 1200
end = 4360
x = csv_df.iloc[:,[3]]
x = x[start:end]
# 使x均值为0
# x = StandardScaler().fit_transform(x)
y = csv_df.iloc[:,[4]]
y = y[start:end]
# y = StandardScaler().fit_transform(y)
z = csv_df.iloc[:,[5]]
z = z[start:end]
# z = StandardScaler().fit_transform(z)
# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(y)
plt.plot(x)
plt.plot(z)
# plt.ylim(-5,5)
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
save_data(pd.concat([y, x, z], axis=1), path + 'force.csv')
z = StandardScaler().fit_transform(z)
z_f = z

# z_p重新设置序号
z_p = z_p.reset_index(drop=True)
# z_f = z_f.reset_index(drop=True)
# 计算新的采样点数
num_samples = int(len(z_p) * (100 / 125))
z_p = resample(z_p, num_samples)

plt.figure(figsize=(20, 3))
plt.plot(z_p)
plt.plot(z_f*5)
plt.xlabel('time(ms)')
plt.ylabel('pose(mm) and force(N)')


csv_file = path + "EMG1739367199383.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 14900
end = 78100
# x = csv_df.iloc[:,[1]]
# x = x[start:end]
# y = csv_df.iloc[:,[1]]
# y = y[start:end]
# z = csv_df.iloc[:,[3]]
# z = z[start:end]

c1 = csv_df.iloc[:,[1]]
c1 = c1[start:end]
c2 = csv_df.iloc[:,[2]]
c2 = c2[start:end]
c3 = csv_df.iloc[:,[3]]
c3 = c3[start:end]
c4 = csv_df.iloc[:,[4]]
c4 = c4[start:end]
c5 = csv_df.iloc[:,[5]]
c5 = c5[start:end]
c6 = csv_df.iloc[:,[0]]
c6 = c6[start:end]

# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(c1)
plt.plot(c2)
plt.plot(c3)
plt.xlabel('time(ms)')
plt.ylabel('force(N)')

save_data(pd.concat([c1, c2, c3, c4, c5, c6], axis=1), path + 'emg.csv')
plt.show()
