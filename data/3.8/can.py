import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import numpy as np

# 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge.pkl')

# 读取数据
emg_data = data[0]
k_data = data[1]


emg_all = np.zeros((4, 0))
k_all = np.zeros((3, 0))
print(len(emg_data))

for i in range(len(emg_data)):
    emg = emg_data[i]
    emg_all = np.concatenate((emg_all,emg),axis=1)
    k = k_data[i]
    num = emg.shape[1]
    # 将k重复num次
    k = np.tile(k, (num, 1))
    k_all = np.concatenate((k_all, k.T), axis=1)


print(emg_all.shape)
print(k_all.shape)


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(emg_all.T)
plt.title('EMG Sensor')
plt.xlabel('Time(s)')
plt.ylabel('EMG Value')
plt.grid()
ax2 = fig.add_subplot(212)
ax2.plot(k_all.T)
plt.title('K')
plt.xlabel('Time(s)')
plt.ylabel('K Value')
plt.grid()
plt.show()

# # 删除emg_data和k_data最后一个元素
# emg_data.pop()
# k_data.pop()
# # 保存数据
# pd.to_pickle([emg_data, k_data], './data/3.8/merge.pkl')
# print('数据保存成功')