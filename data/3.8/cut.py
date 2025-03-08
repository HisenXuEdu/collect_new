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
    if i == 7:
        k[1] = 380
    if i == 14:
        k[1] = 400
    num = emg.shape[1]
    # 将k重复num次
    k = np.tile(k, (num, 1))
    k_all = np.concatenate((k_all, k.T), axis=1)


print(emg_all.shape)
print(k_all.shape)


# 删除emg_data和k_data的正数第8个元素
# emg_data.pop(13)
# k_data.pop(13)

# 删除最后4个元素
# emg_data = emg_data[:-4]
# k_data = k_data[:-4]


# 保存数据
pd.to_pickle([emg_data, k_data], './data/3.8/merge1.pkl')
print('数据保存成功')


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

