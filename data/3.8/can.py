import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import numpy as np

# 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge_new.pkl')

# # 读取数据
# emg_data = data[0]
# k_data = data[1]


# emg_all = np.zeros((4, 0))
# k_all = np.zeros((3, 0))

# # 打印k_data的第一列
# print(np.array(k_data)[:, 1])

# print(len(emg_data))

# for i in range(len(emg_data)):
#     emg = emg_data[i]
#     emg_all = np.concatenate((emg_all,emg),axis=1)
#     k = k_data[i]
#     num = emg.shape[1]
#     # 将k重复num次
#     k = np.tile(k, (num, 1))
#     k_all = np.concatenate((k_all, k.T), axis=1)


# print(emg_all.shape)
# print(k_all.shape)


# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.plot(emg_all.T)
# plt.title('EMG Sensor')
# plt.xlabel('Time(s)')
# plt.ylabel('EMG Value')
# plt.grid()
# ax2 = fig.add_subplot(212)
# ax2.plot(k_all.T)
# plt.title('K')
# plt.xlabel('Time(s)')
# plt.ylabel('K Value')
# plt.grid()
# plt.show()

# # # 删除emg_data和k_data最后一个元素
# # emg_data.pop()
# # k_data.pop()
# # # 保存数据
# # pd.to_pickle([emg_data, k_data], './data/3.8/merge.pkl')
# # print('数据保存成功')


# k_x = [138., 125., 132., 115, 102., 104., 109., 80.48854091, 40., 59.,
#     57., 40.52958818, 62., 43., 31., 33.38115215, 65., 66., 72., 75.30580216,
#     83., 78., 72, 75.30580216]

# # 生成 k_y，是 k_x 的 2.2-2.6 倍数随机
# k_y = [x * np.random.uniform(2.2, 2.6)+ np.random.uniform(-25, 25) for x in k_x]

# k_z = [x * np.random.uniform(1.5, 1.9)+ np.random.uniform(-25, 25) for x in k_x]

# # 将k_x, k_y, k_z转换为numpy数组
# k_x = np.array(k_x)
# k_y = np.array(k_y)
# k_z = np.array(k_z)

# # 合并k_x, k_y, k_z，行三个元素
# k_combined = np.vstack((k_x, k_y, k_z)).T


# print(k_y)
# print(k_z)

# k_all1 = np.zeros((3, 0))
# for i in range(len(emg_data)):
#     emg = emg_data[i]
#     k = k_combined[i]
#     print(k)
#     num = emg.shape[1]
#     # 将k重复num次
#     k = np.tile(k, (num, 1))
#     print(k)
#     k_all1 = np.concatenate((k_all1, k.T), axis=1)


# fig = plt.figure()
# ax1 = fig.add_subplot(211)
# ax1.plot(emg_all.T)
# plt.title('EMG Sensor')
# plt.xlabel('Time(s)')
# plt.ylabel('EMG Value')
# plt.grid()
# ax2 = fig.add_subplot(212)
# ax2.plot(k_all1.T)
# plt.title('K')
# plt.xlabel('Time(s)')
# plt.ylabel('K Value')
# plt.grid()
# plt.show()

# # 保存数据
# pd.to_pickle([emg_data, k_combined], './data/3.8/merge_make.pkl')
# print('数据保存成功')



# # 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge_make1.pkl')

# 读取数据
emg_data = data[0]
k_data = data[1]
print(len(emg_data))
print(len(k_data))

# 生成随机的1-24
random_list = np.random.permutation(24)

# 将emg_data以random_list的顺序打乱
emg_data = [emg_data[i] for i in random_list]
# 将k_data以random_list的顺序打乱
k_data = [k_data[i] for i in random_list]

# 保存数据
pd.to_pickle([emg_data, k_data], './data/3.8/merge_make1.pkl')
print('数据保存成功')
