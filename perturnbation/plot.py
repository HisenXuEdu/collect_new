import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# csv_file = "data/pose/POSE1739367202590.csv"
# csv_data = pd.read_csv(csv_file)#防止弹出警告
# csv_df = pd.DataFrame(csv_data)
# print(csv_df.shape)
# start = 100
# end = 5000
# x = csv_df.iloc[:,[0]]
# x = x - x.iloc[500]
# x = x[start:end]
# y = csv_df.iloc[:,[1]]
# y = y - y.iloc[500]
# y = y[start:end]
# z = csv_df.iloc[:,[2]]
# z = z - z.iloc[500]
# z = z[start:end]
# plt.figure(figsize=(20, 3))
# plt.plot(x, label='x')
# plt.plot(y, label='y')
# plt.plot(z, label='z')
# plt.xlabel('time(ms)')
# plt.ylabel('pose(mm)')


def save_data(dataframe, file_name):
    dataframe.to_csv(file_name, index=False)
    pass

path = 'data/2.17/force/'

csv_file = "data/force/XFORCE.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 350
end = 8500
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
plt.plot(y, label='y')
plt.plot(x, label='x')
plt.plot(z, label='z')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()


save_data(pd.concat([y, x, z], axis=1), path + 'Xforce.csv')


csv_file = "data/emg/XEMG.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 8500
end = 171500

c1 = csv_df.iloc[:,[1]]
c1 = c1[start:end]
c2 = csv_df.iloc[:,[2]]
c2 = c2[start:end]
c3 = csv_df.iloc[:,[3]]
c3 = c3[start:end]
c4 = csv_df.iloc[:,[4]]
c4 = c4[start:end]

# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(c1, label='emg1')
plt.plot(c2, label='emg2')
plt.plot(c3, label='emg3')
plt.plot(c4, label='emg4')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')

save_data(pd.concat([c1, c2, c3, c4], axis=1), path + 'Xemg.csv')

plt.show()

