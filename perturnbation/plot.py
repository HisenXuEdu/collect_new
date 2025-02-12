import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

csv_file = "data/pose/POSE1739367202590.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 100
end = 5000
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

csv_file = "data/force/FORCE1739367195226.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 0
end = 4400
x = csv_df.iloc[:,[3]]
x = x[start:end]
# 使x均值为0
x = StandardScaler().fit_transform(x)
y = csv_df.iloc[:,[4]]
y = y[start:end]
y = StandardScaler().fit_transform(y)
z = csv_df.iloc[:,[5]]
z = z[start:end]
z = StandardScaler().fit_transform(z)
# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(y)
plt.plot(x)
plt.plot(z)
plt.ylim(-5,5)
plt.xlabel('time(ms)')
plt.ylabel('force(N)')


csv_file = "data/emg/EMG1739367199383.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)
start = 1000
end = 100000
x = csv_df.iloc[:,[1]]
x = x[start:end]
y = csv_df.iloc[:,[2]]
y = y[start:end]
z = csv_df.iloc[:,[6]]
z = z[start:end]
# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(y)
plt.plot(x)
plt.plot(z)
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.show()