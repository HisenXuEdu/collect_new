import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

# 读取pkl
data = pd.read_pickle('./data/3.8/pose1/zhong2.pkl')
force_list = data[0]
pose_list = data[1]
emg_list = data[2]


num = 1

# 将emg_list转换为numpy数组
emg_list = np.array(emg_list[num])
print(emg_list.shape)

force_list = np.array(force_list[num])
print(force_list.shape)

pose_list = np.array(pose_list[num])
print(pose_list.shape)

# 画力传感器数据
plt.figure()
ax1 = plt.subplot(311)
ax1.plot(force_list[:, 1], label='x')
ax1.plot(force_list[:, 0], label='y')
ax1.plot(force_list[:, 2], label='z')
plt.legend()
plt.title('Force Sensor')
plt.ylabel('Force(N)')
plt.xlabel('Time(s)')
plt.grid()

ax2 = plt.subplot(312)
ax2.plot(pose_list[:, 0]-140)
ax2.plot(pose_list[:, 1]+480)
ax2.plot(pose_list[:, 2]-500)                           
plt.title('Pose Sensor')
plt.xlabel('Time(s)')


ax3 = plt.subplot(313)
emg_list = abs(emg_list)
ax3.plot(emg_list.T)
plt.title('EMG Sensor')
plt.xlabel('Time(s)')



def calc_impedance(x, fx, y, fy, z, fz):
    """
    计算阻抗参数
    :param x: 位置、速度、加速度数据
    :param y: 力数据
    :return: 阻抗参数
    """
    # 使用最小二乘法计算阻抗参数
    # 使用线性回归拟合参数
    modelx = LinearRegression()
    modelx.fit(x, fx)
    kx, bx, mx = modelx.coef_
    print(f"X轴：刚度 K = {kx:.4f}, 阻尼 B = {bx:.4f}, 惯性 M = {mx:.4f}")

    modely = LinearRegression()
    modely.fit(y, fy)
    ky, by, my = modely.coef_
    print(f"Y轴：刚度 K = {ky:.4f}, 阻尼 B = {by:.4f}, 惯性 M = {my:.4f}")

    modelz = LinearRegression()
    modelz.fit(z, fz)
    kz, bz, mz = modelz.coef_
    print(f"Z轴：刚度 K = {kz:.4f}, 阻尼 B = {bz:.4f}, 惯性 M = {mz:.4f}")

    return kx, ky, kz


x = np.array(pose_list[:, 0]).reshape(-1)/1000
vx = np.gradient(x, 0.01)
ax = np.gradient(vx, 0.01)
X = np.column_stack((x, vx, ax))
fx = np.array(force_list[:, 1]).reshape(-1)


y = np.array(pose_list[:, 1]).reshape(-1)/1000
vy = np.gradient(y, 0.01)
ay = np.gradient(vy, 0.01)
Y = np.column_stack((y, vy, ay))
fy = np.array(force_list[:, 0]).reshape(-1)

z = np.array(pose_list[:, 2]).reshape(-1)/1000
vz = np.gradient(z, 0.01)
az = np.gradient(vz, 0.01)
Z = np.column_stack((z, vz, az))
fz = np.array(force_list[:, 2]).reshape(-1)

kx, ky, kz = calc_impedance(X, fx, Y, fy, Z, fz)

plt.show()

# 等待键盘输入Y，否则退出
if input("Continue? (Y/n)") != 'y':
    exit(0)

# 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge.pkl')

# 保存数据
merge = data[0]
k = data[1]
merge.append(emg_list)
k.append([kx, ky, kz])
# 将merge和k保存为pkl
pd.to_pickle([merge, k], './data/3.8/merge.pkl')
print("数据保存成功！")

# 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge.pkl')