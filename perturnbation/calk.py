import numpy as np
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import resample
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy import signal

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

    #绘制图像
    predicted_fx = modelx.predict(x)
    predicted_fy = modely.predict(y)
    predicted_fz = modelz.predict(z)
    fig = plt.figure(figsize=(18, 8))
    plt.subplot(311)
    plt.plot(fx, label="Fx_real")
    plt.plot(predicted_fx, label="Fx_pred", linestyle='--')
    plt.legend()
    plt.subplot(312)
    plt.plot(fy, label="Fy_real")
    plt.plot(predicted_fy, label="Fy_pred", linestyle='--')
    plt.legend()
    plt.subplot(313)
    plt.plot(fz, label="Fz_real")
    plt.plot(predicted_fz, label="Fz_pred", linestyle='--')
    plt.legend()
    # plt.show()

    return modelx, modely, modelz

# def calp():



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


# 绘制图像,一张图上由3个子图组成
fig = plt.figure(figsize=(18, 8))

plt.rcParams.update({'font.size': 12})
ax1 = fig.add_subplot(311)

x0 = pose_signal.iloc[70:820,[0]]
y0 = pose_signal.iloc[70:820,[1]]
z0 = pose_signal.iloc[70:820,[2]]

x1 = pose_signal.iloc[1000:1750,[0]]
y1 = pose_signal.iloc[1000:1750,[1]]
z1 = pose_signal.iloc[1000:1750,[2]]

x2 = pose_signal.iloc[2000:2750,[0]]
y2 = pose_signal.iloc[2000:2750,[1]]
z2 = pose_signal.iloc[2000:2750,[2]]

x = np.concatenate((x0, x1, x2), axis=0)
y = np.concatenate((y0, y1, y2), axis=0)
z = np.concatenate((z0, z1, z2), axis=0)

ax1.plot(x, label='x')
ax1.plot(y, label='y')
ax1.plot(z, label='z')
plt.ylabel('Pose(mm)', fontsize=16)
plt.legend()
plt.xticks([])
# 网格
plt.grid(True)


ax2 = fig.add_subplot(312)

fx0 = force_signal.iloc[:750,[0]]
# fx = StandardScaler().fit_transform(fx)
fy0 = force_signal.iloc[:750,[1]]
# fy = StandardScaler().fit_transform(fy)
fz0 = force_signal.iloc[:750,[2]]
# fz = StandardScaler().fit_transform(fz)

fx1 = force_signal.iloc[1000:1750,[0]]
# fx1 = StandardScaler().fit_transform(fx1)*3
fy1 = force_signal.iloc[1000:1750,[1]]
# fy1 = StandardScaler().fit_transform(fy1)*3
fz1 = force_signal.iloc[1000:1750,[2]]
# fz1 = StandardScaler().fit_transform(fz1)*3

fx2 = force_signal.iloc[2000:2750,[0]]
# fx2 = StandardScaler().fit_transform(fx2)*5
fy2 = force_signal.iloc[2000:2750,[1]]
# fy2 = StandardScaler().fit_transform(fy2)*5
fz2 = force_signal.iloc[2000:2750,[2]]
# fz2 = StandardScaler().fit_transform(fz2)*5

# 拼接fx和fx1
fx = np.concatenate((fx0, fx1, fx2), axis=0)
fy = np.concatenate((fy0, fy1, fy2), axis=0)
fz = np.concatenate((fz0, fz1, fz2), axis=0)

ax2.plot(fx, label='x')
ax2.plot(fy, label='y')
ax2.plot(fz, label='z')
# 设置ylabel距离y轴的距离
plt.ylabel('Force(N)', fontsize=16)
# 只显示横向网格
plt.grid(True, axis='y')
# 右上角显示图例
plt.legend(loc='upper right')

plt.show()


# 将x变成一行
x = np.array(x0).reshape(-1)/1000
vx = np.gradient(x, 0.01)
ax = np.gradient(vx, 0.01)
X = np.column_stack((x, vx, ax))
fx = np.array(fx0).reshape(-1)

y = np.array(y0).reshape(-1)/1000
vy = np.gradient(y, 0.01)
ay = np.gradient(vy, 0.01)
Y = np.column_stack((y, vy, ay))
fy = np.array(fy0).reshape(-1)

z = np.array(z0).reshape(-1)/1000
vz = np.gradient(z, 0.01)
az = np.gradient(vz, 0.01)
Z = np.column_stack((z, vz, az))
fz = np.array(fz0).reshape(-1)
print("1")
model = calc_impedance(X, fx, Y, fy, Z, fz)




# 将x变成一行
x = np.array(x1).reshape(-1)/1000
vx = np.gradient(x, 0.01)
ax = np.gradient(vx, 0.01)
X = np.column_stack((x, vx, ax))
fx = np.array(fx1).reshape(-1)

y = np.array(y1).reshape(-1)/1000
vy = np.gradient(y, 0.01)
ay = np.gradient(vy, 0.01)
Y = np.column_stack((y, vy, ay))
fy = np.array(fy1).reshape(-1)

z = np.array(z1).reshape(-1)/1000
vz = np.gradient(z, 0.01)
az = np.gradient(vz, 0.01)
Z = np.column_stack((z, vz, az))
fz = np.array(fz1).reshape(-1)
print("2")
model = calc_impedance(X, fx, Y, fy, Z, fz)



# 将x变成一行
x = np.array(x2).reshape(-1)/1000
vx = np.gradient(x, 0.01)
ax = np.gradient(vx, 0.01)
X = np.column_stack((x, vx, ax))
fx = np.array(fx2).reshape(-1)

y = np.array(y2).reshape(-1)/1000
vy = np.gradient(y, 0.01)
ay = np.gradient(vy, 0.01)
Y = np.column_stack((y, vy, ay))
fy = np.array(fy2).reshape(-1)

z = np.array(z2).reshape(-1)/1000
vz = np.gradient(z, 0.01)
az = np.gradient(vz, 0.01)
Z = np.column_stack((z, vz, az))
fz = np.array(fz2).reshape(-1)
print("3")
model = calc_impedance(X, fx, Y, fy, Z, fz)



emg_signal = filter_data(emg_signal, f=(20,50), butterworth_order=4, btype='bandpass')
emg_signal = rectify_data(emg_signal)
emg_signal = emg_signal.rolling(200).mean()
emg_signal = emg_signal.dropna()
print(emg_signal.shape)
num_samples = int(len(emg_signal) * (100 / 2000))
emg_signal = resample(emg_signal, num_samples)
emg_signal = pd.DataFrame(emg_signal)

e0 = emg_signal.iloc[740:750,[0,1,2,3]]
e1 = emg_signal.iloc[1740:1750,[0,1,2,3]]
e2 = emg_signal.iloc[2740:2750,[0,1,2,3]]
e0 = e0.append(e1)
e0 = e0.append(e2)
e0 = e0.reset_index(drop=True)

Kx0 = 10*[198.7361]
Kx1 = 10*[243.1391]
kx2 = 10*[284]

# 拼接kx0,kx1,kx2  
Kx = Kx0 + Kx1 + kx2


# 拼接kx0
Y = force_signal['5']
# Y = StandardScaler().fit_transform(Y)
X = emg_signal.iloc[:,[0,1,2,3]]
# 将
model = LinearRegression()
model.fit(e0, Kx)
kx = model.predict(e0)


a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
e = model.intercept_  # e 是截距

# 打印拟合结果
print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")




# # 假设你有位置、速度、加速度和力的数据
# # 示例数据
# n = 100
# time = np.linspace(0, 10, n)
# position = np.sin(time)  # 假设的位置数据
# velocity = np.gradient(position, time)  # 计算速度
# acceleration = np.gradient(velocity, time)  # 计算加速度
# force = 5 * position + 2 * velocity + 3 * acceleration + np.random.normal(0, 0.1, n)  # 力数据，带有一些噪声

# # 组织数据矩阵X
# X = np.column_stack((position, velocity, acceleration))

# # 使用线性回归拟合参数
# model = LinearRegression()
# model.fit(X, force)

# # 获取拟合的阻抗参数
# K, B, M = model.coef_

# print(f"拟合的阻抗参数：")
# print(f"刚度 K = {K:.4f}")
# print(f"阻尼 B = {B:.4f}")
# print(f"惯性 M = {M:.4f}")

# # 可视化结果：预测的力与实际力
# predicted_force = model.predict(X)

# plt.plot(time, force, label="Force_real")
# plt.plot(time, predicted_force, label="Force_pred", linestyle='--')
# plt.xlabel("Time(s)")
# plt.ylabel("Force(N)")
# plt.legend()
# plt.title("Force Prediction")
# plt.show()