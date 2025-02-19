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



def save_data(dataframe, file_name):
    dataframe.to_csv(file_name, index=False)
    pass


def cal(X, Y):
    model = LinearRegression()
    model.fit(X, Y)

    a, b, c, d = model.coef_  # a, b, c, d 是 x1, x2, x3, x4 的系数
    e = model.intercept_  # e 是截距

    # 打印拟合结果
    print(f"拟合结果：a = {a}, b = {b}, c = {c}, d = {d}, e = {e}")

    # 预测 y 值
    y_pred = model.predict(X)

    # 输出拟合的预测值
    # print("预测的 y 值:", y_pred)

    # 计算相关系数
    r2 = model.score(X, Y)
    print("R^2:", r2)


path = 'data/2.17/force/'

csv_file = "data/emg/EMG1739940859697.csv"
csv_data = pd.read_csv(csv_file)#防止弹出警告
csv_df = pd.DataFrame(csv_data)
print(csv_df.shape)

csv_df = csv_df.iloc[:, 1:5]
emg_signal = filter_data(csv_df, f=(20,50), butterworth_order=4, btype='bandpass')
emg_signal = rectify_data(emg_signal)
emg_signal = emg_signal.rolling(200).mean()
csv_df = emg_signal.dropna()
print(csv_df.shape)

plt.figure(figsize=(20, 3))
plt.plot(csv_df.iloc[:,[1]], label='emg1')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
# plt.show()

num_samples = int(len(csv_df) * (100 / 2000))
csv_df = resample(csv_df, num_samples)
csv_df = pd.DataFrame(csv_df)
print(csv_df.shape)

start = 100
end = 1100

# start = 3100
# end = 4100

c1 = csv_df.iloc[:,[0]]
c1 = c1[start:end]
c2 = csv_df.iloc[:,[1]]
c2 = c2[start:end]
c3 = csv_df.iloc[:,[2]]
c3 = c3[start:end]
c4 = csv_df.iloc[:,[3]]
c4 = c4[start:end]



# 绘制图像, figsize=(20, 10)设置图像大小
plt.figure(figsize=(20, 3))
plt.plot(c1, label='emg1')
plt.plot(c2, label='emg2')
plt.plot(c3, label='emg3')
plt.plot(c4, label='emg4')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()

plt.show()


# kx = [243, 203, 176, 138, 103, 88, 62, 38]
# ky = [837, 732, 598, 512, 427, 342, 256, 140]
# kz = [663, 568, 434, 348, 263, 198, 122, 78]

kx = [243, 203, 176, 138, 103, 88, 62, 38]
ky = [837, 732, 598, 512, 427, 342, 256, 140]
kz = [663, 568, 434, 348, 263, 198, 122, 78]

# 每个重复125次，比如
kx1 = [item/2 for item in kx for i in range(125)]
ky1 = [item/2 for item in ky for i in range(125)]
kz1 = [item/2 for item in kz for i in range(125)]

print(kx1)

# Y = force_signal['5']
# Y = StandardScaler().fit_transform(Y)
X = csv_df.iloc[start:end,[0,1,2,3]]
cal(X, kx1)
cal(X, ky1)
cal(X, kz1)


a = 838014.4280499433
b = 1507693.2629336102
c = 523971.4153734604
d = 77018.15021548994
e = 10.074040096367826


custom_coef = np.array([a, b, c, d])
custom_intercept = e
custom_model = LinearRegression()
custom_model.coef_ = custom_coef
custom_model.intercept_ = custom_intercept
y_pred = custom_model.predict(X)
r2 = custom_model.score(X, kx1)
print("R^2:", r2)
# 绘制y_pred和Y的图像
plt.figure(figsize=(20, 3))
plt.plot(y_pred, label='y_pred')
plt.plot(kx1, label='kx1')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()
# plt.show()




a = 572682.873130085
b = 237572.79217439564
c = 2131824.090954872
d = 2336218.0881788656
e = 115.878686020261562


custom_coef = np.array([a, b, c, d])
custom_intercept = e
custom_model = LinearRegression()
custom_model.coef_ = custom_coef
custom_model.intercept_ = custom_intercept
y_pred = custom_model.predict(X)
r2 = custom_model.score(X, ky1)
print("R^2:", r2)
# 绘制y_pred和Y的图像
plt.figure(figsize=(20, 3))
plt.plot(y_pred, label='y_pred')
plt.plot(ky1, label='ky1')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()
# plt.show()


a = 372434.07012185
b = 1220167.8851617225
c = 46693.71400183649
d = 701693.5158285548
e = 4.688445994105313


custom_coef = np.array([a, b, c, d])
custom_intercept = e
custom_model = LinearRegression()
custom_model.coef_ = custom_coef
custom_model.intercept_ = custom_intercept
y_pred = custom_model.predict(X)
r2 = custom_model.score(X, kz1)
print("R^2:", r2)
# 绘制y_pred和Y的图像
plt.figure(figsize=(20, 3))
plt.plot(y_pred, label='y_pred')
plt.plot(kz1, label='kz1')
plt.xlabel('time(ms)')
plt.ylabel('force(N)')
plt.legend()
plt.show()





