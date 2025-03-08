import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.signal as signal
from scipy.signal import resample


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
    model = LinearRegression(positive=True)
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

    correlation_matrix = np.corrcoef(y_pred, Y)
    correlation_coefficient = correlation_matrix[0, 1]
    print("Correlation Coefficient:", correlation_coefficient)

# 读取merge.pkl
data = pd.read_pickle('./data/3.8/merge.pkl')

# 读取数据
emg_data = data[0]
k_data = data[1]


emg_all = np.zeros((0, 4))
k_all = np.zeros((0, 3))
print(len(emg_data))
# 打印k_data的第一列
print(np.array(k_data)[:, 0])

for i in range(len(emg_data)):
    emg = emg_data[i].T
    print(emg.shape)

    emg_signal = filter_data(emg, f=(20,50), butterworth_order=4, btype='bandpass')
    emg_signal = rectify_data(emg_signal)
    emg_signal = emg_signal.rolling(200).mean()
    emg_signal = emg_signal.dropna()
    # print(emg_signal.shape)


    num_samples = int(len(emg_signal) * (100 / 2000))
    emg_signal = resample(emg_signal, num_samples)
    emg_all = np.concatenate((emg_all,emg_signal),axis=0)


    k = k_data[i]
    num = emg_signal.shape[0]
    # 将k重复num次
    k = np.tile(k, (num, 1))
    k_all = np.concatenate((k_all, k), axis=0)


print(emg_all.shape)
print(k_all.shape)


fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(emg_all[:, 0], label='EMG1')
ax1.plot(emg_all[:, 1], label='EMG2')
ax1.plot(emg_all[:, 2], label='EMG3')
ax1.plot(emg_all[:, 3], label='EMG4')
plt.title('EMG Sensor')
plt.xlabel('Time(s)')
plt.ylabel('EMG Value')
plt.legend()
plt.grid()
ax2 = fig.add_subplot(212)
ax2.plot(k_all[:, 0])
plt.title('K')
plt.xlabel('Time(s)')
plt.ylabel('K Value')
plt.grid()
plt.show()

kx = [79.31079154, 69.27259106, 61.10189922, 47.14482155, 50.44457611,
    47.79658523, 113.77702495, 108.41897918, 114.25191396, 47.95932098,
    55.72313824, 106.83812568, 88.40854469, 103.54169181, 144.06978989,
    57.42396714, 40.95469424, 40.52958818, 23.38115215, 19.5019651,
    26.6147746, 23.10835571, 59.47959855, 34.87518607]
