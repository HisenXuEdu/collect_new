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
data = pd.read_pickle('./data/3.8/merge_make1.pkl')

# 读取数据
emg_data = data[0]
k_data = data[1]


emg_all = np.zeros((0, 4))
k_all = np.zeros((0, 3))
print(len(emg_data))

for i in range(len(emg_data)):
    emg = emg_data[i].T
    print(emg.shape)

    emg_signal = filter_data(emg, f=(20,50), butterworth_order=4, btype='bandpass')
    emg_signal = rectify_data(emg_signal)
    emg_signal = emg_signal.rolling(200).mean()
    emg_signal = emg_signal.dropna()
    print(emg_signal.shape)


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
ax1.plot(emg_all)
plt.title('EMG Sensor')
plt.xlabel('Time(s)')
plt.ylabel('EMG Value')
plt.grid()
ax2 = fig.add_subplot(212)
ax2.plot(k_all)
plt.title('K')
plt.xlabel('Time(s)')
plt.ylabel('K Value')
plt.grid()
# plt.show()



kx1 = k_all[:, 0]
ky1 = k_all[:, 1]
kz1 = k_all[:, 2]

print(kx1.shape)
print(k_all.shape)
print(emg_all.shape)

X = pd.DataFrame(emg_all)
X = X.iloc[:, [0, 1, 2, 3]]



cal(X, kx1)
cal(X, ky1)
cal(X, kz1)

plt.show()



a = 838014.4280499433
b = 1507693.2629336102
c = 523971.4153734604
d = 77018.15021548994
e = 10.074040096367826


custom_coef = np.array([a, b, c, d])
custom_intercept = e
custom_model = LinearRegression(positive=True)
custom_model.coef_ = custom_coef
custom_model.intercept_ = custom_intercept
y_pred = custom_model.predict(X)


correlation_matrix = np.corrcoef(y_pred, kx1)
correlation_coefficient = correlation_matrix[0, 1]
print("Correlation Coefficient:", correlation_coefficient)

# r2 = custom_model.score(X, kx1)
# print("R^2:", r2)
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

correlation_matrix = np.corrcoef(y_pred, ky1)
correlation_coefficient = correlation_matrix[0, 1]
print("Correlation Coefficient:", correlation_coefficient)

# r2 = custom_model.score(X, ky1)
# print("R^2:", r2)
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

correlation_matrix = np.corrcoef(y_pred, kz1)
correlation_coefficient = correlation_matrix[0, 1]
print("Correlation Coefficient:", correlation_coefficient)