import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 假设你有位置、速度、加速度和力的数据
# 示例数据
n = 100
time = np.linspace(0, 10, n)
position = np.sin(time)  # 假设的位置数据
velocity = np.gradient(position, time)  # 计算速度
acceleration = np.gradient(velocity, time)  # 计算加速度
force = 5 * position + 2 * velocity + 3 * acceleration + np.random.normal(0, 0.1, n)  # 力数据，带有一些噪声

# 组织数据矩阵X
X = np.column_stack((position, velocity, acceleration))

# 使用线性回归拟合参数
model = LinearRegression()
model.fit(X, force)

# 获取拟合的阻抗参数
K, B, M = model.coef_

print(f"拟合的阻抗参数：")
print(f"刚度 K = {K:.4f}")
print(f"阻尼 B = {B:.4f}")
print(f"惯性 M = {M:.4f}")

# 可视化结果：预测的力与实际力
predicted_force = model.predict(X)

plt.plot(time, force, label="Force_real")
plt.plot(time, predicted_force, label="Force_pred", linestyle='--')
plt.xlabel("Time(s)")
plt.ylabel("Force(N)")
plt.legend()
plt.title("Force Prediction")
plt.show()


# import numpy as np
# from sklearn.linear_model import LinearRegression
# import matplotlib.pyplot as plt

# # 假设你有三维的位移、速度、加速度和力的数据
# # 示例数据
# n = 100
# time = np.linspace(0, 10, n)
# position = np.column_stack((np.sin(time), np.cos(time), np.sin(2*time)))  # 三维位置数据
# velocity = np.gradient(position, time, axis=0)  # 计算速度（在三维空间）
# acceleration = np.gradient(velocity, time, axis=0)  # 计算加速度（在三维空间）

# # 假设的力数据 (三维力)
# force = 5 * position + 2 * velocity + 3 * acceleration + np.random.normal(0, 0.1, (n, 3))

# # 组织数据矩阵X（包括位置、速度、加速度）
# X = np.column_stack((position, velocity, acceleration))

# # 使用线性回归拟合参数
# model = LinearRegression()
# model.fit(X, force)

# # 获取拟合的阻抗参数 (刚度、阻尼、惯性)
# # model.coef_的形状是 (3, 9)，每行是一个方向的系数
# K, B, M = model.coef_

# # 输出每个方向的刚度、阻尼、惯性
# print("拟合的阻抗参数：")
# for i, direction in enumerate(['x', 'y', 'z']):
#     print(f"方向 {direction}:")
#     print(f"  刚度 K = {K[i]:.4f}")
#     print(f"  阻尼 B = {B[i]:.4f}")
#     print(f"  惯性 M = {M[i]:.4f}")

# # 可视化结果：每个方向的力拟合
# fig, axes = plt.subplots(3, 1, figsize=(8, 8))

# for i, direction in enumerate(['x', 'y', 'z']):
#     # 预测的力
#     predicted_force = model.predict(X)[:, i]
    
#     axes[i].plot(time, force[:, i], label="Force_real")
#     axes[i].plot(time, predicted_force, label="Force_pred", linestyle='--')
#     axes[i].set_xlabel("Time(s)")
#     axes[i].set_ylabel(f"Force(N)")
#     axes[i].legend()
#     axes[i].set_title(f"{direction}-direction Force Prediction")

# plt.tight_layout()
# plt.show()



