from sklearn.linear_model import LinearRegression
import numpy as np

# 自变量 每周学习时长
X = [[5],[8],[10],[12],[15],[3],[7],[9],[14],[6]]
# 因变量，数学考试成绩
y = [55, 65, 70, 75, 85, 50, 60, 72, 80, 50]

#2 创建模型
model = LinearRegression()

#3 模型训练
model.fit(X,y)

#4 打印模型参数
print(model.coef_)
print(model.intercept_)

#5 预测
x_new = [[11]]
y_pred = model.predict(x_new)
print(y_pred)

#6 画图
import matplotlib.pyplot as plt
x_line = np.arange(0,15,0.1).reshape(-1,1)
y_line = model.predict(x_line)
plt.scatter(X,y)
plt.plot(x_line, y_line,color='red')
plt.scatter(x_new,y_pred,color='green')
plt.show()

# 用方差和协方差验证数学求解公式

x = np.array(X).reshape(-1)
cov = np.cov(x,y) #和协方差
print(cov)

bata1 = cov[0][1] / cov[0][0]
print(bata1)

model = LinearRegression(fit_intercept=False)
model.fit(X,y)

print(model.intercept_)
print(model.coef_)

#5 预测
x_new = [[11]]
y_pred = model.predict(x_new)
print(y_pred)

#6 画图
import matplotlib.pyplot as plt
x_line = np.arange(0,15,0.1).reshape(-1,1)
y_line = model.predict(x_line)
plt.scatter(X,y)
plt.plot(x_line, y_line,color='red')
plt.scatter(x_new,y_pred,color='green')
plt.show()
