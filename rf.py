# coding=utf-8
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

'''
首先 X 是十分钟内的数据, y 是对应的攻击类型, 我们按照一下规定去定义
我们采用了 6 个特征, 用于区分 4 种攻击类型

X = [
  0: 服务访问次数 (DDOS 特征)
  1: 请求失败率 (DDOS 特征、入侵特征)
  2: 登录尝试失败次数 (密码播撒和密码爆破)
  3: 端口请求个数 (扫描特征)
  4: 发送内容特殊字符数量 (XSS、SQL 注入特征)
  5: 请求的 URL 长度 (XSS、SQL 注入特征)
]

y = [
  0: 正常访问
  1: 拒绝服务攻击
  2: 扫描攻击
  3: 注入攻击
  4: 密码爆破或密码播撒
]
'''
sample_x = np.array([
  [20000, 0.045219, 23/20000, 12, 8, 447],
  [436, 0.000000, 1/436, 1, 0, 0],
  [3861, 0.983450, 700/861, 1, 3, 36],
  [9864, 0.05143, 9851/9864, 2253, 12, 125],
  [365, 0.000123, 3/365, 10, 215, 1034],
  
  [30000, 0.078321, 33/30000, 15, 10, 450],
  [540, 0.000030, 2/540, 2, 0, 2],
  [6900, 0.990123, 720/900, 2, 4, 40],
  [10000, 0.05234, 9800/10000, 2500, 14, 130],
  [500, 0.000145, 5/500, 12, 220, 1050],
  
  [25000, 0.067890, 27/25000, 18, 9, 460],
  [480, 0.000015, 1/480, 3, 0, 1],
  [8870, 0.975600, 710/870, 3, 5, 42],
  [12500, 0.04987, 9400/9500, 2400, 13, 135],
  [380, 0.000105, 4/380, 11, 210, 1040],
  
  [21000, 0.048234, 25/21000, 14, 7, 440],
  [520, 0.000010, 2/520, 3, 1, 0],
  [2890, 0.981230, 715/890, 3, 4, 39],
  [9700, 0.05000, 9650/9700, 2350, 13, 128],
  [370, 0.000130, 4/370, 11, 218, 1045],
  
  [27000, 0.065000, 28/27000, 17, 8, 470],
  [470, 0.000020, 1/470, 2, 0, 1],
  [1860, 0.976000, 705/806, 2, 3, 43],
  [9200, 0.04890, 9018/9200, 2450, 14, 132],
  [390, 0.000110, 3/390, 10, 212, 1030],
  
  [31000, 0.088765, 34/31000, 19, 10, 450],
  [570, 0.000045, 3/570, 5, 1, 4],
  [5880, 0.985432, 725/880, 4, 7, 37],
  [9800, 0.05123, 9400/9800, 2600, 16, 124],
  [520, 0.000160, 5/520, 14, 230, 1075],
  
  [22000, 0.060987, 26/22000, 13, 6, 455],
  [540, 0.000012, 2/540, 4, 0, 2],
  [4875, 0.983210, 710/875, 3, 5, 35],
  [9500, 0.04765, 9300/9500, 2400, 15, 129],
  [375, 0.000118, 4/375, 12, 220, 1060]
], dtype=np.float32)

sample_y = np.array([
  1, 0, 4, 2, 3, 1, 0, 4, 2, 3, 1, 0, 4, 2, 3, 1, 0, 4, 2, 3, 1, 0, 4, 2, 3, 1, 0, 4, 2, 3, 1, 0, 4, 2, 3
], dtype=np.int32)

# 数据集重复 n 次, 相当于总数乘 2^n 倍
X = sample_x
y = sample_y

multiplier = 5
for i in range(multiplier):
  gaussian = np.random.normal(loc=1.0, scale=0.2, size=X.shape)
  
  X = np.concatenate((X, X*gaussian), axis=0)
  y = np.concatenate((y, y), axis=0)
  
# 查看我们的数据集
print(pd.DataFrame(X), "\n", pd.DataFrame(y))

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pf = pd.DataFrame([str(X_train.shape), str(X_test.shape), str(y_train.shape), str(y_test.shape)], columns=['Shape']).T
pf.columns = ['X_train', 'X_test', 'y_train', 'y_test']
print(pf)

# 初始化随机森林分类器
clf = RandomForestClassifier(n_estimators=1000, random_state=42)

# 训练模型
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)

# 计算准确率
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 输出混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(cm)

# 输出分类报告
report = classification_report(y_test, y_pred)
print('Classification Report:')
print(report)

# 降维到二维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 输出 PCA 贡献度
feature = {
  0: "服务访问数",
  1: "请求失败率",
  2: "登录尝试数",
  3: "端口请求数",
  4: "特殊字符数",  
  5: "请求体长度"
}
print(pd.DataFrame({
  "特征0": pca.components_[0],
  "特征1": pca.components_[1]
}, index=[feature[i] for i in range(6)]))

# 创建子图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 绘制混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')
axes[0].set_title('Confusion Matrix')

# 绘制二维数据可视化图
axes[1].scatter(X_reduced[y == 4, 0], X_reduced[y == 4, 1], c='green', label='Blaster Attack')
axes[1].scatter(X_reduced[y == 3, 0], X_reduced[y == 3, 1], c='purple', label='Injection Attack')
axes[1].scatter(X_reduced[y == 2, 0], X_reduced[y == 2, 1], c='pink', label='Scan Attack')
axes[1].scatter(X_reduced[y == 1, 0], X_reduced[y == 1, 1], c='lightblue', label='DDOS Attack')
axes[1].scatter(X_reduced[y == 0, 0], X_reduced[y == 0, 1], c='red', label='Nomal Access')

axes[1].set_xlabel("Principal Component 1")
axes[1].set_ylabel("Principal Component 2")
axes[1].set_title('2D Visualization of the Data')
axes[1].legend()

# 显示图像
plt.tight_layout()
plt.show()

# 保存模型
model_path = os.path.join('static', 'random_forest_model.pkl')
joblib.dump(clf, model_path)
print("\n模型保存为 {}\n".format(model_path))