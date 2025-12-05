import numpy as np
from sklearn.linear_model import LogisticRegression

filename = r"horseColicTest.txt"

#=====================
# 1. 数据读取函数
#=====================
def load_dataset(filename):
    data = np.loadtxt(filename)
    X = data[:, :-1]   # 特征
    y = data[:, -1]    # 标签
    return X, y

#=====================
# 2. 缺失值处理函数
#   （缺失值替换为该列均值）
#=====================
def replace_nan_with_mean(X):
    for i in range(X.shape[1]):
        col = X[:, i]
        # 选择非0的数作为有效特征
        valid = col[col != 0]
        if len(valid) > 0:
            mean_val = np.mean(valid)
            col[col == 0] = mean_val
            X[:, i] = col
    return X

#=====================
# 3. 主流程
#=====================
# 读取训练集
train_file = "horseColicTraining.txt"  # 训练集文件名
X_train, y_train = load_dataset(train_file)
X_train = replace_nan_with_mean(X_train)

# 读取测试集
test_file = "horseColicTest.txt"  # 测试集文件名
X_test, y_test = load_dataset(test_file)
X_test = replace_nan_with_mean(X_test)

#=====================
# 4. 构建并训练逻辑回归模型
#=====================
# 创建逻辑回归模型，调整max_iter防止不收敛警告
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#=====================
# 5. 测试集预测
#=====================
y_pred = model.predict(X_test)

#=====================
# 6. 计算准确率
#=====================
accuracy = np.mean(y_pred == y_test) * 100
print(f"测试集准确率: {accuracy:.2f}%")

# from sklearn.metrics import classification_report, confusion_matrix
#
# print("\n分类报告:")
# print(classification_report(y_test, y_pred))
#
# print("混淆矩阵:")
# print(confusion_matrix(y_test, y_pred))
#
# y_pred_prob = model.predict_proba(X_test)
# print(f"\n前5个样本的预测概率:")
# for i in range(5):
#     print(f"样本{i+1}: 类别0概率={y_pred_prob[i][0]:.4f}, 类别1概率={y_pred_prob[i][1]:.4f}, 真实标签={y_test[i]}")