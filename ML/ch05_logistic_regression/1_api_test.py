from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    solver='sag',  # 优化算法
    multi_class='multinomial',
    max_iter=1000,
    class_weight='balanced',
    random_state=42,
    penalty='l1',  # 正则化类型
    C=1.0   # 正则化强度
)

# OVR
# 1. 创建LogisticRegression模型
model_ovr1 = LogisticRegression(multi_class='ovr')

from sklearn.multiclass import OneVsRestClassifier
# 2. 创建OneVsRestClassifier模型
model_ovr2 = OneVsRestClassifier(LogisticRegression())

# Softmax 逻辑回归
model_softmax = LogisticRegression(multi_class='multinomial')
 
