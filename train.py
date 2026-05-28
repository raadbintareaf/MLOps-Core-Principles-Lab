import pandas as pd
#标准化
from sklearn.preprocessing import StandardScaler
#一个可以获取很多数据集的函数
from sklearn.datasets import load_breast_cancer
#随机森林模型
from sklearn.ensemble import RandomForestClassifier
#线性回归模型
from sklearn.linear_model import LogisticRegression
#向量机模型
from sklearn.svm import SVC
#XGBoost模型
from xgboost import XGBClassifier # pip install xgboost
#数据集切分
from sklearn.model_selection import train_test_split
#保存模型
import joblib
#metrics指标、度量、评估标准;导入准确率指标
from sklearn.metrics import accuracy_score
#F1-score指标
from sklearn.metrics import f1_score

#建一个结果比较的df
results = pd.DataFrame(columns = ["Model Name","Accuracy","F1 Score"])

# Load Data
#这个数据集是细胞核指标比如半径面积纹理，对应的肿瘤为良性肿瘤1还是恶性肿瘤0
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

#print(X.head())
#print(y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#缩放
scaler = StandardScaler()
#fit是一个功能，相当于使用当前数据集找出均值，标准差便于之后缩放
#transform是一个功能，使用均值，标准差进行缩放
#fit_transform就是先fit再transform，可以分两步也可以一步完成
#只在训练集上进行fit，对测试集只有transform
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model1 Random Forest Model
#设定随机森林的模型参数
random_forest_model = RandomForestClassifier(n_estimators=10, random_state=42)
#使用数据集训练模型
random_forest_model.fit(X_train, y_train)
#计算准确率
#先预测结果，然后用预测的结果和真实的标签比较是否相同；计算准确率；
#真实的标签在前，预测的结果在后；
y_pred = random_forest_model.predict(X_test)
accuracy1 = accuracy_score(y_test,y_pred)
f1_score1 = f1_score(y_test,y_pred)
results.loc[len(results)] = ["random_forest_model", accuracy1, f1_score1]
print("Random Forest Model Accuracy:",accuracy1)
print("Random Forest Model F1-score:",f1_score1)
print("----------------------------------------------")

# Train Model2 Linear Regression Model
#默认迭代次数100会提示迭代次数不足
Logistic_regression_model = LogisticRegression(max_iter=500, random_state=42)
Logistic_regression_model.fit(X_train_scaled,y_train)
y_pred = Logistic_regression_model.predict(X_test_scaled)
accuracy2 = accuracy_score(y_test,y_pred)
f1_score2 = f1_score(y_test,y_pred)
results.loc[len(results)] = ["Logistic_regression_model", accuracy2, f1_score2]
print("Logistic Regression Model Accuracy:",accuracy2)
print("Logistic Regression F1-score:",f1_score2)
print("----------------------------------------------")

#Train Model3 SVC Model
#在默认 probability=False 下，SVC 训练过程几乎是确定性的，random_state 影响很小
Svc_model = SVC()
Svc_model.fit(X_train_scaled,y_train)
y_pred =Svc_model.predict(X_test_scaled)
accuracy3 = accuracy_score(y_test,y_pred)
f1_score3 = f1_score(y_test,y_pred)
results.loc[len(results)] = ["Svc_model", accuracy3, f1_score3]
print("SVC Model Accuracy:",accuracy3)
print("SVC F1-score:",f1_score3)
print("----------------------------------------------")

#Train Model Xgboost Model
Xgboost_model = XGBClassifier(random_state=42)
Xgboost_model.fit(X_train,y_train)
y_pred = Xgboost_model.predict(X_test)
accuracy4 = accuracy_score(y_test,y_pred)
f1_score4 = f1_score(y_test,y_pred)
results.loc[len(results)] = ["Xgboost_model", accuracy4, f1_score4]
print("Xgboost Model Accuracy:",accuracy4)
print("Xgboost F1-score:",f1_score4)
print("----------------------------------------------")

#Train MLP Model前馈神经网络的一种
#30 -> 64(ReLU) -> 32(ReLU) -> 1(Sigmoid) + 标准化 + EarlyStopping
import numpy as np
#用来初始化一个最简单的、一层一层往前堆叠的神经网络容器.
#创建了 model = Sequential() 之后，就可以用 .add() 往里面一节一节地车厢（网络层）往后拼接。
from tensorflow.keras.models import Sequential #pip install tensorflow 
#Dense（全连接层）,最经典的“线性层 + 激活函数”。
#Dropout（随机失活层）;防止模型过拟合.随机去掉小部分神经元。给网络增加训练难度。提升模型在未知测试集上的泛化能力。
from tensorflow.keras.layers import Dense,Dropout
#检查验证集的loss满足规则的话就直接停止
from tensorflow.keras.callbacks import EarlyStopping

#构建MlP，大框架
#Rectified Linear Unit，修正线性单元。
Dl_model = Sequential([
    #传入的input_shape要求是元组，所以有个逗号
    #X_train_scaled.shape也就是特征矩阵X的形状，[0]和[1]分别是样本数量（n）和特征数量(列名)
    Dense(64,activation = "relu",input_shape = (X_train_scaled.shape[1],)),
    Dropout(0.2),#20%
    Dense(32,activation = "relu"),
    Dense(1,activation = "sigmoid")#二分类输出
])

#编译，指定优化过程中的优化器，损失函数之类的
#有一些加方括号的记得
Dl_model.compile(
    optimizer = "adam",
    loss = "binary_crossentropy",
    metrics = ["accuracy"]
)

#早停模型规定
#如果Validation Loss验证损失在8轮还没下降，就停止
#restore_best_weights = True返回到之前历史记录表现最好（历史验证损失最低）的值，而不是结束时的那个值
early_stop = EarlyStopping(
    monitor = "val_loss",
    patience = 8,
    restore_best_weights = True
)

#训练模型
history = Dl_model.fit(
    X_train_scaled,y_train,
    #训练过程中用20%的数据做测试
    validation_split = 0.2,
    epochs = 100,
    callbacks = [early_stop],
    #不打印
    verbose = 0
)

#预测并计算
#ravel拉直，二维变一维
#Keras 默认输出的形状是一个二维矩阵，长这样：
#[[0.12],
# [0.89],
# [0.03]] ->[0.12, 0.89, 0.03]
y_prob = Dl_model.predict(X_test_scaled,verbose = 0).ravel()
#结果是概率，0.5及以上变成1，其余变成0
#astype把true,false变成1，0
y_pred = (y_prob >= 0.5).astype(int)

accuracy5 = accuracy_score(y_test,y_pred)
f1_score5 = f1_score(y_test,y_pred)
results.loc[len(results)] = ["Dl_model", accuracy5, f1_score5]

print("DL Model Accuracy:",accuracy5)
print("DL F1-score:",f1_score5)
print("----------------------------------------------")

#递减排序一下，然后第一个是最大的
results = results.sort_values(["Accuracy","F1 Score"],ascending = False)
#太长了保留5位小数，3位还分不出来
results[["Accuracy", "F1 Score"]] = results[["Accuracy", "F1 Score"]].round(5)
best_row = results.iloc[0]

#保存最优的模型

# 按类型保存赢家模型
best_name = best_row["Model Name"]
if best_name == "Dl_model":
    Dl_model.save(f"{best_name}.h5")
    saved_artifact = f"{best_name}.h5"
else:
    #在全局变量名里找到best_name对应的那个，也就是从字符串转换成变量
    joblib.dump(globals()[best_name], f"{best_name}.pkl")
    saved_artifact =  f"{best_name}.pkl"


winning_model = {
    "model_name": best_name,
    "accuracy": round(float(best_row["Accuracy"]), 3),
    "f1_score": round(float(best_row["F1 Score"]), 3),
    "artifact_path": saved_artifact
}

#把数据导入metrics.json里面
import json

metrics = {
    "winning_model": winning_model, 
    #以行为核心转换df,会把每一行变成一行json，如果默认为转换成列的话，就是每一列为json的一行
    "Rank":results.to_dict(orient = "records")
}
with open ("metrics.json","w",encoding = "utf-8") as f:
    #这是python的json写入的函数，indent是缩进,series不好直接写入所有用default去转换成字符串
    json.dump(metrics,f,indent=2)