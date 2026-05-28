
import json, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = LogisticRegression().fit(X_train, y_train)
acc_a = accuracy_score(y_test, clf.predict(X_test))
model = Sequential([Dense(64, activation='relu', input_shape=(20,)), Dense(1, activation='sigmoid')])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=2, verbose=0)
acc_b = model.evaluate(X_test, y_test, verbose=0)[1]
best_acc = max(acc_a, acc_b)
with open('metrics.json', 'w') as f: json.dump({"accuracy": float(best_acc)}, f)
if acc_b >= acc_a: model.save('best_model.h5')
else: joblib.dump(clf, 'best_model.pkl')
print('train.py complete')
