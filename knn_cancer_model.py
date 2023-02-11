from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

cancer = datasets.load_breast_cancer()
print(cancer.data.shape)
print(cancer.data[0])
print(cancer.target[0:20])

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size = 0.2, random_state = 0)
print('Train Dimension: ', X_train.shape)
print('Test Dimension: ', X_test.shape)

model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Target: ', y_test)
print('Predict: ', y_pred)
print('Accurate: ', model.score(X_test, y_test))

accuracy = []
for K in range(1, 200):
    model = KNeighborsClassifier(n_neighbors= K)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy.append(metrics.accuracy_score(y_test, y_pred))
K = range(1, 200)
plt.plot(K, accuracy)
plt.show()