import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from Implementation.preprocess import preprocess
from Implementation.KNN import KNearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score

# Reading dataset
dataset = pd.read_excel('./Dataset/dataset.xlsx')

# Preprocess dataset to be used for classification
train_dataset, targets, in_progress_dataset = preprocess(dataset)

# Splitting preprocessed train dataset 30/70
X_train, X_test, y_train, y_test = train_test_split(train_dataset, targets, test_size=0.3, random_state=0)

# Find best value for K via plotting error rate, accuracy and f1-score of sklearn KNN on this dataset
error_rate = []
acc = []
f1_Won = []
f1_Lost = []
for i in range(1, 16, 2):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    predict = knn.predict(X_test)
    error_rate.append(np.mean(predict != y_test))
    acc.append(accuracy_score(y_test, predict))
    f1_Won.append(f1_score(y_test, predict, pos_label='Won'))
    f1_Lost.append(f1_score(y_test, predict, pos_label='Lost'))

# Error rate
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16, 2), error_rate, color='purple', linestyle='dashed',
         marker='8', markerfacecolor='pink', markersize=10)
plt.title('Error Rate / K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
print("Minimum error:-", min(error_rate), "at K =", error_rate.index(min(error_rate)))

# Accuracy
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16, 2), acc, color='green', linestyle='solid',
         marker='D', markerfacecolor='lime', markersize=10)
plt.title('Accuracy / K Value')
plt.xlabel('K')
plt.ylabel('Accuracy')
print("Maximum accuracy:-", max(acc), "at K =", acc.index(max(acc)))

# F1-score of Won
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16, 2), f1_Won, color='blue', linestyle='solid',
         marker='s', markerfacecolor='gray', markersize=10)
plt.title('F1-score of Won labels / K Value')
plt.xlabel('K')
plt.ylabel('F1-score of Won labels')
print("Minimum f1-score:-", min(f1_Won), "at K =", f1_Won.index(min(f1_Won)))

# F1-score of Lost
plt.figure(figsize=(10, 6))
plt.plot(range(1, 16, 2), f1_Lost, color='red', linestyle='dashed',
         marker='p', markerfacecolor='gold', markersize=10)
plt.title('F1-score of Lost labels / K Value')
plt.xlabel('K')
plt.ylabel('F1-score of Lost labels')
print("Maximum f1-score:-", max(f1_Lost), "at K =", f1_Lost.index(max(f1_Lost)))

# Classifying data with our KNN classifier
clf = KNearestNeighbors(3)
clf.fit(X_train, y_train)
mypred = clf.predict(X_test)
mypred = np.array(mypred)

# classifying data with sklearn KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)

# classifying data with sklearn Decision Tree classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)


def classification_data_report(name, y_test_internal, pred_internal):
    """
    A function for reporting some analytical data about classification's performance
    :param name: classification model name
    :param y_test_internal: labels of test data
    :param pred_internal: predicted labels for test data
    :return: None
    """
    print('*** {} Classification Report ***'.format(name), end='\n\n')
    print('Confusion Matrix:\n', confusion_matrix(y_test_internal, pred_internal), end='\n\n')
    print('Classification Report:\n', classification_report(y_test_internal, pred_internal), end='\n\n')
    print('Accuracy:\n', accuracy_score(y_test_internal, pred_internal), end='\n\n')


# Printing classification statistics
classification_data_report('My KNN', y_test, mypred)
classification_data_report('sklearn KNN', y_test, knn_pred)
classification_data_report('sklearn DT', y_test, dt_pred)

# Predicting in-progress transactions results
clf.fit(train_dataset, targets)
predicted_data = clf.predict(in_progress_dataset)
in_progress_dataset['Deal_Stage'] = in_progress_dataset.apply(lambda row: predicted_data[row.level_0], axis=1)
in_progress_dataset.to_csv('prediction.csv')
