import numpy as np
import pandas as pd


dataset = pd.read_csv("hotelloyaltydata.csv")
dataset = dataset[["Income","Status","Reedemer"]]
dataset['Reedemer'] = dataset['Reedemer'].replace(["Yes","No"],[1,0])
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

X = pd.get_dummies(X)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

for i in range(0.1,1)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

true_negative = cm[0,0]
false_positive = cm[0,1]
true_positive = cm[1,1]
false_negative = cm[1,0]
Accuracy_score = (true_negative+true_positive)/(np.sum(cm))*100
print(Accuracy_score)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(accuracies.mean()*100)
