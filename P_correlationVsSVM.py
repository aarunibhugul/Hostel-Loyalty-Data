import numpy as np
import pandas as pd


dataset = pd.read_csv("hotelloyaltydata.csv")
dataset = dataset[["Income","Status","Reedemer"]]
dataset['Reedemer'] = dataset['Reedemer'].replace(["Yes","No"],[1,0])
X = dataset.iloc[:,:-1]
y = dataset.iloc[:,-1]

# for feature Scaling using pearson pearson_corr
# print(pearson_corr_matrix.iloc[:,0].sort_values(axis=0, ascending=False))
# result looks somewhat like this
# Customer Key          1.000000
# Status_Silver         0.074531
# Income_G              0.050077
# Customer Segment_Q    0.035667
# Customer Segment_P    0.034589
# Customer Segment_C    0.030487
# Income_J              0.024495
# Customer Segment_L    0.021828
# Status_Gold           0.021518
# Income_K              0.016974
# Customer Segment_H    0.016612
# Spend                 0.016609



X = pd.get_dummies(X)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
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
