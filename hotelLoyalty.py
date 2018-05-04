import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

# Read the CSV file
df = pd.read_csv('creditcard.csv')

# Show the contents

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

from sklearn.cross_validation  import train_test_split
X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size = 0.20, random_state=0)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

# print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

true_negative = cm[0,0]
false_positive = cm[0,1]
true_positive = cm[1,1]
false_negative = cm[1,0]
Accuracy_score = (true_negative+true_positive)/(np.sum(cm))*100
print("LogisticRegression\n")
print(Accuracy_score)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()
print(accuracies.mean()*100)


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
print("DecisionTreeClassifier\n")
print(Accuracy_score)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(accuracies.mean()*100)



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

print("DecisionTreeClassifier\n")
print(Accuracy_score)


from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

print(accuracies.mean()*100)



from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
