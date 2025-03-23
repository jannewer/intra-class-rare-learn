"""
============================================
Fitting and evaluating an ICR Random Forest
============================================

In this example we fit and evaluate an
:class:`icrlearn.ICRRandomForestClassifier` on the Iris dataset.
"""

# %%
# Load the Iris dataset and split it into training and test sets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# %%
# Train an ICRRandomForestClassifier on the training set
from icrlearn import ICRRandomForestClassifier

clf = ICRRandomForestClassifier().fit(X_train, y_train)

# %%
# Evaluate the classifier on the test set and print some metrics
from sklearn.metrics import classification_report

y_pred = clf.predict(X_test)
classification_report = classification_report(y_test, y_pred)
print(classification_report)

# %%
# Plot a confusion matrix of the classifier's predictions
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

conf_matr = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matr, annot=True)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
