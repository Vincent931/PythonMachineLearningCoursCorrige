from sklearn.datasets import load_breast_cancer
X, y = load_breast_cancer(return_X_y=True)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=808).fit(X, y)

print(X.shape)
print(y)

# split train, test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Prediction d'un échantillon
print(clf.predict([X[8, :]]))
print("Prédiction",clf.predict([X[8, :]])[0])
print(clf.predict_proba([X[8, :]]))
print("Probabilité",clf.predict_proba([X[8, :]])[0][0])

# Prediction d'un autre échantillon
print(clf.predict([X[13, :]]))
print("Prédiction",clf.predict([X[13, :]])[0])
print(clf.predict_proba([X[13, :]]))
print("Probabilité",clf.predict_proba([X[13, :]])[0][1])

#tracer histogramme des probabilités
y_hat_proba = clf.predict_proba(X_test)[:,1]
import seaborn as sns
sns.histplot(y_hat_proba)

import matplotlib.pyplot as plt
plt.title('Histogramme des y en fonction de X')
plt.xlabel('Valeur des Y')
plt.ylabel('Count, nombre de y (prévision)')
plt.show()

# predictions Accuracy
y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("accuracy",accuracy_score(y_test, y_pred))

#matrice de confusion
from sklearn.metrics import confusion_matrix
print("matrice confusion de test\n",confusion_matrix(y_test, y_pred))

####Seuil de séparation des classes
# Probabilité de la classe 1
y_hat_proba = clf.predict_proba(X_test)[:,1]

# classes prédites pour les  seuils 0.3 et 0.7
y_pred_03 = [ 0 if value < 0.3 else 1 for value in y_hat_proba ]
y_pred_07 = [ 0 if value < 0.7 else 1 for value in y_hat_proba ]

# Matrice de confusion pour le seuil 0.3
print("matrice confusion pour seuil de 0.3\n", confusion_matrix(y_test, y_pred_03))

# Matrice de confusion pour le seuil 0.7
print("matrice confusion pour seuil de 0.7\n", confusion_matrix(y_test, y_pred_07))

#####Precision, Recall et ROC_AUC
from sklearn.metrics import precision_score, recall_score, roc_auc_score
print("Precision:",precision_score(y_test, y_pred))
print("Recall:",recall_score(y_test, y_pred))
print("ROC-AUC", roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1]))

from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.plot(fpr, tpr)
plt.grid()
plt.title("ROC curve")
plt.show()

#####classification sur le dataset iris
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=8)

# entrainons le modele
clf = LogisticRegression(random_state=8).fit(X_train, y_train)

# prediction sur le set de test
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)

# Matrice de confusion
print("matrice confusion pour seuil de 0.7\n", confusion_matrix(y_test, y_pred))

#roc_auc
print("ROC-AUC", roc_auc_score(y_test, clf.predict_proba(X_test),multi_class='ovr'))

#classification report donne différents metrics pour chacune des classes (sorties)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))