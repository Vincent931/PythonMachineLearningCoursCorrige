import pandas as pd
import numpy as np

df = pd.read_csv('age_vs_poids_vs_taille_vs_sexe.csv')

# les variables prédictives
X = df[['sexe','age', 'taille']]

# la variable cible, le poids
y = df.poids

# on choisit un modèle de régression linéaire
from sklearn.linear_model import LinearRegression
reg = LinearRegression()

# on entraîne ce modèle sur les données avec la méthode fit
reg.fit(X.values, y.values)

# on ajoute un enfant garçon de 150 mois et 153 cm
#import numpy as np
#poids = reg.predict(np.array([[0, 150, 153]]))

y_pred = reg.predict(X.values)

# et on obtient directement un score
print(f"R^2 = {np.round(reg.score(X.values, y.values), 3)}")

# ainsi que les coefficients a,b,c de la régression linéaire
print(f"poids = {np.round(reg.coef_[0],  2)} * sexe + {np.round(reg.coef_[1],  2)} * age +  {np.round(reg.coef_[2],  2)} * taille + du bruit")

# prediction du poids pour un garçon agé de 150 mois et de taille 170 cm
reg.predict(np.array([[0, 150, 170]]))

#ici on fait le mse le mae mele mape
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
print("Modele: poids ~ sexe + age + taille")
print(f"\tmean_squared_error(y, y_pred): {mean_squared_error(y, y_pred)}")
print(f"\tmean_absolute_error(y, y_pred): {mean_absolute_error(y, y_pred)}")
print(f"\tmean_absolute_percentage_error(y, y_pred): {mean_absolute_percentage_error(y, y_pred)}")
print()