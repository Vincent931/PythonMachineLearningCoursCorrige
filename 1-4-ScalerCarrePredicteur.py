import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv('advertising.csv')

reg = LinearRegression()

from sklearn.model_selection import train_test_split
X = df[['tv','radio','journaux']]
# Creer la variable quadratique tv^2
df['tv2'] = df.tv**2
variables = ['tv','radio','journaux','tv2']

from sklearn.preprocessing import MinMaxScaler
# Instancier le Scaler
scaler = MinMaxScaler()
# et l'appliquer 
data_array = scaler.fit_transform(df[variables])
df_scaled = pd.DataFrame(data_array, columns = variables)
print(df_scaled.describe().loc[['min','max']])
# scindons les donnees en test et train
X = df_scaled[variables]
y = df.ventes

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

reg.fit(X_train, y_train)
y_hat_test = reg.predict(X_test)

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
# entrainons le modele
reg = LinearRegression()
reg.fit(X_train, y_train)

# Prediction sur le test set
y_pred_test = reg.predict(X_test)

# Scores
print(f"-- Regression ventes ~ radio + journaux + tv + tv^2")
print(f"\tRMSE: {mean_squared_error(y_test, y_pred_test)}")
print(f"\tMAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")

#Sur le même principe, on va comparer les 3 régressions,ventes ~ tv + radio + journaux, ventes ~ tv + radio + journaux + tv^2, ventes ~ tv + radio + journaux + tv * radio
# créons la variable tv * radio
df['tv_radio'] = df.tv * df.radio
regressions = {
    'simple: y ~ tv + radio + journaux'  :     ['tv','radio','journaux'],
    'quadratique: y ~ tv + radio + journaux + tv2': ['tv','radio','journaux', 'tv2'],
    'terme croisée: y ~ tv + radio + journaux + tv*radio':['tv','radio','journaux', 'tv_radio']
}
# la varibale cible est toujours la même
y = df.ventes
# boucle sur les régressions
for title, variables in regressions.items():
    # il faut limiter l'amplitude sur toutes les variables predictrices
    scaler = MinMaxScaler()
    data_array = scaler.fit_transform(df[variables])
    # df_scaled = pd.DataFrame(data_array, columns = variables)
    
    # X = df_scaled[variables]
    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(data_array, y, test_size=0.20, random_state=42)

    # entrainer le modele
    reg.fit(X_train, y_train)
    # Prediction sur le test set
    y_pred_test = reg.predict(X_test)

    # Scores
    print(f"\n-- Regression {title}")
    print(f"\tRMSE: {mean_squared_error(y_test, y_pred_test)}")
    print(f"\tMAPE: {mean_absolute_percentage_error(y_test, y_pred_test)}")