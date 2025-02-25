import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("advertising.csv")
# back to simple regression
X = df[['tv','radio','journaux']]
y = df.ventes

scores = []
reg = LinearRegression()

#faisons varier le random seeed pour des tailles de test vs train set différents
train_test_ratio = [0.2, 0.4, 0.6, 0.8]
random_seeds = [n for n  in  range(0,200,1)]

# Loop progress
from tqdm import tqdm
for ratio in tqdm(train_test_ratio):       
    for seed in random_seeds:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ratio, random_state=seed)
        
        reg.fit(X_train, y_train)
        # Prediction sur le test set
        y_pred_test = reg.predict(X_test)

        scores.append({
            'ratio': ratio,
            'seed': seed, 
            'rmse': mean_squared_error(y_test, y_pred_test)
        })
scores = pd.DataFrame(scores)

#on affiche le tableau
fig = plt.figure(figsize=(10, 6))
sns.boxplot(x = 'ratio', y = 'rmse', hue = 'ratio', data = scores )
sns.despine()
plt.title("Variation du score en fonction de random_state et du % de données d'entraînement")
plt.grid()

plt.tight_layout()
plt.show()