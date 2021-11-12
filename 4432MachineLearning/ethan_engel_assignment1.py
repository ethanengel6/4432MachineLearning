import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def sklearn_to_df(sklearn_dataset):
    data = sklearn_dataset['data']
    cols = sklearn_dataset['feature_names']
    target = sklearn_dataset['target']
    df = pd.DataFrame(data, columns=cols)
    df['target'] = pd.Series(target)
    return df

bunch3 = datasets.load_diabetes()
#print(bunch3['DESCR'])
df3 = sklearn_to_df(bunch3)
chol_array = df3[['s4']].to_numpy()
target_array = df3[['target']].to_numpy()
# I am curious whether total cholesterol is a predictor of diabetes disease progression.  This is general curiosity & also there are personal reasons.

chol_train=chol_array[0:422]
chol_test=chol_array[422:442]
target_train=target_array[0:422]
target_test=target_array[422:442]

lin_reg = LinearRegression()
lin_reg.fit(chol_train, target_train)

diab_preds = lin_reg.predict(chol_train)
print("\nThe first ten predictions:\n", diab_preds[:10],"\n")
print("Feature Coefficient:\n",lin_reg.coef_,"\n")
lin_reg_mse = mean_squared_error(target_train, diab_preds)
print("Root mean squared error of the model:\n",np.sqrt(lin_reg_mse))

plt.scatter(chol_train, target_train, color="black")
plt.plot(chol_train, diab_preds, color='blue', linewidth=2)
plt.xlabel('Total Cholesterol (scaled)')
plt.ylabel('Diabetes Progression Score')
plt.show()
