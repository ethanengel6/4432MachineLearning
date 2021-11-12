import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.datasets import load_diabetes

def sklearn_to_df(sklearn_dataset):
    data = sklearn_dataset['data']
    cols = sklearn_dataset['feature_names']
    target = sklearn_dataset['target']
    df = pd.DataFrame(data, columns=cols)
    df['target'] = pd.Series(target)
    return df

diab_bunch =load_diabetes()
print(diab_bunch['DESCR'])
diab_df = sklearn_to_df(diab_bunch)
print(diab_df.head())
print(diab_df.info())
print(diab_df.describe())

diab_df.hist(bins=30, figsize=(16,9))
plt.show()

diab_train, diab_test= train_test_split(diab_df, test_size=0.2, random_state=42)
diab_cor=diab_df.corr()
print( diab_cor)
ax = sns.heatmap(diab_cor)
plt.show()
sns.pairplot(diab_df,vars=["target", "bmi", "bp","s5"])
plt.show()


diab_target_train = diab_train['target']
diab_train.drop(columns=['target'],inplace=True)
diab_target_test=diab_test['target']
diab_test.drop(columns=['target'],inplace=True)

diab_lin_reg = LinearRegression()
diab_lin_reg.fit(diab_train, diab_target_train)
target_train_pred=diab_lin_reg.predict(diab_train)
diab_rmse=np.sqrt(mean_squared_error(target_train_pred, diab_target_train))
print("RMSE for linear regression model on the training set:",diab_rmse)

diab_DT_reg = DecisionTreeRegressor()
diab_DT_reg.fit(diab_train, diab_target_train)
print("Cross validation scores:")
diab_DTscores = cross_val_score(diab_DT_reg, diab_train, diab_target_train,scoring="neg_mean_squared_error", cv=10)
diab_DTresults = np.sqrt(-diab_DTscores)
print("Decision Tree RMSE:",diab_DTresults.mean(),"\nDecision Tree Standard deviation", diab_DTresults.std())
#significant rise in RMSE

diab_RF_reg = RandomForestRegressor()
diab_RF_reg.fit(diab_train, diab_target_train)
diab_RFscores = cross_val_score(diab_RF_reg, diab_train, diab_target_train,scoring="neg_mean_squared_error", cv=10)
diab_RFresults = np.sqrt(-diab_RFscores)
print("Random Forest RMSE is:",diab_RFresults.mean(),"\nRandom Forest standard deviation:", diab_RFresults.std())
#In this instance, linear regression is the best performing model as measured by RMSE.
print(diab_RF_reg.get_params())

param_grid = [ {'n_estimators':[3,10,30], 'max_features':[2,4,6,8]},
                {'bootstrap':[False], 'n_estimators':[3,10], 'max_features':[2,3,4],} ]
diab_RF_reg2 = RandomForestRegressor()
grid_search = GridSearchCV(diab_RF_reg2, param_grid, cv=5, scoring="neg_mean_squared_error", \
  return_train_score=True)
grid_search.fit(diab_train, diab_target_train)
print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
# The 3rd feature, BMI, had the highest importance rating, which is consistent with its r value from the correlation matrix
# s5 was also 2nd most important from the feature importance grid, as well as the correlation matrix.
final_model = grid_search.best_estimator_
y_FM_pred = final_model.predict(diab_test)
rmse_FM = np.sqrt(mean_squared_error(diab_target_test, y_FM_pred))
print(f'Final Model RMSE = {rmse_FM:.2f}')
print(final_model.get_params())
#The best performing model for assignment2 was the RandomForest Regressor, with the 4 most relevant features included.
#The Linear Regression model from assignment1 yielded an RMSE of 69.5, significantly higher than the RandomForest in assignment2
#Excluding feature variables that are weakly correlated with the target variable helped improve the model.
