import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_val_predict, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline

titanic_df = pd.read_csv ('titanic_train.csv')
print(titanic_df.isna().sum())
plt.figure(figsize=(10, 6))
sns.heatmap(titanic_df.isnull(), yticklabels=False, cbar=False)
plt.title("Missing values")
plt.show()
#Categorical variable columns of the dataframe are: Survived, PassengerClass, Sex & Embarked.

titanic_crosstab=pd.crosstab(index=titanic_df['Survived'], columns=titanic_df['Sex'])
rel_crosstab=pd.crosstab(index=titanic_df['Survived'], columns=titanic_df['Sex'],normalize='columns')
print("\n",titanic_crosstab,"\n",rel_crosstab)
#More females than males survived, in terms of raw numbers(233 vs. 109), and especially proportianally (74% of females vs. 19% of males).
class_crosstab=pd.crosstab(index=titanic_df['Survived'], columns=titanic_df['Pclass'],normalize='columns')
print("\n",class_crosstab)
#1st class passengers survived at a rate of 67%, followed by 2nd(47%) and 3rd (24%).

sns.boxplot(data=titanic_df,x="Fare")
plt.show()
#The distribution of fare is heavily skewed to the right.  While the median is under 50pounds, there are several
# upper outliers, including one nearly 500 pound datapoint.
sns.boxplot(data=titanic_df,x="Age")
plt.show()
#The distribution of non-outlier values in the Age category is realtively symmetric, although, again, there are several upper outliers.
print(titanic_df.groupby(["Pclass"])["Age"].median())
sns.boxplot(data=titanic_df,x="Pclass",y="Age")
plt.show()

titanic_df.drop(columns=['Cabin'],inplace=True)
titanic_df['Age'] = titanic_df.groupby(['Pclass'], sort=False)['Age'].apply(lambda x: x.fillna(x.median()))
titanic_df.dropna(inplace=True)
print(titanic_df.isna().sum())
titanic_df=pd.get_dummies(titanic_df, columns=['Pclass','Sex', 'Embarked'])
titanic_label=titanic_df['Survived']
titanic_df.drop(columns=['Name','PassengerId','Ticket','Survived'],inplace=True)

titanic_df.to_csv('titanic_clean.csv')
titanic_label.to_csv('titanic_label.csv')
#Part3
titanic_lr=LogisticRegression()
titanic_lr.set_params(solver='lbfgs',max_iter=1000)
titanic_lr.fit(titanic_df, titanic_label)
lr_predictions=titanic_lr.predict(titanic_df)
lr_classification_report= classification_report(titanic_label, lr_predictions)
lr_cm=confusion_matrix(titanic_label, lr_predictions)
lr_roc=roc_auc_score(titanic_label, lr_predictions)
fpr, tpr, thresholds =roc_curve(titanic_label, lr_predictions)
print("\nLogistic Regression:\n\nClassification Report\n", lr_classification_report, "\nConfusion Matrix:\n", lr_cm, "\n\nROC score\n", lr_roc,"\n")

titanic_svc=SVC(kernel="poly",degree=3, coef0=1,C=5,probability=True)
titanic_svc.fit(titanic_df, titanic_label)
svc_predictions=titanic_svc.predict(titanic_df)
svc_classification_report= classification_report(titanic_label, svc_predictions)
svc_cm=confusion_matrix(titanic_label, svc_predictions)
svc_roc=roc_auc_score(titanic_label, svc_predictions)
fpr2, tpr2, thresholds =roc_curve(titanic_label, svc_predictions)
print("\nSupport Vector Classifier :\n\nClassification Report\n", svc_classification_report, "\nConfusion Matrix:\n", svc_cm, "\n\nROC score\n", svc_roc,"\n")

titanic_sgd=SGDClassifier()
titanic_sgd.fit(titanic_df, titanic_label)
sgd_predictions=titanic_sgd.predict(titanic_df)
sgd_classification_report= classification_report(titanic_label, sgd_predictions)
sgd_cm=confusion_matrix(titanic_label, sgd_predictions)
sgd_roc=roc_auc_score(titanic_label, sgd_predictions)
fpr3, tpr3, thresholds =roc_curve(titanic_label, sgd_predictions)
print("\nStochastic Gradient Descent Classifier :\n\nClassification Report\n", sgd_classification_report, "\nConfusion Matrix:\n", sgd_cm, "\n\nROC score\n", sgd_roc,"\n")

titanic_svc_scaled=Pipeline([("scaler",StandardScaler()),("titanic_svc2",SVC(kernel="poly",degree=3, coef0=1,C=3,probability=True))])
titanic_svc_scaled.fit(titanic_df, titanic_label)
svc_scaled_predictions=titanic_svc_scaled.predict(titanic_df)
svc_scaled_classification_report= classification_report(titanic_label, svc_scaled_predictions)
svc_scaled_cm=confusion_matrix(titanic_label, svc_scaled_predictions)
svc_scaled_roc=roc_auc_score(titanic_label, svc_scaled_predictions)
fpr4, tpr4, thresholds =roc_curve(titanic_label, svc_scaled_predictions)
print("\nSupport Vector Classifier with scaled data:\n\nClassification Report\n", svc_scaled_classification_report, "\nConfusion Matrix:\n", svc_scaled_cm, "\n\nROC score\n", svc_scaled_roc,"\n")
#Scaling input data for the SVC classifier consistently yields a higher ROC score than with unscaled data.(approx. .81  vs .75)

plt.figure()
plt.plot(fpr, tpr, color='darkorange',lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc)
plt.plot(fpr2, tpr2, color='m',lw=2, label='SVC Classifier (area = %0.2f)' % svc_roc)
plt.plot(fpr3, tpr3, color='k',lw=2, label='SGD Classifier (area = %0.2f)' % sgd_roc)
plt.plot(fpr4, tpr4, color='c',lw=2, label='SGD Classifier(scaled inputs) (area = %0.2f)' % svc_scaled_roc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

titanic_svc_pipe=Pipeline([("scaler",StandardScaler()),("pipe_svc",SVC(max_iter=2000))])
svc_params = {'pipe_svc__kernel':["linear"],'pipe_svc__gamma': [0.0001, 0.001,0.01, 0.1, 1], 'pipe_svc__C': [1,10,50,100,200,300]}
gs = GridSearchCV(titanic_svc_pipe, svc_params,scoring="roc_auc",cv=5,n_jobs=-1)
gs.fit(titanic_df, titanic_label)
print(gs.best_params_)
best_model=gs.best_estimator_
print(best_model)
print(gs.best_score_)

train_sizes, train_scores, test_scores = learning_curve(best_model, titanic_df, titanic_label, cv=5,n_jobs=-1)
plt.plot(train_sizes, np.mean(train_scores,axis=1), color='red', label='Training Accuracy')
plt.plot(train_sizes, np.mean(test_scores,axis=1), color='green', linestyle='--', label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Training Examples')
plt.ylabel('Score')
plt.legend(loc="lower right")
plt.show()
#The training and validation curves do converge, so it seems there was no overfitting.  A slight cause for concern
#is that the two curves may be diverging after approximately 550 training examples.  Perhaps the training should
#be ceased at this point.
