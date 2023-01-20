import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

data = pd.read_csv("../datasets/9-fraud-detection.csv")
total_transactions = len(data)
normal = len(data[data.Class == 0])
fraudulent = len(data[data.Class == 1])
fraud_percentage = round(fraudulent / normal * 100, 2)
print(f'Total number of Transactions are {total_transactions}')
print(f'Number of Normal Transactions are {normal}')
print(f'Number of fraudulent Transactions are {fraudulent}')
print(f'Percentage of fraud Transactions is {fraud_percentage * 100}%')

X = data.drop('Class', axis=1).values
y = data['Class'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

xgb = XGBClassifier(max_depth=4)
xgb.fit(X_train, y_train)
xgb_yhat = xgb.predict(X_test)

print(f'Accuracy score of the XGBoost model is {(accuracy_score(y_test, xgb_yhat) * 100):.2f} %')
print(f'F1 score of the XGBoost model is {(f1_score(y_test, xgb_yhat) * 100):.2f} %')
