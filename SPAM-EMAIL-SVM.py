import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv('datasets/2-spam_emails.csv')
X = data['EmailText'].values
y = data['Label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

cv = CountVectorizer()
X_train = cv.fit_transform(X_train)
X_test = cv.transform(X_test)

classifier = SVC()
classifier.fit(X_train, y_train)
accuracy = classifier.score(X_test, y_test)
print(f"Accuracy: {accuracy*100} %")
