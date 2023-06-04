import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


data = pd.read_csv("drug200.csv")

encoded_data = data.copy()

label_encoder = LabelEncoder()
encoded_data['Sex'] = label_encoder.fit_transform(encoded_data['Sex'])
encoded_data['BP'] = label_encoder.fit_transform(encoded_data['BP'])
encoded_data['Cholesterol'] = label_encoder.fit_transform(encoded_data['Cholesterol'])
encoded_data['Drug'] = label_encoder.fit_transform(encoded_data['Drug'])

X = encoded_data.drop('Drug', axis=1)
y = encoded_data['Drug']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model1 = DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3 = SVC()
model4 = LogisticRegression()

ensemble = VotingClassifier(estimators=[('dt', model1), ('knn', model2), ('svc', model3), ('lr', model4)])

ensemble.fit(X_train, y_train)

y_pred = ensemble.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

correlation_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", correlation_matrix)

param_grid = {}

grid_search = GridSearchCV(estimator=ensemble, param_grid=param_grid, scoring='accuracy', cv=5)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Model - Parameters:", best_model)