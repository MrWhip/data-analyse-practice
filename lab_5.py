import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


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

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')

correlation_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:\n", correlation_matrix)

param_grid = {'C': [0.1, 1, 10], 'penalty': ['l1', 'l2']}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("Best Model - Parameters:", grid_search.best_params_)