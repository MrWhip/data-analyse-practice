import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("drug200.csv")

class_distribution = data["Drug"].value_counts()
print(class_distribution)

sex_distribution = data["Sex"].value_counts()
print(sex_distribution)

bp_distribution = data["BP"].value_counts()
print(bp_distribution)

cholesterol_distribution = data["Cholesterol"].value_counts()
print(cholesterol_distribution)

plt.figure(figsize=(10, 6))
plt.subplot(2, 2, 1)
plt.bar(class_distribution.index, class_distribution.values)
plt.xlabel("Drug")
plt.ylabel("Count")
plt.subplot(2, 2, 2)
plt.bar(sex_distribution.index, sex_distribution.values)
plt.xlabel("Sex")
plt.ylabel("Count")
plt.subplot(2, 2, 3)
plt.bar(bp_distribution.index, bp_distribution.values)
plt.xlabel("BP")
plt.ylabel("Count")
plt.subplot(2, 2, 4)
plt.bar(cholesterol_distribution.index, cholesterol_distribution.values)
plt.xlabel("Cholesterol")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data["Age"], kde=True)
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(data["Na_to_K"], kde=True)
plt.xlabel("Na_to_K")
plt.ylabel("Count")
plt.show()

encoded_data = data.copy()

encoded_data = pd.get_dummies(encoded_data, columns=["BP", "Cholesterol", "Sex"])

scaler = MinMaxScaler()

data[["Age", "Na_to_K"]] = scaler.fit_transform(data[["Age", "Na_to_K"]])

selected_features = ["Age", "Sex", "BP", "Cholesterol", "Na_to_K"]

subset_data = data[selected_features]

encoded_data = pd.get_dummies(subset_data, columns=["Sex", "BP", "Cholesterol"])

correlation_matrix = encoded_data.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()