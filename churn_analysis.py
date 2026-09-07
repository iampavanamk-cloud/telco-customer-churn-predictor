import pandas as pd
df = pd.read_csv("Telco-Customer-Churn.csv")
print(df.head())
print(df.info())
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print(df['TotalCharges'].isnull().sum())
df['TotalCharges'] = df['TotalCharges'].fillna(0)
print(df['Churn'].value_counts())
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Contract", hue="Churn", palette="Set2")
plt.title("Customer Churn Count by Contract Type", fontsize=14, fontweight='bold')
plt.xlabel("Contract Type", fontsize=12)
plt.ylabel("Number of Customers", fontsize=12)
plt.legend(title="Churn")
plt.savefig("churn_by_contract.png")
plt.show()
from sklearn.model_selection import train_test_split
df_model = df.drop('customerID', axis=1)
df_model['Churn'] = df_model['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
df_model = pd.get_dummies(df_model, drop_first=True)
X = df_model.drop('Churn', axis=1)
y = df_model['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training features shape: {X_train.shape}")
print(f"Testing features shape: {X_test.shape}")
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_test, y_pred))
import matplotlib.pyplot as plt
import pandas as pd
importances = rf_model.feature_importances_
feature_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(10)
plt.figure(figsize=(10, 6))
plt.barh(feature_imp_df['Feature'], feature_imp_df['Importance'], color='teal')
plt.xlabel("Importance Score", fontsize=12)
plt.ylabel("Features", fontsize=12)
plt.title("Top 10 Features Driving Customer Churn", fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.savefig("feature_importance.png")
plt.show()