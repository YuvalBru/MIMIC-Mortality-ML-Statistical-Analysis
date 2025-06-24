import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,brier_score_loss
)
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import calibration_curve


data = pd.read_csv('./data/boosting_models_data.csv')

#Splitting into target variable 30-day expire flag (what we want to predict) and features which will predict it
target = 'thirtyday_expire_flag'

x = data.drop(columns=[target])
y = data[target]

#Splitting for training data and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

#Defining the model
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)
#Training the model
model.fit(x_train, y_train)


#Using the test set to evaluate the model
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Survived', 'Expired'],
            yticklabels=['Survived', 'Expired'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#Saving the model
joblib.dump(model, './models/random_forest_model.joblib')

#Extracting feature importance
importances = model.feature_importances_
feature_names = x.columns
feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(10)


#Plotting the feature importance
plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=feat_imp_df)
plt.title("Top 10 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()

#CALIBRATION of the model.
prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Random Forest')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
plt.title('Calibration Curve')
plt.xlabel('Mean Predicted Probability')
plt.ylabel('Fraction of Positives')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


print("Brier Score:", brier_score_loss(y_test, y_proba))
