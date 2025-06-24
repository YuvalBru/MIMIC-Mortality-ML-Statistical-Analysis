import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, brier_score_loss
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.calibration import calibration_curve

data = pd.read_csv("./data/clustering_data_normalized_encoded.csv")

#Splitting the data into target variable and the rest (target variable is the variable we want to predict)
target = 'thirtyday_expire_flag'


x = data.drop(columns=[target])
y = data[target]

#Splitting the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

#Defining the model
model = LogisticRegression(
    penalty='l2',
    solver='liblinear',
    max_iter=1000,
    class_weight='balanced',
    random_state=42
)
#Training the model
model.fit(x_train, y_train)

#Evaluating the model using the test set
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_proba))
print(classification_report(y_test, y_pred))

#plotting confusion matrix
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
joblib.dump(model, './models/logistic_regression_model.joblib')


#Extracting top features
odds_ratios = np.exp(model.coef_[0])
top_features = pd.Series(odds_ratios, index=x.columns).sort_values(ascending=False).head(10)
print(top_features)

#Calibration of the model
brier = brier_score_loss(y_test, y_proba)
print(f"Brier Score: {brier:.6f}")

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
