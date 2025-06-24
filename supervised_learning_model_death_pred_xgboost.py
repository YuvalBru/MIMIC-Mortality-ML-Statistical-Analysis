import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report, brier_score_loss
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.calibration import calibration_curve

data = pd.read_csv('./data/boosting_models_data.csv')

#We divide the data to target variable(what we want to predict) and the features which
#predict it
target = 'thirtyday_expire_flag'
x = data.drop(columns = [target])
y = data[target]

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size= 0.2, random_state= 42, stratify= y)

#Defining our XGBoost model
model = XGBClassifier(n_estimators = 1000,
                                      max_depth = 6, learning_rate = 0.01,
eval_metric ='logloss',
random_state = 42, early_stopping = 50
)
#Training the model
model.fit(x_train,y_train)

#Discrimination ( Testing the test set and geting our metrics)
y_pred = model.predict(x_test)
y_proba = model.predict_proba(x_test)[:, 1]
print(f"The test's accuracy is: {accuracy_score(y_test,y_pred)}")
print(f"The test's precision is: {precision_score(y_test,y_pred)}")
print(f"The test's F1-Score is: {f1_score(y_test,y_pred)}")
print(f"The test's ROC AUC is: {roc_auc_score(y_test,y_proba)}")
print(classification_report(y_test,y_pred))

#plotting confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Expired'], yticklabels=['Survived', 'Expired'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


#Feature importance extraction
importances = model.feature_importances_

feature_names = x_train.columns

feat_imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
feat_imp_df = feat_imp_df.sort_values(by='importance', ascending=False).head(10)

print(feat_imp_df)

plt.figure(figsize=(8, 6))
plt.barh(feat_imp_df['feature'], feat_imp_df['importance'])
plt.gca().invert_yaxis()
plt.xlabel("Importance")
plt.title("Top 10 Most Influential Features (XGBoost)")
plt.tight_layout()
plt.show()

#Making the folder for the models since this is the first model we defined
if not os.path.exists('./models'):
    os.makedirs('./models')

#Saving the model.
model.save_model("./models/xgboost_model.json")

#Calibration of the model.
#When testing for calibration we would like to see if the model for example predicts it's correct 30% of the time is it actually correct 30% of the time
brier = brier_score_loss(y_test, y_proba)
print(f"The Brier Score is: {brier:.6f}")

prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='uniform')

plt.figure(figsize=(8, 6))
plt.plot(prob_pred, prob_true, marker='o', label='XGBoost')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")
plt.title("Calibration Curve (Reliability Diagram)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
