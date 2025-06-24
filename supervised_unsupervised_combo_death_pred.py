import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv('./data/clustering_data_normalized_encoded.csv', index_col=0)


#We divide the data to target variable(what we want to predict) and the features which
#predict it
target = 'thirtyday_expire_flag'

x = data.drop(columns= [target])
y = data[target]

#We split the data for training data and testing data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42, stratify = y)

#Doing the kmeans in the training set and on the testing set respectively we
#prevent data leakage which in turn would lead the model to overfit.
kmeans = KMeans(n_clusters=2, random_state=42)
train_clusters = kmeans.fit_predict(pd.concat([x_train, y_train], axis = 1))
test_clusters = kmeans.predict(pd.concat([x_test,y_test], axis = 1))


x_train['cluster'] = train_clusters
x_test['cluster'] = test_clusters

#Defining the model
model = XGBClassifier( n_estimators= 1000, max_depth= 8, learning_rate= 0.01, eval_metric= 'logloss', random_state= 42)

#Training the model
model.fit(x_train,y_train)


#evaluating the model using the test set
y_pred = model.predict(x_test)

print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
print(f"Precision: {precision_score(y_test,y_pred)}")
print(f"{classification_report(y_pred,y_test)}")

cm = confusion_matrix(y_pred,y_test)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Expired'], yticklabels=['Survived', 'Expired'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

#extracting feature importance
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
#Saving The model
model.save_model("./models/xgboost_model_clustered.json")

#Defining the neural network model
model1 = Sequential([
    Dense(32, activation='relu', input_shape=(x_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Defining the model's loss and optimizer
model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Accuracy()])

#Training the model
history = model1.fit(x_train, y_train, validation_split=0.2, epochs=30, batch_size=32, verbose=1)


#Evaluation the model
y_pred_proba = model1.predict(x_test).flatten()
y_pred = (y_pred_proba > 0.5).astype(int)

cm2 =  confusion_matrix(y_test, y_pred)
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test,y_pred)}")
print("ROC AUC Score:", roc_auc_score(y_test, y_pred_proba))
print("Confusion Matrix:\n", cm2)

#Plotting loss convergence
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Curve')
#Plotting accuracy convergence
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()
plt.title('Accuracy Curve')
plt.tight_layout()
plt.show()


sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', xticklabels=['Survived', 'Expired'], yticklabels=['Survived', 'Expired'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


#Saving the neural network model
model1.save('./models/cluster_fnn_.keras')
