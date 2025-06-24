#In this part we pre process the data, remove missing values normalize data,
#encode categories if beneficial, reduce data dimensionality and other methods.
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

inital_data = pd.read_csv("./data/MIMIC_data_sample_mortality.csv")

#Mapping and encoding different columns to numerical values to make it feasible for different models
inital_data['gender'] = inital_data['gender'].map({'M': 0, 'F':1})


inital_data = inital_data.drop(columns = ['is_male'])


#We fill NANS with the mean value of each column
#To all the data columns which are either
# Lab tests or Vital signs we take note that these are the columns which contain the missing values.
#We also see that there are approximately 100 missing value rows and that a row which has a missing value
#typically have so for 1-3 columns where the other vital signs and lab tests data are valid

#Plot of missing values
plt.figure(figsize=(16, 8))
sns.heatmap(inital_data.isna(),
            cmap="YlOrRd",
            cbar=False,
            yticklabels=False)
plt.title("Heatmap of Missing Values (NaNs)", fontsize=16)
plt.xlabel("Columns", fontsize=12)
plt.ylabel("Rows", fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

#Fillining missing values with mean
inital_data = inital_data.fillna(inital_data.mean(numeric_only=True))




#We remove extreme values here
inital_data = inital_data[inital_data['urineoutput'] <= 20000]


inital_data = inital_data[inital_data['glucose_max1'] < 20000]
inital_data = inital_data[inital_data['glucose_mean'] < 20000]


#We drop more unnecessary columns
inital_data = inital_data.drop(columns = ['icustay_id', 'hadm_id', 'subject_id'])
inital_data.select_dtypes(include = 'number').hist(figsize=(16, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms of Numerical Features", fontsize=18)
plt.tight_layout()
plt.show()

#Saving one data post pre processing which will be used in statistical analysis
inital_data.to_csv('./data/statistical_analysis_data.csv')

#Dropping more unneccessary columns, it's information is already contained within ethnicity column
inital_data = inital_data.drop(columns = ['race_white', 'race_black', 'race_hispanic', 'race_other'])


#Encoding data for modeling purposes
initial_data_encoded = pd.get_dummies(inital_data, columns=['ethnicity', 'first_service'])

#Saving the data which will be relevant for the tree based models such as XGBoost and Random Forest
initial_data_encoded.to_csv("./data/boosting_models_data.csv")

non_num_col = ['ethnicity', 'first_service', 'thirtyday_expire_flag', 'gender', 'vent', 'diabetes', 'metastatic_cancer']
#Normalizing the data according to Z-score (or standardscaler)
for col in initial_data_encoded:
    if col not in non_num_col and 'ethnicity' not in col and 'first_service' not in col:
        mean = initial_data_encoded[col].mean()
        std = initial_data_encoded[col].std()
        initial_data_encoded[col] = (initial_data_encoded[col] - mean) / std


#Saving the relevant data for the logistic regression model.
initial_data_encoded.to_csv('./data/clustering_data_normalized_encoded.csv')


