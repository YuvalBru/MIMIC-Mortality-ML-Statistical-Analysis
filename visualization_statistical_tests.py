import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, mannwhitneyu
import decimal
from decimal import Decimal


data = pd.read_csv("./data/statistical_analysis_data.csv")

#We want to understand the different distributions of the features
#These distribution will also help us understand what statistical tests to do.
data.select_dtypes(include = 'number').hist(figsize=(16, 10), bins=30, edgecolor='black')
plt.suptitle("Histograms For Distribution Analysis", fontsize=18)
plt.tight_layout()
plt.show()

print(data.describe())
numeric = data.select_dtypes(include='number')

cor_with_expire = numeric.corr()['thirtyday_expire_flag'].abs().sort_values(ascending=False)
cor_with_los = numeric.corr()['icu_los'].abs().sort_values(ascending=False)


top_features = list(set(cor_with_expire.head(10).index) | set(cor_with_los.head(10).index))
top_features += ['thirtyday_expire_flag']

#We want to understand the correlation between different features in our dataset and so therefore we plot a correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data[top_features].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Among Top Features and Targets")
plt.show()


#UNderstanding the relation between features that had high correlations to the 30-day expire flag to the feature.
plt.figure(figsize=(6, 4))
sns.boxplot(x='thirtyday_expire_flag', y=data['sofa'], data=data)
plt.title(f"sofa vs. 30-Day Mortality")
plt.xlabel("Expired (1=Yes, 0=No)")
plt.ylabel('sofa')
plt.tight_layout()
plt.show()

ct1 = pd.crosstab(data['vent'], data['thirtyday_expire_flag'],
                 rownames=['Ventilation'], colnames=['Expired (30d)'])

mortality_rate = data.groupby('vent')['thirtyday_expire_flag'].mean()


sns.barplot(x='vent', y='thirtyday_expire_flag', data=data, ci=None)
plt.xticks([0, 1], ['Not Ventilated', 'Ventilated'])
plt.ylabel("30-Day Mortality Rate")
plt.title("Mortality Rate by Ventilation Status")
plt.show()

#We perform statistical tests to gain deeper understanding of said correlations between features
#specifically we examine the expire flag correlation
chi2, p2, dof, ex = chi2_contingency(ct1)


group0 = data[data['thirtyday_expire_flag'] == 0]['sofa']
group1 = data[data['thirtyday_expire_flag'] == 1]['sofa']

stat, p1 = mannwhitneyu(group0, group1, alternative='two-sided')

decimal.getcontext().prec = 50

p1_precise = Decimal(p1)
p2_precise = Decimal(p2)
print(f"Precise p-value for Mann-Whitney U Test (Sofa Score vs Expire Flag): {p1_precise} Precise p-value for Chi-Squared test (Ventialation vs Expire Flag): \n {p2_precise}")


counts = data['thirtyday_expire_flag'].value_counts().sort_index()


plt.bar(counts.index, counts.values, color=['skyblue', 'salmon'])
plt.xticks([0, 1], ['Survived (0)', 'Expired (1)'])
plt.xlabel('30-Day Expire Flag')
plt.ylabel('Number of Patients')
plt.title('Class Distribution of 30-Day Mortality')
plt.tight_layout()
plt.show()

