import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest
from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_excel(r"C:\Users\hp\Downloads\Health.xlsx")

print(df.head(10))
print(df.columns)
print(df.dtypes)
print(df.shape)

print(df.isnull().sum())

# Fix column names safely
value_col = 'DataValue' if 'DataValue' in df.columns else 'Value'
state_col = 'LocationDesc' if 'LocationDesc' in df.columns else 'State'
year_col = 'YearStart' if 'YearStart' in df.columns else ('Year' if 'Year' in df.columns else 'year')
df[value_col] = df[value_col].fillna(df[value_col].mean())
df[state_col] = df[state_col].fillna(df[state_col].mode()[0])

print(df[year_col].value_counts())
print(df.describe())

state_avg = df.groupby(state_col)[value_col].mean()
plt.figure()
state_avg.plot(kind='bar')
plt.title("State-wise Health Indicator Performance")
plt.xlabel("State")
plt.ylabel("Average Value")
plt.xticks(rotation=90)
plt.show()

year_trend = df.groupby(year_col)[value_col].mean()
plt.figure()
year_trend.plot(kind='line', marker='o')
plt.title("Trend of Health Indicators Over Years")
plt.xlabel("Year")
plt.ylabel("Average Value")
plt.show()

state_list = df[state_col].unique()
state1 = df[df[state_col] == state_list[0]][value_col]
state2 = df[df[state_col] == state_list[1]][value_col]

t_stat, t_p = ttest_ind(state1, state2, nan_policy='omit')
print(t_p)

if t_p < 0.05:
    print("Significant difference between two states")
else:
    print("No significant difference")

if 'Stratification1' in df.columns and 'Male' in df['Stratification1'].values and 'Female' in df['Stratification1'].values:
    male = df[df['Stratification1'] == 'Male'][value_col]
    female = df[df['Stratification1'] == 'Female'][value_col]
    z_stat, z_p = ztest(male.dropna(), female.dropna())
    print(z_p)
    if z_p < 0.05:
        print("Significant difference between categories")
    else:
        print("No significant difference")
else:
    print("Z-test skipped")

X = df[[year_col]]
y = df[value_col]

model = LinearRegression()
model.fit(X, y)

future_year = np.array([[2025]])
prediction = model.predict(future_year)

print(prediction[0])

plt.figure()
plt.scatter(X, y)
plt.plot(X, model.predict(X))
plt.title("Linear Regression: Year vs DataValue")
plt.xlabel("Year")
plt.ylabel("DataValue")
plt.show()
