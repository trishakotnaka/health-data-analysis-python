import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.weightstats import ztest
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
df = pd.read_excel(r"C:\Users\hp\Downloads\Health.xlsx")

# Column names
value_col = 'Value'
state_col = 'State'
year_col = 'year'

# Basic info
print(df.head())
print(df.columns)
print(df.describe())

# Handle missing values
df[value_col] = df[value_col].fillna(df[value_col].median())
df[state_col] = df[state_col].fillna(df[state_col].mode()[0])

# -------------------------------
# 1. BAR CHART (Top 10 States)
# -------------------------------
state_avg = df.groupby(state_col)[value_col].mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10,6))
state_avg.plot(kind='barh', color='skyblue')
plt.title("Top 10 States - Health Indicator")
plt.xlabel("Average Value")
plt.ylabel("State")
plt.show()


# -------------------------------
# 2. LINE PLOT (Year Trend)
# -------------------------------
year_avg = df.groupby(year_col)[value_col].median()
plt.figure(figsize=(8,5))
plt.plot(year_avg.index, year_avg.values, marker='o', color='green')
plt.title("Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Value")
plt.grid()
plt.show()


# -------------------------------
# 3. SCATTER PLOT
# -------------------------------
plt.figure(figsize=(8,5))
sns.scatterplot(x=year_col, y=value_col, data=df, color='red')
plt.title("Scatter Plot: Year vs Value")
plt.show()
# -------------------------------
# 4. HISTOGRAM
# -------------------------------
plt.figure(figsize=(8,5))
plt.hist(df[value_col], bins=30, color='purple')
plt.title("Histogram of Values")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

# -------------------------------
# 5. BOX PLOT
# -------------------------------
top_states = df[state_col].value_counts().head(10).index
df_top = df[df[state_col].isin(top_states)]
plt.figure(figsize=(12,6))
sns.boxplot(x=state_col, y=value_col, hue=state_col, data=df_top, palette='Set2', legend=False)
plt.xticks(rotation=45)
plt.title("Box Plot by State")
plt.show()

# -------------------------------
# 6. VIOLIN PLOT
# -------------------------------
df_top = df[df[state_col].isin(df[state_col].value_counts().head(10).index)]

df_top = df_top[df_top[state_col] != "United States"]   # remove outlier

plt.figure(figsize=(12,6))
sns.violinplot(x=state_col, y=value_col, hue=state_col, data=df_top, palette='coolwarm', legend=False)

   # ⭐ ADD THIS LINE HERE

plt.xticks(rotation=45)
plt.title("Violin Plot by State (Final Clean)")
plt.yscale('log')
plt.show()

# -------------------------------
# 7. PIE CHART
# -------------------------------
state_sum = df.groupby(state_col)[value_col].sum().sort_values(ascending=False).head(5)

plt.figure(figsize=(6,6))
plt.pie(state_sum, labels=state_sum.index, autopct='%1.1f%%',
        colors=['gold','lightcoral','skyblue','green','orange'])
plt.title("Top 5 States Contribution")
plt.show()
# -------------------------------
# 8. HYPOTHESIS TEST
# -------------------------------
states = df[state_col].unique()

state1 = df[df[state_col] == states[0]][value_col]
state2 = df[df[state_col] == states[1]][value_col]

t_stat, t_p = ttest_ind(state1, state2, nan_policy='omit')
print("T-test p-value:", t_p)

# -------------------------------
# 9. REGRESSION
# -------------------------------
X = year_avg.index.values.reshape(-1,1)
y = year_avg.values

model = LinearRegression()
model.fit(X, y)

future = np.array([[2025]])
print("Prediction for 2025:", model.predict(future)[0])

plt.figure(figsize=(8,5))
plt.plot(year_avg.index, y, marker='o', color='blue', label='Actual')
plt.plot(year_avg.index, model.predict(X), color='red', label='Regression')

plt.title("Regression Trend")
plt.xlabel("Year")
plt.ylabel("Value")
plt.legend()
plt.show()
