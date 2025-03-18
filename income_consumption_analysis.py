##Research Question
##Does higher income lead to more consumption?

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

#Load the dataset
df = pd.read_csv("/home/jeumma/econometrics_analysis/Family Income and Expenditure.csv")
data = np.nan_to_num(df, nan=0, posinf=0, neginf=0)

#Show the first few rows
df.head()

# Drop rows with NaN values
df_cleaned = df.dropna(subset=['Total Household Income', 'Total Food Expenditure'])


#Scatter plot: Income vs. Consumption
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Total Household Income'], y=df['Total Food Expenditure'])
plt.xlabel("Income")
plt.ylabel("Food Consumption")
plt.title("Income vs. Food Consumption")
plt.show()



#Regression Analysis
X= df["Total Household Income"]
y= df["Total Food Expenditure"]

X= sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())


#Log-transform
df['Log Total Household Income'] = np.log(df['Total Household Income'] + 1)
df['Log Total Food Expenditure'] = np.log(df['Total Food Expenditure'] + 1)

X_log = sm.add_constant(df['Log Total Household Income'])
y_log = df['Log Total Food Expenditure']
model_log = sm.OLS(y_log, X_log).fit()

print(model_log.summary())