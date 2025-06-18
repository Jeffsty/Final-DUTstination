import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the CSV file
df = pd.read_csv('./Consolidated_Data/Consolidated_All_Tests_Data.csv')

df = df.drop(['Test Start Time', 'Test End Time'], axis=1)

# 2. Inspect the data
print(df.head())
print(df.info())
print(df.describe())

# 3. Check for missing values
print(df.isnull().sum())

# 4. Distribution of numerical columns
df.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

# 6. Pairplot (only for small datasets)
sns.pairplot(df)
plt.show()