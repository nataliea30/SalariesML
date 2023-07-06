import pandas as pd


# ingesting the csv dataset
dataFrame = pd.read_csv('ds_salaries.csv')

# removing the 'salary' and 'salary_currency' columns to focus on USD
dataFrame.drop(dataFrame[['salary','salary_currency']], axis = 1, inplace = True)

print(dataFrame.shape)
print(dataFrame.head())