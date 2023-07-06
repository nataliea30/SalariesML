import pandas as pd


# ingesting the csv dataset
dataFrame = pd.read_csv('ds_salaries.csv')

# removing the 'salary' and 'salary_currency' columns to focus on USD
dataFrame.drop(dataFrame[['salary','salary_currency']], axis = 1, inplace = True)

print(dataFrame.shape)
print(dataFrame.head())


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Transform data using LabelEncoder to all be numerical
cols = ['experience_level', 'employment_type', 'job_title','employee_residence','company_location','company_size']
dataFrame[cols]=dataFrame[cols].apply(LabelEncoder().fit_transform)

# Convert all data types to int64
dataFrame = dataFrame.astype('int64')

# Create X, y
X = dataFrame.drop(["salary_in_usd"], axis=1)
y = dataFrame["salary_in_usd"]

# Build train and test sets
# It is important to note that stratification is not possible, however oversampling the minority/undersampling the majority class can be done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dataFrame.info()
