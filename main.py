import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Ingesting the csv dataset
dataFrame = pd.read_csv('ds_salaries.csv')

# Removing the 'salary' and 'salary_currency' columns to focus on USD
    #Added them back in, it seems that when these two are removed the accuracy severely drops to around <50% 
# DataFrame.drop(dataFrame[['salary','salary_currency']], axis = 1, inplace = True)

print(dataFrame.shape)
print(dataFrame.head())



# Transform data using LabelEncoder to all be numerical
cols = ['experience_level', 'employment_type', 'job_title','employee_residence','company_location','company_size', 'salary_currency', 'salary']
dataFrame[cols]=dataFrame[cols].apply(LabelEncoder().fit_transform)

# Convert all data types to int64
dataFrame = dataFrame.astype('int64')

# Create X, y
X = dataFrame.drop(["salary_in_usd"], axis=1).values
y = dataFrame["salary_in_usd"]

# Build train and test sets
    # It is important to note that stratification is not possible, however oversampling the minority/undersampling the majority class can be done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dataFrame.info()

# Create an instance of the Random Forest Regressor
    #Each decision tree has a maximun depth of 7, at each split only 3 features are considered, and there are 100 variants of decision trees
rf_regressor =RandomForestRegressor(max_depth=7 , max_features=3,n_estimators= 100)

# Train the model on the training data
rf_regressor.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = rf_regressor.predict(X_test)

# Evaluate the model's performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)
print("Accuracy:", rf_regressor.score(X_train, y_train))

# Displaying the results in a chart
y_pred = rf_regressor.predict(X_test)
df3 = pd.DataFrame({"Y_test": y_test , "Y_pred" : y_pred})
df3 = df3.reset_index(drop=True)
print(df3.head(20))

# Plotting results on a graph
plt.figure(figsize= (20,6))

plt.plot(df3[:500])
plt.legend(["Actual" , "Predicted"])
plt.show()