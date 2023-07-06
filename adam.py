import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from imblearn.over_sampling import SMOTE

df_tf = pd.read_csv("ds_salaries.csv")

# Transform data using LabelEncoder
cols = ['experience_level', 'employment_type', 'job_title','salary_currency','employee_residence','company_location','company_size']
df_tf[cols]=df_tf[cols].apply(LabelEncoder().fit_transform)

# Create X, y
X = df_tf.drop(["salary_in_usd"], axis=1)
y = df_tf["salary_in_usd"]

# Build train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tf.random.set_seed(42)

# Create a model
model = tf.keras.Sequential([
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(1),
])

# Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["mae"])

# Fit the model
history = model.fit(X_train, y_train, epochs=100)

pd.DataFrame(history.history).plot(figsize=(10,5))

prediction = model.predict(X_test)
prediction = np.ravel(prediction)

# Prepare data for plot
df_final = pd.DataFrame({"Y_test": y_test , "Prediction" : prediction})

# Sort index before plot
df_final = df_final.sort_index()

# Plot the final result
plt.figure(figsize= (15,5))
plt.plot(df_final)
plt.legend(["Actual" , "Prediction"])
plt.show()