#Regression Metrics
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
# load the data
df = sns.load_dataset('tips')
df.head()

X = df[['total_bill', 'size']]
y = df['tip']

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)

# initialize the model
model = LinearRegression()

# fit/train the model
model.fit(X_train, y_train)

# metric to evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
y_pred = model.predict(X_test)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")
print(f"Mean Absolute Percentage Error: {mean_absolute_percentage_error(y_test, y_pred)}")
# root mean squared error
print(f"Root Mean Squared Error: {np.sqrt(mean_squared_error(y_test, y_pred))}")