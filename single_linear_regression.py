import pandas as pd
from  sklearn.linear_model import LinearRegression
import seaborn as sns
import warnings

df = sns.load_dataset('tips')
df.head()
#double column value because it is depends on another column
#select the feature column from dataframe(df)
X_bill = df[['total_bill']]
#select the target(label) column from dataframe(df)
y_tip = df['tip']
#Initialize the model
model=LinearRegression()
#trained the model
model.fit(X_bill, y_tip)
#Predict the tip

warnings.filterwarnings("ignore", category=UserWarning)
X_test=[[50]]
output=model.predict(X_test)
print(output)