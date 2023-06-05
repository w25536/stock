import pandas as pd
from prophet import Prophet

df = pd.read_csv('./TSLA.csv')
df.head()

data = df[["Date", "Adj Close"]]
data = data.rename(columns={"Date": "ds", "Adj Close": "y"})

# Reset the index without adding a new column
data.reset_index(drop=True, inplace=True)

# Now it is usable for Prophet
data.head()

len(data)

from sklearn.model_selection import train_test_split

# Split the data into train and test sets
df_train, df_test = train_test_split(data, test_size=0.2, shuffle=False)

df_train = df_train.rename(columns={"Date": "ds", "Adj Close": "y"})
df_test = df_test.rename(columns={"Date": "ds", "Adj Close": "y"})

df_train

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=411)
forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

fig1 = m.plot(forecast)

plt.show()