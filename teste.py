from System_2 import DBN_System_2
import pandas as pd
from DBN import DBN

n_rows = 100


print("____ Loading Data")
x_train = pd.read_csv('csv/traffic/x_train.csv', nrows=n_rows)
x_test = pd.read_csv('csv/traffic/x_test.csv', nrows=n_rows)

y_train = pd.read_csv('csv/traffic/y_train.csv', nrows=n_rows)
y_test = pd.read_csv('csv/traffic/y_test.csv', nrows=n_rows)

x_train_weather = pd.read_csv('csv/weather/x_train.csv', nrows=n_rows)
x_test_weather = pd.read_csv('csv/weather/x_test.csv', nrows=n_rows)

y_train_weather = pd.read_csv('csv/weather/y_train.csv', nrows=n_rows)
y_test_weather = pd.read_csv('csv/weather/y_test.csv', nrows=n_rows)


print("____ Creating model")

model = DBN_System_2()

print("____ Training model")

model2 = DBN([250, 200, 100], [150, 150, 150])

model2.fit(x_train)
model2.fit_supervised(x_train, y_train, x_test, y_test)

model.train_unsurpevised(x_train, x_train_weather)

model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test)

model.save('model/weather.pkl')
