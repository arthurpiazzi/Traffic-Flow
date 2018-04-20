def run(model):
	import pandas as pd

	n_rows = None

	print("____ Loading Data")
	x_train = pd.read_csv('csv/traffic/x_train.csv', nrows=n_rows)
	x_test = pd.read_csv('csv/traffic/x_test.csv', nrows=n_rows)

	y_train = pd.read_csv('csv/traffic/y_train.csv', nrows=n_rows)
	y_test = pd.read_csv('csv/traffic/y_test.csv', nrows=n_rows)

	x_train_weather = pd.read_csv('csv/weather/x_train.csv', nrows=n_rows)
	x_test_weather = pd.read_csv('csv/weather/x_test.csv', nrows=n_rows)

	y_train_weather = pd.read_csv('csv/weather/y_train.csv', nrows=n_rows)
	y_test_weather = pd.read_csv('csv/weather/y_test.csv', nrows=n_rows)

	assert x_train.shape[0] == x_train_weather.shape[0]
	assert x_test.shape[0] == x_test_weather.shape[0]

	print("____ Training model")
	model.train_unsurpevised(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test, load=False)
	model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test)

