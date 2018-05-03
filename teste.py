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

	model.train_unsurpevised(x_train, x_train_weather, load=False)
	
	
	model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test)


def cross(model):
	import pandas as pd
	from sklearn.model_selection import LeaveOneGroupOut 
	n_rows = 120

	x_traffic = pd.read_csv('x_traffic.csv', nrows= n_rows)
	y_traffic = pd.read_csv('y_traffic.csv', nrows= n_rows)

	x_weather = pd.read_csv('x_weather.csv', nrows= n_rows)
	y_weather = pd.read_csv('y_weather.csv', nrows= n_rows)
	if n_rows:
		m = pd.read_csv('mounths_test.csv')
	else:
		m = pd.read_csv('mounths.csv')


	logo = LeaveOneGroupOut()
	metrics = []
	for train_index, x_test_index in logo.split(x_traffic , groups = m['m']):
		x_train, x_test = x_traffic.iloc[train_index], x_traffic.iloc[x_test_index]
		y_train, y_test = x_traffic.iloc[train_index], x_traffic.iloc[x_test_index]
		x_train_weather, x_test_weather = x_weather.iloc[train_index], x_weather.iloc[x_test_index]
		y_train_weather, y_test_weather = y_weather.iloc[train_index], y_weather.iloc[x_test_index]

		try:
			model.train_unsurpevised(x_train, x_train_weather, load=True)	
		except:
			model.train_unsurpevised(x_train, x_train_weather, load=True)

		metrics = model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test, metrics=metrics)

	metrics = pd.Dataframe(metrics)
	log_msg = 'Execution on {}: \n\n \
		Results: \n\n \
		Training set\n \
		MSE: {}({}) \n \
		MAE: {}({})\n \
		MAPE:{}({}) \n \
		R^2: {}({}) \n\n \
		Teste set\n \
		MSE: {}({})\n \
		MAE: {}({})\n \
		MAPE:{}({})\n \
		R^2: {}({}) \n\n ---------------- \n\n\n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"), metrics[0].mean(), metrics[0].std(), metrics[1].mean(), metrics[1].std(), metrics[2].mean(), metrics[2].std(), metrics[3].mean(), metrics[3].std(), metrics[4].mean(), metrics[4].std(), metrics[5].mean(), metrics[5].std(), metrics[6].mean(), metrics[6].std(), metrics[7].mean(), metrics[7].std())
	print(log_msg)
	with open('log_stat_{}.txt'.format('{}'.format(model.name)), 'a') as fid:
		fid.write(log_msg)	


