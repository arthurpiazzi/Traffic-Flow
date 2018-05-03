from System_2 import DBN_System_2
from keras.models import Sequential
from keras.layers import Dense 
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
import tensorflow as tf
import numpy as np

from datetime import datetime

from teste import run
from misc import coeff_determination
from sklearn.metrics import r2_score


def get_second_last_out(model, x):
	get_out = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-2].output])
	# output in test mode = 0
	return get_out([x, 0])[0]

class System_3(DBN_System_2):
	def __init__(self, name = None, par_1 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}, par_2= {'iter': [150, 150, 150], 'n_units': [50, 40, 35]}, par_3 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}):
		super(System_3, self).__init__(name, par_1 , par_2, par_3)

	def train_greed(self, X_traffic, X_weather, Y_traffic, X_traffic_test = None, X_weather_test = None, Y_traffic_test = None, n_epoch = 1000, dropout = 0.1, batch = 100, path = None, metrics = []):
		print("____Training Greed")

		if not path:
			path = 'model/{}-{}'.format(self.name, datetime.now().strftime("%Y-%m-%d_%H:%M"))
			print(path)

		model_1, model_2, model_3 = self.create_greed_model()

		model_3.add(Dense(Y_traffic.shape[1]))


		check_pointer_3 = ModelCheckpoint(filepath='{}-{}.hdf5'.format(path,'_3'), save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-5, verbose = 2)

		model_3.compile(loss='mean_squared_error', optimizer='nadam', metrics =['mae', 'mape', coeff_determination])

		X, X_test = model_1.predict(X_traffic), model_1.predict(X_traffic_test)

		X_w, X_test_w = model_2.predict(X_weather), model_2.predict(X_weather_test)
		
		X = np.hstack([X, X_w])
		X_test = np.hstack([X_test, X_test_w])

		model_3.fit(X, Y_traffic, validation_data=(X_test, Y_traffic_test), callbacks=[monitor, check_pointer_3], batch_size=32, epochs=n_epoch, verbose=2)
		model_3.load_weights('{}-{}.hdf5'.format(path,'_3'))
		mse, mae, mape, r2 = model_3.evaluate(X, Y_traffic)
		mse_t, mae_t, mape_t, r2_t = model_3.evaluate(X_test, Y_traffic_test)
		metrics.append([mse, mae, mse_t, mae_t])

		log_msg = 'Execution on {}: \n\n \
						Results: \n\n \
						Training set\n \
						MSE: {} \n \
						MAE: {}\n \
						MAPE:{} \n \
						R^2: {} \n\n \
						Teste set\n \
						MSE: {}\n \
						MAE: {}\n \
						MAPE:{} \n \
						R^2: {} \n\n ---------------- \n\n\n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"), mse, mae, mape, r2, mse_t, mae_t, mape_t, r2_t)
		print(log_msg)
		with open('log_{}.txt'.format(self.name), 'a') as fid:
			fid.write(log_msg)	
		return metrics				

	def create_greed_model(self, dropout =  0.2):
		model_1 = self.dbn_1.get_greed_model(dropout = dropout)

		model_2 = self.dbn_2.get_greed_model(dropout = dropout)

		model_3 = self.dbn_3.get_greed_model(dropout = dropout)

		return model_1, model_2, model_3


	def train_unsurpevised(self,  X_traffic, X_weather, Y_traffic, Y_weather, X_traffic_test = None, X_weather_test = None, Y_traffic_test = None, Y_weather_test = None, load = False):
		if load:
			self.model = self.load()
		else:
			print("____Training unsurpevised")
			self.dbn_1.fit(X_traffic, transform = False, save=False)
			self.dbn_1.fit_supervised(X_traffic, Y_traffic, X_traffic_test, Y_traffic_test)

			out_1 = get_second_last_out(self.dbn_1.model, X_traffic)
			
			self.dbn_2.fit(X_weather, transform = False, save=False)
			self.dbn_2.fit_supervised(X_weather, Y_weather, X_weather_test, Y_weather_test)
			out_2 = get_second_last_out(self.dbn_2.model, X_weather)

			input_3 = tf.concat([out_1, out_2], axis = 1)
			print(input_3.shape)
			with tf.Session() as sess:
				self.dbn_3.fit(input_3.eval(), transform = False, save=False)

			

if __name__ == '__main__':

	n_rows = None
	import pandas as pd
	from sklearn.model_selection import LeaveOneGroupOut 

	x_traffic = pd.read_csv('x_traffic.csv', nrows=n_rows)
	y_traffic = pd.read_csv('y_traffic.csv', nrows=n_rows)

	x_weather = pd.read_csv('x_weather.csv', nrows=n_rows)
	y_weather = pd.read_csv('y_weather.csv', nrows=n_rows)
	m = pd.read_csv('mounths.csv')

	model = System_3(name= 'System_3')

	logo = LeaveOneGroupOut()
	idx = 1
	metrics = []
	print(logo.get_n_splits(x_traffic, groups=m['m']))
	
	for train_index, test_index in logo.split(x_traffic, y_traffic , m['m']):
		print(idx)
		x_train, x_test = x_traffic.iloc[train_index], x_traffic.iloc[test_index]
		y_train, y_test = x_traffic.iloc[train_index], x_traffic.iloc[test_index]
		x_train_weather, x_test_weather = x_weather.iloc[train_index], x_weather.iloc[test_index]
		y_train_weather, y_test_weather = y_weather.iloc[train_index], y_weather.iloc[test_index]

		try:
			model.train_unsurpevised(x_train, x_train_weather, y_train, y_train_weather, x_test, x_test_weather, y_test, y_test_weather, load=True)
		except FileNotFoundError:
			model.train_unsurpevised(x_train, x_train_weather, y_train, y_train_weather, x_test, x_test_weather, y_test, y_test_weather, load=False)
		metrics = model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test, metrics=metrics)
		idx+=1


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
	with open('log_{}.txt'.format(model.name), 'a') as fid:
		fid.write(log_msg)	
