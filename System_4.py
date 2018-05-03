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

class System_4(DBN_System_2):
	def __init__(self, name = None, par_1 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}, par_2= {'iter': [150, 150, 150], 'n_units': [50, 40, 35]}, par_3 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}):
		super(System_4, self).__init__(name, par_1 , par_2, par_3)

	def train_greed(self, X_traffic, X_weather, Y_traffic, X_traffic_test = None, X_weather_test = None, Y_traffic_test = None, n_epoch = 1000, dropout = 0.1, batch = 100, path = None):
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
			out_1 = self.dbn_1.fit(X_traffic, transform = True, save=False)
			out_2 = self.dbn_2.fit(X_weather, transform = True, save=False)

			input_3 = tf.concat([out_1, out_2], axis = 1)
			with tf.Session() as sess:
				self.dbn_3.fit(input_3.eval(), transform = False, save=False)

			input_3 = tf.concat([out_1, out_2], axis = 1)
			print(input_3.shape)
			with tf.Session() as sess:
				self.dbn_3.fit(input_3.eval(), transform = False, save=False)

			

if __name__ == '__main__':

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

	model = System_4(name= 'System_4')

	print("____ Training model")
	try:
		model.train_unsurpevised(x_train, x_train_weather, y_train, y_train_weather, x_test, x_test_weather, y_test, y_test_weather, load=True)
	except:
		model.train_unsurpevised(x_train, x_train_weather, y_train, y_train_weather, x_test, x_test_weather, y_test, y_test_weather, load=False)

	model.train_greed(x_train, x_train_weather, y_train, x_test, x_test_weather, y_test)
