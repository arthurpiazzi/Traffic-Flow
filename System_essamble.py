from DBN_keras import DBN
import tensorflow as tf 
from base import Base
from keras.layers import Dense, Merge, Dropout
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

from datetime import datetime
from misc import coeff_determination, perturbation_rank
from lstm import create_lstm, create_gru


class DBN_System_essamble(DBN):
	
	def __init__(self, name = None, par_1 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}, par_2= {'iter': [150, 150, 150], 'n_units': [50, 40, 35]}, par_3 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}):
		self.iter_1 = par_1['iter']
		self.iter_2 = par_2['iter']
		self.iter_3 = par_3['iter']

		self.n_units_1 = par_1['n_units']
		self.n_units_2 = par_2['n_units']
		self.n_units_3 = par_3['n_units']
		self.name = self.__class__.__name__
		self.create_unsuprvised_model()

	def create_unsuprvised_model(self):

		self.dbn_1 = DBN(self.n_units_1, self.iter_1)
		self.dbn_2 = DBN(self.n_units_2, self.iter_2)
		self.dbn_3 = DBN(self.n_units_3, self.iter_3)

	def train_unsurpevised(self, X_traffic, X_wheater, load = False):
		if load:
			self.model = self.load()
		else:
			print("____Training unsurpevised")
			out_1 = self.dbn_1.fit(X_traffic, transform = True, save=False)
			out_2 = self.dbn_2.fit(X_wheater, transform = True, save=False)

			input_3 = tf.concat([out_1, out_2], axis = 1)
			with tf.Session() as sess:
				self.dbn_3.fit(input_3.eval(), transform = False, save=False)

			self.model = self.create_greed_model()
			self.save(self.model)


	def create_greed_model(self, dropout =  0.2):
		model_1 = self.dbn_1.get_greed_model(dropout = dropout)
		model_2 = self.dbn_2.get_greed_model(dropout = dropout)

		model = Sequential()
		model.add(Merge([model_1, model_2], mode = 'concat'))

		model = self.dbn_3.get_greed_model(dropout = dropout, model = model)

		return model

	def train_greed(self, X_traffic, X_weather, Y_traffic, X_traffic_test = None, X_weather_test = None, Y_traffic_test = None, n_epoch = 1000, dropout = 0.1, batch = 100, path = None):
		print("____Training Greed")

		time_steps = 20
		x_train = X_traffic.values.reshape(-1, time_steps, int(X_traffic.shape[1]/time_steps))
		x_test = X_traffic_test.values.reshape(-1, time_steps, int(X_traffic_test.shape[1]/time_steps))


		if not path:
			path = 'model/{}-{}.hdf5'.format(self.__class__.__name__, datetime.now().strftime("%Y-%m-%d_%H:%M"))
			print(path)

		model_dbn = self.model
		model_gru = create_gru(x_train.shape, Y_traffic.shape[1], n_units = 128, n_layer = 1)
		#model_lstm = create_lstm(x_train.shape, Y_traffic.shape[1], n_units = 128, n_layer = 1)
		model = Sequential()
		model.add(Merge([model_dbn, model_gru], mode='concat'))
		model.add(Dense(128, activation='sigmoid'))
		model.add(Dropout(0.2))
		model.add(Dense(Y_traffic.shape[1]))

		check_pointer = ModelCheckpoint(filepath=path, save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-5, verbose = 2)

		model.compile(loss='mean_squared_error', optimizer='nadam', metrics =['mae', 'mape', coeff_determination])
		model.fit([X_traffic, X_weather, x_train], Y_traffic, validation_data=([X_traffic_test, X_weather_test, x_test], Y_traffic_test), callbacks=[monitor, check_pointer], batch_size=32, epochs=1000, verbose=2)
		model.load_weights(path)
		self.model = model
		mse, mae, mape, r2 = model.evaluate([X_traffic, X_weather, x_train], Y_traffic)
		mse_t, mae_t, mape_t, r2_t = model.evaluate([X_traffic_test, X_weather_test, x_test], Y_traffic_test)
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
		with open('log_{}.txt'.format(self.__class__.__name__), 'a') as fid:
			fid.write(log_msg)	


	def score(self):
		pass 


if __name__ == '__main__':
	from teste import run
	print("____ Creating model")
	model = DBN_System_essamble()

	run(model)

