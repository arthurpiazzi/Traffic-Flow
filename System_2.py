from DBN_keras import DBN
import tensorflow as tf 
from base import Base
from keras.layers import Dense, Merge
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping

from datetime import datetime
from sklearn.metrics import r2_score
import pandas as pd
from teste import perturbation_rank_system

class DBN_System_2(DBN):
	
	def __init__(self, name = None, par_1 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}, par_2= {'iter': [150, 150, 150], 'n_units': [50, 40, 35]}, par_3 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}):
		self.iter_1 = par_1['iter']
		self.iter_2 = par_2['iter']
		self.iter_3 = par_3['iter']

		self.n_units_1 = par_1['n_units']
		self.n_units_2 = par_2['n_units']
		self.n_units_3 = par_3['n_units']
		self.name = name
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

		if not path:
			path = 'model/{}-{}.hdf5'.format(self.name, datetime.now().strftime("%Y-%m-%d_%H:%M"))
			print(path)

		model = self.model
		model.add(Dense(Y_traffic.shape[1]))

		check_pointer = ModelCheckpoint(filepath=path, save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-5, verbose = 2)

		model.compile(loss='mean_squared_error', optimizer='nadam', metrics =['mae', 'mape'])
		model.fit([X_traffic, X_weather], Y_traffic, validation_data=([X_traffic_test, X_weather_test], Y_traffic_test), callbacks=[monitor, check_pointer], batch_size=32, epochs=1000, verbose=2)
		model.load_weights(path)

		result = perturbation_rank_system(model, X_traffic_test.values, X_weather_test.values, Y_traffic_test.values, names=list(range(X_traffic.shape[1]+X_weather.shape[1])), regression = True)


		mse, mae, mape = model.evaluate([X_traffic, X_weather], Y_traffic)
		r2 = r2_score(Y_traffic, model.predict([X_traffic, X_weather]))
		mse_t, mae_t, mape_t = model.evaluate([X_traffic_test, X_weather_test], Y_traffic_test)
		r2_t = r2_score(Y_traffic_test, model.predict([X_traffic_test, X_weather_test]))

		try:
			df_result = pd.read_csv('result/system_2.csv')
			df_result.append(pd.DataFrame([mse, mae, mape, r2, mse_t, mae_t, mape_t, r2_t, result.error.mean(), result.error_mae.mean(), result.error_r2.mean()], columns = ['mse', 'mae', 'mape', 'r2', 'mse_t', 'mae_t', 'mape_t', 'r2_t', 'pertubation_mse', 'pertubation_mae', 'pertubation_r2']))
		except FileNotFoundError:
			df_result = pd.DataFrame([mse, mae, mape, r2, mse_t, mae_t, mape_t, r2_t, result.error, result.error_mae, result.error_r2], columns = ['mse', 'mae', 'mape', 'r2', 'mse_t', 'mae_t', 'mape_t', 'r2_t', 'pertubation_mse', 'pertubation_mae', 'pertubation_r2'])

		df_result.to_csv('result/system_2.csv')

		log_msg = 'Execution on {}: \n\n \
						Results: \n\n \
						Training set\n \
						MSE: {} \n \
						MAE: {}\n \
						MAPE:{} \n\n \
						Teste set\n \
						MSE: {} \
						\n MAE: {}\n \
						MAPE:{} \n\n\n ---------------- \n\n\n'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"), mse, mae, mape, mse_t, mae_t, mape_t)
		print(log_msg)
		with open('log_{}.txt'.format(self.name), 'a') as fid:
			fid.write(log_msg)


	def score(self):
		pass 


if __name__ == '__main__':
	from teste import run
	print("____ Creating model")
	model = DBN_System_2(name='TREINA_AS_3')
	for _ in range(10)
		run(model)

