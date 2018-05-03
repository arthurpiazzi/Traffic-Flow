from RBM import RBM
from base import Base
import tensorflow as tf

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime 
from teste import run
from misc import coeff_determination

class DBN(Base):
	def __init__(self, params, n_iter, name=None ):
		self.n_layers = len(params)
		self.params = params
		self.name = name
		if type(n_iter) == int:
			self.n_iter = [n_iter]*self.n_layers
		elif len(n_iter) != self.n_layers:
			raise ValueError("The length of params and n_iter must agree")
		else:
			self.n_iter = n_iter
            
	def fit(self, X, transform = False, load = False, name = None, save = True):
		if load:
			self.model = self.load()
		else: 
			self.layer = []
			X_input = X
			input_size = X_input.shape[1]
			self.input_shape = input_size
			for i in range(self.n_layers):
				output_size = self.params[i]
				print(input_size)
				self.layer.append(RBM(input_size, output_size))
				input_size = output_size
				self.layer[i].fit(X_input, self.n_iter[i])
				X_input = self.layer[i].predict(X_input)

			self.model = self.get_greed_model()
			if save:
				self.save(self.model)

			if transform:
				return self.predict(X)

            
	def predict(self, X):
		X_input = X
		for i in range(self.n_layers):
			X_input = self.layer[i].predict(X_input)
            
		return X_input

	def get_greed_model(self, trainable = True, dropout = 0.2, model = None):

		shapes = self.layer[0].w_best.shape
		if not model:
			model = Sequential()
			model.add(Dense(shapes[1], input_dim = shapes[0], activation='sigmoid'))
		else:
			model.add(Dense(shapes[1], activation='sigmoid'))

		model.layers[-1].set_weights([self.layer[0].w_best, self.layer[0].hb_best])
		model.add(Dropout(dropout))

		for i, layer in enumerate(self.layer[1:], start = 1):
			shapes = layer.w_best.shape
			model.add(Dense(int(shapes[1]), activation='sigmoid' ))
			model.layers[-1].set_weights([self.layer[i].w_best, self.layer[i].hb_best])
			model.add(Dropout(dropout))


		self.model = model
		return model

	def fit_supervised(self, X, Y, X_test, Y_test, batch = 100, learning_rate = 1e-4, dropout = 0.2, epoch = 1000, path = None, log=False, metrics = []):


		if not path:
			path = 'model/{}.hdf5'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"))
			print(path)
		model = self.model 
			
		model.add(Dense(Y_test.shape[1]))

		check_pointer = ModelCheckpoint(filepath=path, save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-5, verbose = 2)

		model.compile(loss='mean_squared_error', optimizer='nadam', metrics =['mae', 'mape', coeff_determination])
		model.fit(X, Y, validation_data=(X_test, Y_test), callbacks=[monitor, check_pointer], batch_size=32, epochs=epoch, verbose=2)
		model.load_weights(path)
		if log:
			print(type(X), type(Y), X.shape, Y.shape)
			mse, mae, mape, r2 = model.evaluate(X, Y)
			mse_t, mae_t, mape_t, r2_t = model.evaluate(X_test, Y_test)
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

		self.model = model


	def load(self):
		# load json and create model
		print("model/{}.json".format(self.name))
		json_file = open("model/{}.json".format(self.name), 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		loaded_model = model_from_json(loaded_model_json)
		# load weights into new model
		loaded_model.load_weights("model/{}.hdf5".format(self.name))
		print("Loaded model from disk")
		return loaded_model
		 
	def save(self, model):
		model_json = model.to_json()
		with open("model/{}.json".format(self.name), "w") as json_file:
		    json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights("model/{}.hdf5".format(self.name))
		print("Saved model to disk")


if __name__ == '__main__':

	import pandas as pd

	n_rows = None
	if False:

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

		model = DBN(name= 'DBN_trad', n_iter = 1000, params=[250,200,100] )

		print("____ Training model")
		model.fit(x_train, load=False)
		model.fit_supervised(x_train, y_train, x_test, y_test)

	else:
		from sklearn.model_selection import LeaveOneGroupOut 


		x_traffic = pd.read_csv('x_traffic.csv', nrows=n_rows)
		y_traffic = pd.read_csv('y_traffic.csv', nrows=n_rows)

		x_weather = pd.read_csv('x_weather.csv', nrows=n_rows)
		y_weather = pd.read_csv('y_weather.csv', nrows=n_rows)
		if n_rows:
			m= pd.read_csv('mounths_test.csv')
		else:
			m = pd.read_csv('mounths.csv')

		assert x_traffic.shape[0] == x_weather.shape[0]
		assert y_traffic.shape[0] == y_weather.shape[0]

		model = DBN(name= 'DBN_trad', n_iter = 1000, params=[250,200,100] )

		logo = LeaveOneGroupOut()
		idx = 1
		metrics = []

		for train_index, x_test_index in logo.split(x_traffic , groups = m['m']):
			x_train, x_test = x_traffic.iloc[train_index], x_traffic.iloc[x_test_index]
			y_train, y_test = x_traffic.iloc[train_index], x_traffic.iloc[x_test_index]
			x_train_weather, x_test_weather = x_weather.iloc[train_index], x_weather.iloc[x_test_index]
			y_train_weather, y_test_weather = y_weather.iloc[train_index], y_weather.iloc[x_test_index]

			try:
				model.fit(x_train, load=False)
			except:
				model.fit(x_train, load=True)
			metrics = model.fit_supervised(x_train, y_train, x_test, y_test, metrics = metrics)		
			

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
