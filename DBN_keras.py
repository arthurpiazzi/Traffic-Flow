from RBM import RBM
from base import Base
import tensorflow as tf

from keras.models import Sequential, model_from_json
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint

from datetime import datetime 
from teste import run

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

	def fit_supervised(self, X, Y, X_test, Y_test, batch = 100, learning_rate = 1e-4, dropout = 0.2, epoch = 1000, path = None, log=False):


		if not path:
			path = 'model/{}.hdf5'.format(datetime.now().strftime("%Y-%m-%d_%H:%M"))
			print(path)
		model = self.model 
			
		model.add(Dense(Y_test.shape[1]))

		check_pointer = ModelCheckpoint(filepath=path, save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-5, verbose = 2)

		model.compile(loss='mean_squared_error', optimizer='nadam')
		model.fit(X, Y, validation_data=(X_test, Y_test), callbacks=[monitor, check_pointer], batch_size=32, epochs=epoch, verbose=2)
		model.load_weights(path)
		if log:
			mse, mae, mape = model_3.evaluate([X, X_w], Y_traffic)
			mse_t, mae_t, mape_t = model_3.evaluate([X_test, X_test_w], Y_traffic_test)
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

	print("____ Creating model")
	model = DBN(name='DBN')

	run(model)
