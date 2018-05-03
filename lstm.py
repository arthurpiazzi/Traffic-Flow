import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dropout, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from misc import coeff_determination
from datetime import datetime



# x_train_weather = pd.read_csv('csv/weather/x_train.csv', nrows=n_rows)
# x_test_weather = pd.read_csv('csv/weather/x_test.csv', nrows=n_rows)

# y_train_weather = pd.read_csv('csv/weather/y_train.csv', nrows=n_rows)
# y_test_weather = pd.read_csv('csv/weather/y_test.csv', nrows=n_rows)

def create_lstm(input_shape, output_shape, n_layer=1, n_units = 128):
	if n_layer > 1: return_sequence = True
	else: return_sequence = False
	model = Sequential()
	print(input_shape)
	model.add(LSTM(n_units, input_shape=(input_shape[1], input_shape[2]), return_sequences = return_sequence)) 
	for i in range(n_layer-1):
		#n_units = int(n_units/2)
		if i == n_layer - 2 : return_sequence = False
		model.add(LSTM(n_units, return_sequences = return_sequence))
	model.add(Dense(n_units))
	model.add(Dropout(0.2))
	model.add(Dense(output_shape))

	return model


def create_gru(input_shape, output_shape, n_layer=1, n_units = 128, get_inter = False):
	if n_layer > 1: return_sequence = True
	else: return_sequence = False
	model = Sequential()
	print(input_shape)
	model.add(GRU(n_units, input_shape=(input_shape[1], input_shape[2]), return_sequences = return_sequence)) 
	for i in range(n_layer-1):
		#n_units = int(n_units/2)
		if i == n_layer - 2 : return_sequence = False
		model.add(GRU(n_units, return_sequences = return_sequence))
	if get_inter:
		model.add(Dense(n_units))
		model.add(Dropout(0.2))
		model.add(Dense(output_shape))

	return model


if __name__=='__main__':
	n_rows = None
	time_steps = 20

	print("____ Loading Data")
	x_train = pd.read_csv('csv/traffic/x_train.csv', nrows=n_rows)
	x_test = pd.read_csv('csv/traffic/x_test.csv', nrows=n_rows)


	y_train = pd.read_csv('csv/traffic/y_train.csv', nrows=n_rows)
	y_test = pd.read_csv('csv/traffic/y_test.csv', nrows=n_rows)

	x_train = x_train.values.reshape(-1, time_steps, int(x_train.shape[1]/time_steps))
	print(y_train.shape)
	x_test = x_test.values.reshape(-1, time_steps, int(x_test.shape[1]/time_steps))
	y_train = y_train.values
	y_test = y_test.values
	for i in range(1,4):
		model_name = 'gru_layer_{}'.format(i)
		model = create_gru(x_train.shape, y_test.shape[1], n_units = 128, n_layer = i)
		check_pointer = ModelCheckpoint(filepath='best.hdf5', save_best_only=True)
		monitor = EarlyStopping(monitor='val_loss', patience = 15, min_delta = 1e-7, verbose = 2)
		model.compile(loss='mean_squared_error', optimizer='nadam', metrics =['mae', 'mape', coeff_determination])
		model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, check_pointer], batch_size=32, epochs=1000, verbose=0)
		model.load_weights('best.hdf5')
		model.save('model/{}'.format(model_name))
		mse, mae, mape, r2 = model.evaluate(x_train, y_train)
		mse_t, mae_t, mape_t, r2_t = model.evaluate(x_test, y_test)
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
		with open('log_{}.txt'.format(model_name), 'a') as fid:
			fid.write(log_msg)	