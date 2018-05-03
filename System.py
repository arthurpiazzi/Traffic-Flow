import tensorflow as tf
from base import Base

class DBN_Heterogeneus(Base):
	def __init__(self, params1 = {'n_units': [250, 200, 100], 'n_iter': [150, 150, 150]}, params2 = {'n_units': [50, 40, 35], 'n_iter': [100, 100, 100]}):
		self.params_traffic = params1['n_units']
		self.params_weather = params2['n_units']

		self.n_iter_traffic = params1['n_iter']
		self.n_iter_weather = params2['n_iter']

		self.n_layer_traffic = len(self.params_traffic)
		self.n_layer_weather = len(self.params_weather)

	def fit_DBN(self, X, traffic_or_weather = 'T'):
		if traffic_or_weather.upper() == "T":
			n_layers = self.n_layer_traffic
			params  = self.params_traffic
		elif traffic_or_weather.upper() == "W":
			n_layers = self.n_layer_weather
			params = self.params_weather

		layer = []
		X_input = X
		input_size = X_input.shape[1]
		for i in range(n_layers):
			output_size = params[i]
			layer.append(RBM(input_size, output_size))
			input_size = output_size
			layer[i].fit(X_input, n_iter[i])
			X_input = layer[i].predict(X_input)

		if traffic_or_weather.upper() == "T":
			self.layer_traffic = layer
		elif traffic_or_weather == "W":
			self.layer_weather.upper() = layer

	def fit(self, X_traffic, Y_traffic, X_weather, Y_weather):

		if not self.layer_traffic:
			self.fit_DBN(X_traffic, "T")

		if not self.layer_weather:
			self.fit_DBN(X_weather, "W")



		#Create placeholders for input, weights, biases, output
		_a_traffic = [None] * (self.n_layer_traffic + 1)

		_a_weather = [None]* (self.n_layer_weather + 1)


		_w_traffic = [None] * (self.n_layer_traffic)
		_b_traffic = [None] * (self.n_layer_traffic)
		_a_traffic[0] = tf.placeholder("float", [None, self._X.shape[1]])

		_w_weather = [None] * (len(self.n_layer_weather) + 1)
		_b_weather = [None] * (len(self.n_layer_weather) + 1)
		_a_weather[0] = tf.placeholder("float", [None, self._X.shape[1]])
		y = tf.placeholder("float", [None, self._Y.shape[1]])

        #Define variables and activation functoin
		for layer in range(self.n_layer_traffic):
			_w_traffic[i] = tf.Variable(self.layer_traffic[i].w_best)
			_b_traffic[i] = tf.Variable(self.layer_traffic[i].hb_best)

		for i in range(1, self.n_layer_traffic + 1):
			_a_traffic[i] = tf.nn.sigmoid(tf.matmul(_a_traffic[i - 1], _w_traffic[i - 1]) + _b_traffic[i - 1])

		for i in range(self.n_layer_traffic):
			_w_weather[i] = tf.Variable(self.layer_weather[i].w_best)
			_b_weather[i] = tf.Variable(self.layer_weather[i].hb_best)

		for i in range(1, self.n_layer_traffic + 1):
			_a_traffic[i] = tf.nn.sigmoid(tf.matmul(_a_traffic[i - 1], _w_traffic[i - 1]) + _b_traffic[i - 1])

		fused = tf.concat([_a_traffic[-1], _a_weather[-1]], axis=1)

		

		#Define the cost function
		cost = tf.reduce_mean(tf.square(_a[-1] - y))

		#Define the training operation (Momentum Optimizer minimizing the Cost function)
		train_op = tf.train.MomentumOptimizer(self._learning_rate, self._momentum).minimize(cost)

		#Prediction operation
		predict_op = tf.argmax(_a[-1], 1)

		#Training Loop
		with tf.Session() as sess:
			#Initialize Variables
			sess.run(tf.global_variables_initializer())

			#For each epoch
			for i in range(self._epoches):

				#For each step
				for start, end in zip(
					range(0, len(self._X), self._batchsize), range(self._batchsize, len(self._X), self._batchsize)):

					#Run the training operation on the input data
					sess.run(train_op, feed_dict={_a[0]: self._X[start:end], y: self._Y[start:end]})

				for j in range(len(self._sizes) + 1):
					#Retrieve weights and biases
					self.w_list[j] = sess.run(_w[j])
					self.b_list[j] = sess.run(_b[j])

				if i % 10:
					print ("{}Accuracy rating for epoch {} : {}".format(color, i, np.mean(np.argmax(self._Y, axis=1) == sess.run(predict_op, feed_dict={_a[0]: self._X, y: self._Y}))))