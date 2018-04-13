from RBM import RBM
from base import Base
import tensorflow as tf

class DBN(Base):
	def __init__(self, params, n_iter ):
		self.n_layers = len(params)
		self.params = params
		if type(n_iter) == int:
			self.n_iter = [n_iter]*self.n_layers
		elif len(n_iter) != self.n_layers:
			raise ValueError("The length of params and n_iter must agree")
		else:
			self.n_iter = n_iter
            
	def fit(self, X, transform = False):
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

		if transform:
			return self.predict(X)
            
	def predict(self, X):
		X_input = X
		for i in range(self.n_layers):
			X_input = self.layer[i].predict(X_input)
            
		return X_input

	def get_greed_model(self, X, trainable = True, dropout = 0.2):

		_layers_array = [None]*(self.n_layers+1)
		_layers_array[0] = X
		for i, layer in enumerate(self.layer, start = 1):
			_layers_array[i] = tf.nn.dropout(tf.nn.sigmoid(tf.matmul(_layers_array[i-1], tf.Variable(layer.w_best, trainable=trainable)) + tf.Variable(layer.hb_best, trainable=trainable)), 1-dropout)
		
		return _layers_array[-1]

	def fit_supervised(self, X, Y, X_test, Y_test, batch = 100, learning_rate = 1e-3, dropout = 0.2, epoch = 100):

		_input = tf.placeholder('float', [None, X.shape[1]])
		y_hat = tf.placeholder('float', [None, Y.shape[1]])

		out = self.get_greed_model(_input)

		W = tf.Variable(tf.truncated_normal([out.get_shape().as_list()[1], Y.shape[1]]))
		b = tf.zeros([Y.shape[1]])

		out = (tf.matmul(out, W) + b)

		loss = tf.reduce_mean(tf.squared_difference(out, y_hat))
		optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)

		error_best = 0
		epoch_best = 0
		patience = 0

		with tf.Session() as sess:
			for i in range(epoch):
				epoch_loss = 0 
				for start, end in zip(range(0, X.shape[0], batch),range(batch, X.shape[0], batch)):
					batch_x = X.loc[start:end]
					batch_y = Y.loc[start:end]

					_, cost = sess.run([optimizer, loss], feed_dict = {_input: batch_x, y_hat: batch_y})
					epoch_loss += cost
				val_loss = loss.eval(feed_dict={_input: X_test, y_hat: Y_test})
				if val_loss <= error_best or i == 0:
					error_best = val_loss
					epoch_best = i
					star = '*'

					#self.save('model.pkl')
					patience = 0
				else:
					patience += 1
					star = ''
					if patience >= 20: break
				print('Epoch: {}, train error: {}, validation loss: {},  best: {} at epoch: {} {}'.format(i, epoch_loss, val_loss,error_best, epoch_best, star))



