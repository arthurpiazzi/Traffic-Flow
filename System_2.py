from DBN import DBN
import tensorflow as tf 
from base import Base

class DBN_System_2(Base):
	
	def __init__(self, par_1 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}, par_2= {'iter': [150, 150, 150], 'n_units': [50, 40, 35]}, par_3 = {'iter': [150, 150, 150], 'n_units': [250, 200, 100]}):
		self.iter_1 = par_1['iter']
		self.iter_2 = par_2['iter']
		self.iter_3 = par_3['iter']

		self.n_units_1 = par_1['n_units']
		self.n_units_2 = par_2['n_units']
		self.n_units_3 = par_3['n_units']
		self.create_unsuprvised_model()

	def create_unsuprvised_model(self):

		self.dbn_1 = DBN(self.n_units_1, self.iter_1)
		self.dbn_2 = DBN(self.n_units_2, self.iter_2)
		self.dbn_3 = DBN(self.n_units_3, self.iter_3)

	def train_unsurpevised(self, X_traffic, X_wheater):
		print("____Training unsurpevised")
		out_1 = self.dbn_1.fit(X_traffic, transform = True)
		out_2 = self.dbn_2.fit(X_wheater, transform = True)

		input_3 = tf.concat([out_1, out_2], axis = 1)

		self.dbn_3.fit(input_3, transform = False)

	def create_greed_model(self):
		pass

	def train_greed(X_traffic, X_wheater, Y_traffic):
		print("____Training Greed")

		input_traffic = tf.placeholder('float', [None, X_traffic.shape[1]])
		input_weather = tf.placeholder('float', [None, X_wheater.shape[1]])
		y_hat = tf.placeholder('float', [None, Y_traffic.shape[1]])

		out_1 = self.dbn_1.get_greed_model(input_traffic)
		out_2 = self.dbn_2.get_greed_model(input_weather)

		input_3 = tf.concat([out_1, out_2], axis=1)

		out_pre_regress = self.dbn_3.get_greed_model(input_3)
		out  = tf.layers.Dense(inputs=out_pre_regress, units = X_traffic.shape[1])

		loss = tf.losses.mean_squared_error(out, y_hat)
		optimizer = tf.optimizer.Adam().minimize(loss)

		with tf.Session() as sess:
			_, cost = sess.run([optimizer, loss], feed_dict={input_traffic: X_traffic, input_weather: X_wheater, y_hat: Y_traffic})






