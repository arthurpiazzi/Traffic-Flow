from DBN import DBN
import tensorflow as tf 
from base import Base

class DBN_System_1(Base):
	
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
		with tf.Session() as sess:
			self.dbn_3.fit(input_3.eval(), transform = False)

	def create_greed_model(self):
		pass

	def train_greed(self, X_traffic, X_weather, Y_traffic, X_traffic_test = None, X_wheater_test = None, Y_traffic_test = None, n_epoch = 1000, dropout = 0.1, batch = 100):
		print("____Training Greed")

		input_traffic = tf.placeholder('float', [None, X_traffic.shape[1]])
		input_weather = tf.placeholder('float', [None, X_weather.shape[1]])
		y_hat = tf.placeholder('float', [None, Y_traffic.shape[1]])

		self.dbn_1.get_greed_model(input_traffic, dropout = dropout)


		out_2 = self.dbn_2.get_greed_model(input_weather, dropout = dropout)

		input_3 = tf.concat([out_1, out_2], axis=1)

		out_pre_regress = self.dbn_3.get_greed_model(input_3, dropout = dropout)

		weights = tf.Variable(tf.truncated_normal([out_pre_regress.get_shape().as_list()[1], Y_traffic.shape[1]]))
		biases = tf.Variable(tf.zeros([Y_traffic.shape[1]]))

		out  = (tf.matmul(out_pre_regress, weights) + biases)

		loss = tf.reduce_mean(tf.squared_difference(out, y_hat))
		optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)
		error_best = 0
		epoch_best = 0
		saver = tf.train.Saver()
		patience = 0


		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			for i in range(n_epoch):
				epoch_loss = 0 
				for start, end in zip(range(0, X_traffic.shape[0], batch),range(batch, X_traffic.shape[0], batch)):
					traffic_batch = X_traffic.loc[start:end]
					weather_batch = X_weather.loc[start:end]
					y_bacth = Y_traffic.loc[start:end]

					_, cost = sess.run([optimizer, loss], feed_dict={input_traffic: traffic_batch, input_weather: weather_batch, y_hat: y_bacth})
					epoch_loss += cost
				val_loss = loss.eval(feed_dict={input_traffic: X_traffic_test, input_weather: X_wheater_test, y_hat: Y_traffic_test})
				if val_loss <= error_best or i == 0:
					error_best = val_loss
					epoch_best = i
					star = '*'
					self.save('model.pkl')
					patience = 0
				else:
					patience += 1
					star = ''
					if patience >= 20: break
				print('Epoch: {}, train error: {}, validation loss: {},  best: {} at epoch: {} {}'.format(i, epoch_loss, val_loss,error_best, epoch_best, star))

		
	def score(self):
		pass 



