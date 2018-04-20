

class Base(object):


	def save(self, path = 'model.pkl'):
		import pickle as pkl
		import os
		from misc import query_yes_no

		if os.path.exists(path):
			if query_yes_no("The path already exists! Do you want to overwrite?"):
				pkl.dump(self, open(path, 'wb'))

	@classmethod
	def load(cls, path):
		import pickle as pkl 

		return pkl.load(open(path, 'rb'))
