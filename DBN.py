from RBM import RBM
from base import Base


class DBN(Base):
    def __init__(self, params, n_iter ):
        self.n_layers = len(params)
        self.params = params
        print(n_iter)
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

    def get_greed_model(self, X):
        
        _layers_array = [None]*self.n_layers
        _layers_array[0] = X
        for i, layer in enumerate(self.layer, start = 1):
            _layers_array[i] = tf.nn.sigmoid(tf.matmul(_layers_array[i-1], tf.Variable(layer.w_best)) + layer.hb_best)

        return _layers_array[-1]


