
import tensorflow as tf
import keras.backend as K
from keras.utils.np_utils import to_categorical
from keras.models import Model, Sequential
from keras.layers import Input
from keras import layers
from keras import optimizers
import numpy as np
import matplotlib.pyplot as plt

tf.compat.v1.enable_eager_execution()


# define layer with masking weights
class MaskedLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, name='masked layer'):
        super(MaskedLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # dense layer initialization
        W_init = tf.random_normal_initializer(mean=0.0, stddev=0.05)
        b_init = tf.zeros_initializer()
        # default mask initialization
        m_init = tf.ones_initializer()
        # layer variables
        self.mask = tf.Variable(initial_value=m_init(shape=(input_dim, output_dim)), trainable=False, 
                                dtype='float32', name='mask')
        self.W = tf.Variable(initial_value=W_init(shape=(input_dim, output_dim)), trainable=True, 
                             dtype='float32', name='W')
        self.b = tf.Variable(initial_value=b_init(shape=(output_dim,)), trainable=True, dtype='float32', name='b')
    
    def call(self, x):
        return tf.matmul(tf.cast(x, tf.float32), tf.math.multiply(self.W, self.mask)) + self.b
    
    def update_mask(self, mask):
        self.mask.assign(mask)
        
    def reset_mask(self):
        self.mask.assig(tf.ones((self.input_dim, self.output_dim)))
        
    def get_layer_shape(self):
        return self.input_dim, self.output_dim


def make_one_hot(x, d):
    ndim = d*np.prod(x.shape[1:])
    x_ = tf.keras.utils.to_categorical(x, d)
    x_ = tf.reshape(x_, [-1, ndim])
    return x_


class MADE(tf.keras.Model):
    def __init__(self, d, input_shape, output_shape, hidden_layers=[256, 256, 256], use_one_hot = True):
        super(MADE, self).__init__()
        self.in_shape = input_shape
        self.hidden_layers = hidden_layers
        self.out_shape = output_shape
        # symbol length
        self.d = d
        # define dimensionality of input and output layers
        self.nin = np.prod(input_shape)
        self.nout = np.prod(output_shape)
        self.use_one_hot = use_one_hot

        # network
        self.net = []
        hs = [self.nin * self.d if self.use_one_hot else self.nin] + hidden_layers + \
             [self.nout * self.d if self.use_one_hot else self.nout]
        for h0, h1 in zip(hs, hs[1:]):
            self.net.append(MaskedLayer(h0, h1))
            self.net.append(tf.keras.layers.ReLU())
        # remove last ReLU, last Sigmoid activation used in the __call__ method
        self.net.pop()
        # update mask values
        self.masks = []
        self.nconn = {}
        self.ordering = None
        self.update_masks()
        
    def update_masks(self):
        # ordering of inputs
        self.ordering = np.arange(self.nin)
        np.random.shuffle(self.ordering)

        # this entry is the agnostic ordering
        self.nconn[-1] = self.ordering
        L = len(self.hidden_layers)
        # generate maximal number of connections for each hidden unit
        for i in range(L):
            self.nconn[i] = np.random.randint(self.nconn[i-1].min(), self.nin-1, self.hidden_layers[i])
        # construct masks for hidden layers
        self.masks = [self.nconn[i-1][:, np.newaxis] <= self.nconn[i][np.newaxis, :] for i in range(L)]
        self.masks.append(self.nconn[L-1][:, np.newaxis] < self.nconn[-1][np.newaxis, :])

        # replicate mask for one-hot encoding
        if self.use_one_hot:
            self.masks[0] = np.repeat(self.masks[0], self.d, axis=0)
            self.masks[-1] = np.repeat(self.masks[-1], self.d, axis=1)

        # iterate over 'maskable' layers
        imaskedlayer = [i for i in range(len(self.net)) if isinstance(self.net[i], MaskedLayer)]
        for mask, layer in zip(self.masks, imaskedlayer):
            self.net[layer].update_mask(mask)
            
    def loss(self, x):
        logits, _ = self(x)

        if self.use_one_hot:
            logits = tf.reshape(logits, [-1, self.nin, self.d])
            loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=x, logits=logits))
        else:
            x = tf.reshape(tf.cast(x, tf.float32), shape=[-1, self.nin])
            loss_ = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=logits))
        return loss_

    def call(self, x):
        if self.use_one_hot:
            x = make_one_hot(x, self.d)
        else:
            x = tf.reshape(x, shape=[-1, self.nin])
        logits = self.net[0](x)
        for layer in self.net[1:]:
            logits = layer(logits)
        if self.use_one_hot:
            x_out = tf.math.sigmoid(tf.reshape(logits, shape=[-1, self.d, self.nin]))
        else:
            x_out = tf.math.sigmoid(logits)
        return logits, x_out
    
    def get_distribution(self):
        grid = np.mgrid[0:self.d, 0:self.d].reshape(np.prod(self.in_shape), self.d**2).T
        logits, _ = self(grid)
        logits = tf.reshape(logits, [-1, self.nin, self.d])
        logits = tf.transpose(logits, perm=[0, 2, 1])
        logits = tf.math.log_softmax(logits, axis=1)
        dist = np.zeros((tf.shape(logits)[0], self.nin))
        for i in range(tf.shape(logits)[0]):
            dist[i][0] = logits[i][grid[i,0]][0]
            dist[i][1] = logits[i][grid[i,1]][1]
        dist = tf.reduce_sum(dist, axis=1)
        dist = tf.math.exp(tf.reshape(dist, shape=[self.d, self.d]))
        return dist

    def get_samples(self, n):
        samples = np.zeros((n, self.nin))
        for i in range(self.nin):
            logits, _= self(samples)
            prob = tf.math.sigmoid(logits[:, self.ordering[i]])
            samples[:, self.ordering[i]] = tf.keras.backend.random_binomial(shape=[n,], p=prob) #tf.squeeze(b)
        return tf.keras.backend.eval(tf.reshape(samples, shape=[-1, *self.in_shape]))


def train_model(model, train_data, test_data, params):
    (epochs, lr, batch_size, d, use_one_hot) = params

    train_loss = []
    test_loss = []
        
    # first test loss without training
    loss = model.loss(test_data)
    test_loss.append(np.mean(tf.keras.backend.eval(loss)))
    
    # define opimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    # main loop    
    for iepoch in range(epochs):
        i = np.array(range(train_data.shape[0]))
        # Stochastic selection of training data
        np.random.shuffle(i)
        for n in range(train_data.shape[0]//batch_size):
            data_range = i[n*batch_size:(n+1)*batch_size]
            x = train_data[data_range]
            with tf.GradientTape() as tape:
                loss = model.loss(x)
                train_loss.append(np.mean(tf.keras.backend.eval(loss)))
            # compute gradients and update model's parameters      
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # test model
        loss = tf.keras.backend.eval(model.loss(test_data))
        test_loss.append(loss)
        
        print('### Epoch: {} / {} =====> Train loss: {} ### Test loss: {}'.format(iepoch+1, epochs, train_loss[-1], test_loss[-1]))
    return train_loss, test_loss


def q2_a(train_data, test_data, d, dset_id):
  """
  train_data: An (n_train, 2) numpy array of integers in {0, ..., d-1}
  test_data: An (n_test, 2) numpy array of integers in {0, .., d-1}
  d: The number of possible discrete values for each random variable x1 and x2
  dset_id: An identifying number of which dataset is given (1 or 2). Most likely
           used to set different hyperparameters for different datasets

  Returns
  - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
  - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
  - a numpy array of size (d, d) of probabilities (the learned joint distribution)
  """
  
  """ YOUR CODE HERE """
  lr = 2e-3
  epochs = 20
  batch_size = 128
  # reproducible results
  tf.set_random_seed(42)
  model = MADE(d, input_shape=(2,), output_shape=(2,), hidden_layers=[100, 100], use_one_hot=True)
  train_loss, test_loss = train_model(model, train_data, test_data, (epochs, lr, batch_size, d, True))
  distribution = model.get_distribution()
  return train_loss[1:], test_loss, distribution


#hw1.visualize_q2a_data(dset_type=1, data_dir='.//data')
#hw1.visualize_q2a_data(dset_type=2, data_dir='.//data')

#hw1.q2_save_results(1, 'a', q2_a, data_dir='.//data')

#hw1.q2_save_results(2, 'a', q2_a, data_dir='.//data')

def q2_b(train_data, test_data, image_shape, dset_id):
    """
    train_data: A (n_train, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    test_data: An (n_test, H, W, 1) uint8 numpy array of binary images with values in {0, 1}
    image_shape: (H, W), height and width of the image
    dset_id: An identifying number of which dataset is given (1 or 2). Most likely
             used to set different hyperparameters for different datasets

    Returns
    - a (# of training iterations,) numpy array of train_losses evaluated every minibatch
    - a (# of epochs + 1,) numpy array of test_losses evaluated once at initialization and after each epoch
    - a numpy array of size (100, H, W, 1) of samples with values in {0, 1}
    """

    train_data = np.transpose(train_data, (0, 3, 1, 2))
    test_data = np.transpose(test_data, (0, 3, 1, 2))
    lr = 1e-3
    epochs = 100
    batch_size = 128

    (H, W) = image_shape
    d = 1
    model = MADE(d, input_shape=(1, H, W),  output_shape=(1, H, W), hidden_layers=[256, 256, 256, 256], use_one_hot=False)
    tf.random.get_seed(42)
    train_loss, test_loss = train_model(model, train_data, test_data, (epochs, lr, batch_size, d, False))
    samples = model.get_samples(n=100)
    samples = np.transpose(samples, (0, 2, 3, 1))
    return train_loss[1:], test_loss, samples


#hw1.visualize_q2b_data(dset_type=1, data_dir='.//data')
#hw1.visualize_q2b_data(dset_type=2, data_dir='.//data')

hw1.q2_save_results(1, 'b', q2_b, data_dir='.//data')
hw1.q2_save_results(2, 'b', q2_b, data_dir='.//data')

