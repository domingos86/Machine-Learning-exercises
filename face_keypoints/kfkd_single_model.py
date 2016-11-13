# file kfkd_single_model.py

'''
    Based mostly on the code found in https://github.com/dnouri/kfkd-tutorial
    
    To run, simply do
    
    >>> import import kfkd_single_model as kfkd
    >>> net = kfkd.fit()
    
    Furthermore, if you wish to save the predictions for the test set,
    you can do
    
    >>> kfkd.predict(net, save_to='filename.csv')
    
    If using GPU, should open pyhton interperter with
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python
'''

import os
import sys

import numpy as np
from pandas.io.parsers import read_csv
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet, BatchIterator
import theano
import cPickle as pickle

FTRAIN = 'training.csv'
FTEST = 'test.csv'
FLOOKUP = 'IdLookupTable.csv'

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)

print("Using device " + theano.config.device)


def load(test=False, cols=None, dropna=True):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns. If dropna=False, rows with missing labels won't be
    removed.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
    if dropna:
        df = df.dropna()  # drop all rows that have missing values in them

    X = np.vstack(df['Image'].values) / 255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y
    
def load_columns():
    df = read_csv(os.path.expanduser(FTRAIN))
    del df['Image']
    return df.columns

def input_dims():
    X, y = load()
    print("X.shape == {0}; X.min == {1:.3f}; X.max == {2:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {0}; y.min == {1:.3f}; y.max == {2:.3f}".format(
        y.shape, y.min(), y.max()))

def load2d(test=False, cols=None, dropna=True):
    X, y = load(test=test, cols=cols, dropna=dropna)
    X = X.reshape(-1, 1, 96, 96)
    return X, y
   
def shuffle(*arrays, **options):
    random_state = options.pop('random_state', None)
    random_state = np.random.RandomState(random_state)
    
    n_samples = arrays[0].shape[0]
    indices = np.arange(n_samples)
    random_state.shuffle(indices)
    return [a[indices] for a in arrays]

class RotationFlipBatchIterator(BatchIterator):
    def __init__(self, batch_size, angle=5, multiples=4, **kwargs):
        super(RotationFlipBatchIterator, self).__init__(batch_size, **kwargs)
        self.rotations = [Rotation(angle*i) for i in range(-multiples, multiples+1)]
        self.flip_columns = [
            (0, 2), (1, 3),
            (4, 8), (5, 9), (6, 10), (7, 11),
            (12, 16), (13, 17), (14, 18), (15, 19),
            (22, 24), (23, 25),
            ]

    def transform(self, Xb, yb):
        Xb, yb = super(RotationFlipBatchIterator, self).transform(Xb, yb)
        Xb = Xb.copy()
        yb = yb.copy() if yb is not None else None
                
        bs = Xb.shape[0]
        
        if yb is not None:
            # rotate 1/3 of the images in this batch in each direction,
            # unless positions in yb get out of range
            rots = len(self.rotations)
            divisions = np.array([r*bs/rots for r in range(1,rots)])
            rot_indices = np.split(np.random.permutation(bs),divisions)
            for i, rotation in enumerate(self.rotations):
                # transform coordinates of keypoints
                yb_new = rotation.transform_y(yb[rot_indices[i]])
                # make sure that no keypoints vanish with rotation
                valid_y = np.nanmax(np.abs(yb_new),axis=1) <= 1
                indices = rot_indices[i][valid_y]
                # update yb
                yb[indices] = yb_new[valid_y]
                # rotate the image (and interpolate)
                Xb[indices] = rotation.transform_X(Xb[indices])

        # Flip half of the images in this batch at random:
        
        flip_indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[flip_indices] = Xb[flip_indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[flip_indices, ::2] = yb[flip_indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_columns:
                yb[flip_indices, a], yb[flip_indices, b] = (
                    yb[flip_indices, b], yb[flip_indices, a])

        return Xb, yb

class Rotation(object):
    def __init__(self, angle):
        r_angle = angle * np.pi / 180
        self.y_rotation = self._rotation_matrix(r_angle)
        inv_rotation = self._rotation_matrix(r_angle, inverse=True)
        positions = np.indices((96,96), dtype=np.float)
        positions -= 47.5 # center point of all pixels
        positions = np.tensordot(inv_rotation, positions ,axes=(1,0))
        positions += 47.5
        
        self.rot_trunc = positions.astype(np.int32)
        self.factors = positions - self.rot_trunc
        self.rot_ceil = np.minimum(self.rot_trunc, 94) + 1 # avoids out of bounds exceptions
	
    @staticmethod
    def _rotation_matrix(angle, inverse=False):
        c, s = np.cos(angle), np.sin(angle)
        cs = np.abs(c) + np.abs(s)
        m = np.array([[c, -s], [s, c]])
        return m.T / cs if inverse else m * cs

    def transform_y(self, y):
        y_new = y.reshape((y.shape[0], y.shape[1]//2, 2))
        y_new = np.tensordot(y_new, self.y_rotation, axes=(2,0))
        return y_new.reshape(y.shape)
    
    def transform_X(self, X):
        hrt, vrt = self.rot_trunc
        hrc, vrc = self.rot_ceil
        hfact, vfact = self.factors
        return X[:, :,hrt, vrt] * (1-hfact) * (1-vfact) +\
            X[:, :,hrc, vrt] * hfact * (1-vfact) +\
            X[:, :,hrt, vrc] * (1-hfact) * vfact +\
            X[:, :,hrc, vrc] * hfact * vfact
    
    def transform(self, X, y):
        return self.transform_X(X), self.transform_y(y)

class WeightedSquareLoss(object):
    def __init__(self, weights):
        weights = np.asarray(weights, dtype=theano.config.floatX)
        weights = weights.reshape((1,) + weights.shape)
        self.weights = theano.tensor.addbroadcast(theano.shared(weights), 0)
        self.zeros = theano.tensor.addbroadcast(theano.shared(
            np.zeros(weights.shape, dtype=theano.config.floatX)), 0)
    
    def __call__(self, a, b):
        return theano.tensor.switch(theano.tensor.isnan(b),
            self.zeros,
            ((a - b) ** 2) * self.weights)

class LogAdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(np.log(self.start), np.log(self.stop), nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(np.exp(self.ls[epoch - 1]))
        getattr(nn, self.name).set_value(new_value)

class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = nn.get_all_params_values()
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_params_from(self.best_weights)
            raise StopIteration()

def neural_net(epochs=2000, output_units=30, initial_rate=0.03, weights=None):
    if weights is None:
        weights = np.ones(output_units, theano.config.floatX)
    return NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('dropout1', layers.DropoutLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('dropout2', layers.DropoutLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('dropout3', layers.DropoutLayer),
            ('hidden4', layers.DenseLayer),
            ('dropout4', layers.DropoutLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=48, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=96, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=192, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=1500,
        dropout4_p=0.5,
        hidden5_num_units=1000,
        output_num_units=output_units, output_nonlinearity=None,
        
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(initial_rate)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,
        objective_loss_function=WeightedSquareLoss(weights),
        batch_iterator_train=RotationFlipBatchIterator(batch_size=128),
        on_epoch_finished=[
            LogAdjustVariable('update_learning_rate', start=initial_rate, stop=0.0001),
            LogAdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=50),
        ],
        max_epochs=epochs,
        verbose=1,
        )

def fit(plot=False, epochs=1000, save_to=None):
    '''Trains a neural network for all the labels.
    
    It only uses inputs without missing labels.
    
    Returns: trained neural network (nolearn.lasagne.NeuralNet)
    
    Keyword arguments:
    plot -- (bool, False) if true, a plot of the training and validation
        errors at the end of each epoch will be shown once the network
        finishes training.
    epochs -- (int, 3000) the maximum number of epochs for which the
        network should train.
    save_to -- (str, None) name of the file to which the network will be
        pickled.
    '''
    
    X, y = load2d(dropna=False)
      # load 2-d data
    #X = theano.shared(X.astype(theano.config.floatX), borrow=True)
    #y = theano.shared(y.astype(theano.config.floatX), borrow=True)
    
    net = neural_net(epochs,
        weights = y.shape[0]/np.isfinite(y).sum(axis=0))
    
    net.fit(X, y)

    if save_to:
        save_net(net, save_to)
    print_kaggle_measure(net)
    if plot:
        plot_net(net)
    return net

def predict(net, save_to='submission.csv'):
    from pandas import DataFrame
    X = load2d(test=True)[0]
    y_pred = (net.predict(X)*48 + 48).clip(0,96)

    df = DataFrame(y_pred, columns=load_columns())

    lookup_table = read_csv(os.path.expanduser(FLOOKUP))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    submission = DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv(save_to, index=False)
    print("Wrote {}".format(save_to))

def plot_net(net):
    try:
        from matplotlib import pyplot
        train_loss = RMSE(np.array([i["train_loss"] for i in net.train_history_]))
        valid_loss = RMSE(np.array([i["valid_loss"] for i in net.train_history_]))
        pyplot.plot(train_loss, linewidth=3, label="train")
        pyplot.plot(valid_loss, linewidth=3, label="valid")
        pyplot.grid()
        pyplot.legend()
        pyplot.xlabel("epoch")
        pyplot.ylabel("Root-mean-square error (RMSE)")
        pyplot.ylim(np.min(train_loss.min(), valid_loss.min()), np.max(train_loss.max(), valid_loss.max()))
        pyplot.yscale("log")
        pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def plot_samples(net, m, n, _load2d=True):
    from matplotlib import pyplot
    X, _ = load2d(test=True) if _load2d else load(test=True)
    y_pred = net.predict(X)

    fig = pyplot.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(m*n):
        ax = fig.add_subplot(m, n, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y_pred[i], ax)

    try:
        pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

def save_net(net, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(net, f, -1)

def load_net(file_name):
    with open(file_name, 'rb') as f:
        net = pickle.load(f)
    return net

def float32(k):
    return np.cast['float32'](k)

def RMSE(loss):
    return 48*np.sqrt(loss)

def print_kaggle_measure(net):
    print("RMSE: {0:.4f}".format(RMSE(net.train_history_[-1]['valid_loss'])))
