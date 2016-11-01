# file kfkd.py
import os
import sys

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import theano
import cPickle as pickle

FTRAIN = 'training.csv'
FTEST = 'test.csv'
FLOOKUP = 'IdLookupTable.csv'

sys.setrecursionlimit(10000)  # for pickle...
np.random.seed(42)


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    print(df.count())  # prints the number of values for each column
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

def input_dims():
    X, y = load()
    print("X.shape == {0}; X.min == {1:.3f}; X.max == {2:.3f}".format(
        X.shape, X.min(), X.max()))
    print("y.shape == {0}; y.min == {1:.3f}; y.max == {2:.3f}".format(
        y.shape, y.min(), y.max()))

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def fit1(plot=False, epochs=400, save_to=None):
    net = NeuralNet(
        layers=[  # three layers: one hidden layer
            ('input', layers.InputLayer),
            ('hidden', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        # layer parameters:
        input_shape=(None, 9216),  # 96x96 input pixels per batch
        hidden_num_units=100,  # number of units in hidden layer
        output_nonlinearity=None,  # output layer uses identity function
        output_num_units=30,  # 30 target values

        # optimization method:
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,  # flag to indicate we're dealing with regression problem
        max_epochs=epochs,  # we want to train this many epochs
        verbose=1,
        )

    X, y = load()
    net.fit(X, y)
    if save_to:
        save_net(net, save_to)
    print_kaggle_measure(net)
    if plot:
        plot_net(net)
    return net

def fit2(plot=False, epochs=1000, save_to=None):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=30, output_nonlinearity=None,
        
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        max_epochs=epochs,
        verbose=1,
        )

    X, y = load2d()  # load 2-d data
    net.fit(X, y)

    if save_to:
        save_net(net, save_to)
    print_kaggle_measure(net)
    if plot:
        plot_net(net)
    return net

from nolearn.lasagne import BatchIterator

class FlipBatchIterator(BatchIterator):
    flip_columns = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_columns:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

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
    
    @staticmethod
    def _rotation_matrix(angle, inverse=False):
        c, s = np.cos(angle), np.sin(angle)
        cs = np.abs(c) + np.abs(s)
        m = np.array([[c, -s], [s, c]])
        return m.T / cs if inverse else m * cs

    def transform(self, Xb, yb):
        Xb, yb = super(RotationFlipBatchIterator, self).transform(Xb, yb)
        Xb = Xb.copy()
        yb = yb.copy()
        
        bs = Xb.shape[0]
        # rotate 1/3 of the images in this batch in each direction,
        # unless positions in yb get out of range
        rots = len(self.rotations)
        divisions = np.array([r*bs/rots for r in range(1,rots)])
        rot_indices = np.split(np.random.permutation(bs),divisions)
        
        if yb is not None:
            for i, rotation in enumerate(self.rotations):
                # transform coordinates of keypoints
                yb_new = rotation.transform_y(yb[rot_indices[i]])
                # make sure that no keypoints vanish with rotation
                valid_y = np.abs(yb_new).max(axis=1) <= 1
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
	
    @staticmethod
    def _rotation_matrix(angle, inverse=False):
        c, s = np.cos(angle), np.sin(angle)
        cs = np.abs(c) + np.abs(s)
        m = np.array([[c, -s], [s, c]])
        return m.T / cs if inverse else m * cs

    def transform_y(self, y):
        bs = y.shape[0]
        if bs == 0: return y # otherwise reshape with -1 fails
        y_new = y.reshape(bs, -1, 2)
        y_new = np.tensordot(y_new, self.y_rotation, axes=(2,0))
        return y_new.reshape(bs, -1)
    
    def transform_X(self, X):
        hrot, vrot = self.rot_trunc
        hrot_, vrot_ = np.minimum(self.rot_trunc, 94) + 1 # avoids out of bounds exceptions
        hfact, vfact = self.factors
        return X[:, :,hrot, vrot] * (1-hfact) * (1-vfact) +\
            X[:, :,hrot_, vrot] * hfact * (1-vfact) +\
            X[:, :,hrot, vrot_] * (1-hfact) * vfact +\
            X[:, :,hrot_, vrot_] * hfact * vfact
    
    def transform(self, X, y):
        return self.transform_X(X), self.transform_y(y)
    
def fit3(plot=False, epochs=3000, save_to=None):
    net = NeuralNet(
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('conv3', layers.Conv2DLayer),
            ('pool3', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('hidden5', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 96, 96),
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
        conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
        conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
        hidden4_num_units=500, hidden5_num_units=500,
        output_num_units=30, output_nonlinearity=None,
        
        update=nesterov_momentum,
        update_learning_rate=0.01,
        update_momentum=0.9,

        regression=True,
        batch_iterator_train=FlipBatchIterator(batch_size=128),
        max_epochs=epochs,
        verbose=1,
        )

    X, y = load2d()  # load 2-d data
    net.fit(X, y)

    if save_to:
        save_net(net, save_to)
    print_kaggle_measure(net)
    if plot:
        plot_net(net)
    return net

def fit4(plot=False, epochs=3000, save_to=None):
    net = neural_net(epochs)

    X, y = load2d()  # load 2-d data
    net.fit(X, y)

    if save_to:
        save_net(net, save_to)
    print_kaggle_measure(net)
    if plot:
        plot_net(net)
    return net

SPECIALIST_SETTINGS = [
    dict(
        columns=(
            'left_eye_center_x', 'left_eye_center_y',
            'right_eye_center_x', 'right_eye_center_y',
            ),
        flip_columns=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'nose_tip_x', 'nose_tip_y',
            ),
        flip_columns=(),
        ),

    dict(
        columns=(
            'mouth_left_corner_x', 'mouth_left_corner_y',
            'mouth_right_corner_x', 'mouth_right_corner_y',
            'mouth_center_top_lip_x', 'mouth_center_top_lip_y',
            ),
        flip_columns=((0, 2), (1, 3)),
        ),

    dict(
        columns=(
            'mouth_center_bottom_lip_x',
            'mouth_center_bottom_lip_y',
            ),
        flip_columns=(),
        ),

    dict(
        columns=(
            'left_eye_inner_corner_x', 'left_eye_inner_corner_y',
            'right_eye_inner_corner_x', 'right_eye_inner_corner_y',
            'left_eye_outer_corner_x', 'left_eye_outer_corner_y',
            'right_eye_outer_corner_x', 'right_eye_outer_corner_y',
            ),
        flip_columns=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),

    dict(
        columns=(
            'left_eyebrow_inner_end_x', 'left_eyebrow_inner_end_y',
            'right_eyebrow_inner_end_x', 'right_eyebrow_inner_end_y',
            'left_eyebrow_outer_end_x', 'left_eyebrow_outer_end_y',
            'right_eyebrow_outer_end_x', 'right_eyebrow_outer_end_y',
            ),
        flip_columns=((0, 2), (1, 3), (4, 6), (5, 7)),
        ),
    ]

from collections import OrderedDict

def fit_specialists(plot=False, fname_pretrain=None, net_pretrain=None,
        save_to=None, settings=SPECIALIST_SETTINGS,
        specialists=OrderedDict()):
    from sklearn.base import clone

    if fname_pretrain and not net_pretrain:
        net_pretrain = load_net(fname_pretrain)

    for setting in settings:
        cols = setting['columns']
        X, y = load2d(cols=cols)

        model = neural_net(epochs=int(4e6 / y.shape[0]), output_units=y.shape[1])
        model.batch_iterator_train.flip_columns = setting['flip_columns']
        if 'kwargs' in setting:
            # an option 'kwargs' in the settings list may be used to
            # set any other parameter of the net:
            vars(model).update(setting['kwargs'])

        if net_pretrain is not None:
            # if a pretrain model was given, use it to initialize the
            # weights of our new specialist model:
            model.load_params_from(net_pretrain)

        print("Training model for columns {0} for {1} epochs".format(
            cols, model.max_epochs))
        model.fit(X, y)
        specialists[cols] = model
    
    if save_to:
        save_net(specialists, save_to)
    if plot:
        plot_learning_curves(specialists)

def predict(specialists, save_to='submission.csv'):
    from pandas import DataFrame
    X = load2d(test=True)[0]
    y_pred = np.empty((X.shape[0], 0))

    for model in specialists.values():
        y_pred1 = model.predict(X)
        y_pred = np.hstack([y_pred, y_pred1])

    columns = ()
    for cols in specialists.keys():
        columns += cols

    y_pred2 = y_pred * 48 + 48
    y_pred2 = y_pred2.clip(0, 96)
    df = DataFrame(y_pred2, columns=columns)

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

def neural_net(epochs=1000, output_units=30, initial_rate=0.04):
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
        conv1_num_filters=24, conv1_filter_size=(5, 5), pool1_pool_size=(2, 2),
        dropout1_p=0.1,
        conv2_num_filters=48, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.2,
        conv3_num_filters=96, conv3_filter_size=(3, 3), pool3_pool_size=(2, 2),
        dropout3_p=0.3,
        hidden4_num_units=1000,
        dropout4_p=0.5,
        hidden5_num_units=750,
        output_num_units=output_units, output_nonlinearity=None,
        
        update=nesterov_momentum,
        update_learning_rate=theano.shared(float32(initial_rate)),
        update_momentum=theano.shared(float32(0.9)),

        regression=True,
        batch_iterator_train=RotationFlipBatchIterator(batch_size=128),
        on_epoch_finished=[
            LinearAdjustVariable('update_learning_rate', start=initial_rate, stop=0.0001),
            LinearAdjustVariable('update_momentum', start=0.9, stop=0.999),
            EarlyStopping(patience=200),
        ],
        max_epochs=epochs,
        verbose=1,
        )

class LinearAdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
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

def plot_net(net):
    from matplotlib import pyplot
    train_loss = np.array([i["train_loss"] for i in net.train_history_])
    valid_loss = np.array([i["valid_loss"] for i in net.train_history_])
    pyplot.plot(train_loss, linewidth=3, label="train")
    pyplot.plot(valid_loss, linewidth=3, label="valid")
    pyplot.grid()
    pyplot.legend()
    pyplot.xlabel("epoch")
    pyplot.ylabel("loss")
    pyplot.ylim(1e-4, 1e-2)
    pyplot.yscale("log")
    try:
        pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def plot_learning_curves(models):
    from matplotlib import pyplot
    
    fig = pyplot.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_color_cycle(
        ['c', 'c', 'm', 'm', 'y', 'y', 'k', 'k', 'g', 'g', 'b', 'b'])

    valid_losses = []
    train_losses = []

    for model_number, (cg, model) in enumerate(models.items(), 1):
        valid_loss = np.array([i['valid_loss'] for i in model.train_history_])
        train_loss = np.array([i['train_loss'] for i in model.train_history_])
        valid_loss = np.sqrt(valid_loss) * 48
        train_loss = np.sqrt(train_loss) * 48

        valid_loss = rebin(valid_loss, (100,))
        train_loss = rebin(train_loss, (100,))

        valid_losses.append(valid_loss)
        train_losses.append(train_loss)
        ax.plot(valid_loss,
                label='{} ({})'.format(cg[0], len(cg)), linewidth=3)
        ax.plot(train_loss,
                linestyle='--', linewidth=3, alpha=0.6)
        ax.set_xticks([])

    weights = np.array([m.output_num_units for m in models.values()],
                       dtype=float)
    weights /= weights.sum()
    mean_valid_loss = (
        np.vstack(valid_losses) * weights.reshape(-1, 1)).sum(axis=0)
    ax.plot(mean_valid_loss, color='r', label='mean', linewidth=4, alpha=0.8)

    ax.legend()
    ax.set_ylim((1.0, 4.0))
    ax.grid()
    pyplot.ylabel("RMSE")
    try:
        pyplot.show()
    except RuntimeError as e:
        print "Unable to show plot", e

def rebin( a, newshape ):
    from numpy import mgrid
    assert len(a.shape) == len(newshape)

    slices = [ slice(0,old, float(old)/new) for old,new in zip(a.shape,newshape) ]
    coordinates = mgrid[slices]
    indices = coordinates.astype('i')   #choose the biggest smaller integer index
    return a[tuple(indices)]

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

def print_kaggle_measure(net):
    print("Kaggle loss: {0:.4f}".format(np.sqrt(net.train_history_[-1]['valid_loss'])*48))
