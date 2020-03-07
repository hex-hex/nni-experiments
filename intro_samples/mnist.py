"""A deep MNIST classifier using convolutional layers."""

import argparse
import collections
import gzip
import logging
import math
import os
import tempfile
import time
import urllib
import numpy

import tensorflow.compat.v1 as tf

from six.moves import xrange  # pylint: disable=redefined-builtin
from tensorflow.python.util.deprecation import deprecated
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import random_seed
from tensorflow.python.platform import gfile

import nni

tf.disable_v2_behavior()

FLAGS = None
logger = logging.getLogger('mnist_AutoML')

DEFAULT_SOURCE_URL = 'https://storage.googleapis.com/cvdf-datasets/mnist/'

_Datasets = collections.namedtuple('_Datasets', ['train', 'validation', 'test'])


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


@deprecated(None, 'Please use tf.one_hot on tensors.')
def _dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = numpy.arange(num_labels) * num_classes
    labels_one_hot = numpy.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


class _DataSet(object):
    """Container class for a _DataSet (deprecated).

    THIS CLASS IS DEPRECATED.
    """

    @deprecated(None, 'Please use alternatives such as official/mnist/_DataSet.py'
                      ' from tensorflow/models.')
    def __init__(self,
                 images,
                 labels,
                 fake_data=False,
                 one_hot=False,
                 dtype=dtypes.float32,
                 reshape=True,
                 seed=None):
        """Construct a _DataSet.

        one_hot arg is used only if fake_data is true.  `dtype` can be either
        `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
        `[0, 1]`.  Seed arg provides for convenient deterministic testing.

        Args:
          images: The images
          labels: The labels
          fake_data: Ignore inages and labels, use fake data.
          one_hot: Bool, return the labels as one hot vectors (if True) or ints (if
            False).
          dtype: Output image dtype. One of [uint8, float32]. `uint8` output has
            range [0,255]. float32 output has range [0,1].
          reshape: Bool. If True returned images are returned flattened to vectors.
          seed: The random seed to use.
        """
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(dtype).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                            dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                    'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                assert images.shape[3] == 1
                images = images.reshape(images.shape[0],
                                        images.shape[1] * images.shape[2])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                images = images.astype(numpy.float32)
                images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in xrange(batch_size)
                    ], [fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate((images_rest_part, images_new_part),
                                     axis=0), numpy.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]


@deprecated(None, 'Please write your own downloading logic.')
def _maybe_download(filename, work_directory, source_url):
    """Download the data from source url, unless it's already here.

    Args:
        filename: string, name of the file in the directory.
        work_directory: string, path to working directory.
        source_url: url to download from if file doesn't exist.

    Returns:
        Path to resulting file.
    """
    if not gfile.Exists(work_directory):
        gfile.MakeDirs(work_directory)
    filepath = os.path.join(work_directory, filename)
    if not gfile.Exists(filepath):
        urllib.request.urlretrieve(source_url, filepath)
        with gfile.GFile(filepath) as f:
            size = f.size()
        print('Successfully downloaded', filename, size, 'bytes.')
    return filepath


@deprecated(None, 'Please use tf.data to implement this functionality.')
def _extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


@deprecated(None, 'Please use tf.data to implement this functionality.')
def _extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    print('Extracting', f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        if one_hot:
            return _dense_to_one_hot(labels, num_classes)
        return labels


@deprecated(None, 'Please use alternatives such as:'
                  ' tensorflow_datasets.load(\'mnist\')')
def read_data_sets(train_dir,
                   fake_data=False,
                   one_hot=False,
                   dtype=dtypes.float32,
                   reshape=True,
                   validation_size=5000,
                   seed=None,
                   source_url=DEFAULT_SOURCE_URL):
    if fake_data:
        def fake():
            return _DataSet([], [],
                            fake_data=True,
                            one_hot=one_hot,
                            dtype=dtype,
                            seed=seed)

        train = fake()
        validation = fake()
        test = fake()
        return _Datasets(train=train, validation=validation, test=test)

    if not source_url:  # empty string check
        source_url = DEFAULT_SOURCE_URL

    train_images_file = 'train-images-idx3-ubyte.gz'
    train_labels_file = 'train-labels-idx1-ubyte.gz'
    test_images_file = 't10k-images-idx3-ubyte.gz'
    test_labels_file = 't10k-labels-idx1-ubyte.gz'

    local_file = _maybe_download(train_images_file, train_dir,
                                 source_url + train_images_file)
    with gfile.Open(local_file, 'rb') as f:
        train_images = _extract_images(f)

    local_file = _maybe_download(train_labels_file, train_dir,
                                 source_url + train_labels_file)
    with gfile.Open(local_file, 'rb') as f:
        train_labels = _extract_labels(f, one_hot=one_hot)

    local_file = _maybe_download(test_images_file, train_dir,
                                 source_url + test_images_file)
    with gfile.Open(local_file, 'rb') as f:
        test_images = _extract_images(f)

    local_file = _maybe_download(test_labels_file, train_dir,
                                 source_url + test_labels_file)
    with gfile.Open(local_file, 'rb') as f:
        test_labels = _extract_labels(f, one_hot=one_hot)

    if not 0 <= validation_size <= len(train_images):
        raise ValueError(
            'Validation size should be between 0 and {}. Received: {}.'.format(
                len(train_images), validation_size))

    validation_images = train_images[:validation_size]
    validation_labels = train_labels[:validation_size]
    train_images = train_images[validation_size:]
    train_labels = train_labels[validation_size:]

    options = dict(dtype=dtype, reshape=reshape, seed=seed)

    train = _DataSet(train_images, train_labels, **options)
    validation = _DataSet(validation_images, validation_labels, **options)
    test = _DataSet(test_images, test_labels, **options)

    return _Datasets(train=train, validation=validation, test=test)


class MnistNetwork(object):
    '''
    MnistNetwork is for initializing and building basic network for mnist.
    '''

    def __init__(self,
                 channel_1_num,
                 channel_2_num,
                 conv_size,
                 hidden_size,
                 pool_size,
                 learning_rate,
                 x_dim=784,
                 y_dim=10):
        self.channel_1_num = channel_1_num
        self.channel_2_num = channel_2_num
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        self.learning_rate = learning_rate
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.images = tf.placeholder(
            tf.float32, [None, self.x_dim], name='input_x')
        self.labels = tf.placeholder(
            tf.float32, [None, self.y_dim], name='input_y')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.train_step = None
        self.accuracy = None

    def build_network(self):
        '''
        Building network for mnist
        '''

        # Reshape to use within a convolutional neural net.
        # Last dimension is for "features" - there is only one here, since images are
        # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
        with tf.name_scope('reshape'):
            try:
                input_dim = int(math.sqrt(self.x_dim))
            except:
                print(
                    'input dim cannot be sqrt and reshape. input dim: ' + str(self.x_dim))
                logger.debug(
                    'input dim cannot be sqrt and reshape. input dim: %s', str(self.x_dim))
                raise
            x_image = tf.reshape(self.images, [-1, input_dim, input_dim, 1])

        # First convolutional layer - maps one grayscale image to 32 feature maps.
        with tf.name_scope('conv1'):
            w_conv1 = weight_variable(
                [self.conv_size, self.conv_size, 1, self.channel_1_num])
            b_conv1 = bias_variable([self.channel_1_num])
            h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

        # Pooling layer - downsamples by 2X.
        with tf.name_scope('pool1'):
            h_pool1 = max_pool(h_conv1, self.pool_size)

        # Second convolutional layer -- maps 32 feature maps to 64.
        with tf.name_scope('conv2'):
            w_conv2 = weight_variable([self.conv_size, self.conv_size,
                                       self.channel_1_num, self.channel_2_num])
            b_conv2 = bias_variable([self.channel_2_num])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

        # Second pooling layer.
        with tf.name_scope('pool2'):
            h_pool2 = max_pool(h_conv2, self.pool_size)

        # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
        # is down to 7x7x64 feature maps -- maps this to 1024 features.
        last_dim = int(input_dim / (self.pool_size * self.pool_size))
        with tf.name_scope('fc1'):
            w_fc1 = weight_variable(
                [last_dim * last_dim * self.channel_2_num, self.hidden_size])
            b_fc1 = bias_variable([self.hidden_size])

        h_pool2_flat = tf.reshape(
            h_pool2, [-1, last_dim * last_dim * self.channel_2_num])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

        # Dropout - controls the complexity of the model, prevents co-adaptation of features.
        with tf.name_scope('dropout'):
            h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Map the 1024 features to 10 classes, one for each digit
        with tf.name_scope('fc2'):
            w_fc2 = weight_variable([self.hidden_size, self.y_dim])
            b_fc2 = bias_variable([self.y_dim])
            y_conv = tf.matmul(h_fc1_drop, w_fc2) + b_fc2

        with tf.name_scope('loss'):
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.labels, logits=y_conv))
        with tf.name_scope('adam_optimizer'):
            self.train_step = tf.train.AdamOptimizer(
                self.learning_rate).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(y_conv, 1), tf.argmax(self.labels, 1))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))


def conv2d(x_input, w_matrix):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x_input, w_matrix, strides=[1, 1, 1, 1], padding='SAME')


def max_pool(x_input, pool_size):
    """max_pool downsamples a feature map by 2X."""
    return tf.nn.max_pool(x_input, ksize=[1, pool_size, pool_size, 1],
                          strides=[1, pool_size, pool_size, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """bias_variable generates a bias variable of a given shape."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def download_mnist_retry(data_dir, max_num_retries=20):
    """Try to download mnist dataset and avoid errors"""
    for _ in range(max_num_retries):
        try:
            return read_data_sets(data_dir, one_hot=True)
        except tf.errors.AlreadyExistsError:
            time.sleep(1)
    raise Exception("Failed to download MNIST.")


def main(params):
    '''
    Main function, build mnist network, run and send result to NNI.
    '''
    # Import data
    mnist = download_mnist_retry(params['data_dir'])
    print('Mnist download data done.')
    logger.debug('Mnist download data done.')

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'],
                                 channel_2_num=params['channel_2_num'],
                                 conv_size=params['conv_size'],
                                 hidden_size=params['hidden_size'],
                                 pool_size=params['pool_size'],
                                 learning_rate=params['learning_rate'])
    mnist_network.build_network()
    logger.debug('Mnist build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    logger.debug('Saving graph to: %s', graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    test_acc = 0.0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
            batch = mnist.train.next_batch(params['batch_size'])
            mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                    mnist_network.labels: batch[1],
                                                    mnist_network.keep_prob: 1 - params['dropout_rate']}
                                         )

            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})

                nni.report_intermediate_result(test_acc)
                logger.debug('test accuracy %g', test_acc)
                logger.debug('Pipe send intermediate result done.')

        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)
        logger.debug('Send final result done.')


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='./sample_data', help="data directory")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--channel_1_num", type=int, default=32)
    parser.add_argument("--channel_2_num", type=int, default=64)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_num", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
