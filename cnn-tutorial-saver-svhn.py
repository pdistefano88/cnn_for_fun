import glob
import tensorflow as tf
import numpy as np
import os
import argparse
import pickle
import scipy.io as sio
from sklearn.utils import shuffle
from itertools import groupby
from collections import defaultdict

num_classes = 10

def parse_dict(dataset, shuff = True):
    ###takes a label:data dictionary, shuffles it and returns two lists for data and labels
    data = []
    labels = []
    i = 0
    for k, vl in dataset.items():
        for v in vl:
            labels.append(i)
            data.append(v)
        i += 1

    data_shuffled = []
    labels_shuffled = []
    if shuff:
        indices = shuffle(np.arange(len(data)))
    else:
        indices = np.arange(len(data))
    for ind in indices:
        data_shuffled.append(data[ind])
        labels_shuffled.append(labels[ind])
    return data_shuffled, labels_shuffled

def input_parser(img, label):
    ##Input transformation routine for tensorflow dataset API
    ##Takes a jpg filename path, decodes the image, converts to greyscale and resizes it
    ##Encodes the labels into one-hot
    one_hot = tf.one_hot(label, num_classes)
    return tf.image.per_image_standardization(img), one_hot

def create_dicts(image_filenames, shuff = True):
    ##Creates a dictionary to be parsed by parse_dict
    ##Takes img filenames and uses the directory tree of the Stanford dogs dataset to get labels and create a label:data dictionary
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    valid_dataset = defaultdict(list)
    image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename), image_filenames)
    for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x: x[0]):
        # Enumerate each breed's image and send ~15% of the images to a testing set and ~15% of the images to a validation set
        for i, breed_image in enumerate(breed_images):
            if i % 20 in [0, 1, 2]:
                testing_dataset[dog_breed].append(breed_image[1])
            elif i % 20 in [3, 4, 5]:
                valid_dataset[dog_breed].append(breed_image[1])
            else:
                training_dataset[dog_breed].append(breed_image[1])

        # Check that each breed includes at least 13% of the images for testing
        breed_training_count = len(training_dataset[dog_breed])
        breed_testing_count = len(testing_dataset[dog_breed])
        breed_valid_count = len(valid_dataset[dog_breed])

        assert round(breed_testing_count / (breed_training_count + breed_testing_count + breed_valid_count), 2) > 0.13,    "Not enough testing images."
        assert round(breed_valid_count / (breed_training_count + breed_testing_count + breed_valid_count), 2) > 0.13,    "Not enough validation images."

    return training_dataset, testing_dataset, valid_dataset

def create_dataset():
    image_filenames = glob.glob("./Images/n02*/*.jpg")
    train_dataset, test_dataset, valid_dataset = create_dicts(image_filenames)    
    train_data, train_labels = parse_dict(train_dataset)
    valid_data, valid_labels = parse_dict(valid_dataset)
    test_data, test_labels = parse_dict(test_dataset)
    with open('train_data.pkl', 'wb') as df, open('train_labels.pkl', 'wb') as lf:
        pickle.dump(train_data,file = df)
        pickle.dump(train_labels,file = lf)
        
    with open('test_data.pkl', 'wb') as df, open('test_labels.pkl', 'wb') as lf:
        pickle.dump(test_data,file = df)
        pickle.dump(test_labels,file = lf)
        
    with open('valid_data.pkl', 'wb') as df, open('valid_labels.pkl', 'wb') as lf:
        pickle.dump(valid_data,file = df)
        pickle.dump(valid_labels,file = lf)
        
    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels

def get_dataset_from_backup():
    with open('train_data.pkl', 'rb') as df, open('train_labels.pkl', 'rb') as lf:
        train_data = pickle.load(df)
        train_labels = pickle.load(lf)
        
    with open('test_data.pkl', 'rb') as df, open('test_labels.pkl', 'rb') as lf:
        test_data = pickle.load(df)
        test_labels = pickle.load(lf)
        
    with open('valid_data.pkl', 'rb') as df, open('valid_labels.pkl', 'rb') as lf:
        valid_data = pickle.load(df)
        valid_labels = pickle.load(lf)
        
    return train_data, train_labels, test_data, test_labels, valid_data, valid_labels
    
        
class withableWriter:
    ##Opens a tf.summary.FileWriter at the beginning of a with clause and closes it the end 
    def __init__(self, directory, graph = tf.get_default_graph()):
        self.directory = directory
        self.graph = graph
        
    def __enter__(self):
        self.writer = tf.summary.FileWriter(self.directory)
        self.writer.flush()

        return(self.writer)
    
    def __exit__(self, type, value, traceback):
        self.writer.close()

class cnn:
    ##Creates a cnn model with two convolutional layers with max pooling and a dense level
    def __init__(self, train_data, train_labels,
                 test_data, test_labels,
                 valid_data,valid_labels,
                 batch_size, graph = tf.Graph()):
        ##Initializes the graph
        self.num_classes = len(set(train_labels + valid_labels + test_labels))
        self.graph = graph
        with self.graph.as_default():
            self.batch_size = batch_size
            train_dataset, self.train_iterator = self._getDataset(train_data, train_labels, "train_dataset", self.batch_size)
            valid_dataset, self.valid_iterator = self._getDataset(valid_data, valid_labels, "valid_dataset", len(valid_labels))
            valid_dataset, self.test = self._getDataset(test_data, test_labels, "test_dataset", len(test_labels))
            self.handle = tf.placeholder(tf.string, shape=[])
            self.iterator = tf.contrib.data.Iterator.from_string_handle(
                self.handle, train_dataset.output_types, train_dataset.output_shapes)

            
            self.image_batch, self.label_batch = self.iterator.get_next()
            self.kp_in = tf.placeholder(tf.float32)
            self.image_batch = tf.nn.dropout(self.image_batch, self.kp_in)
            
            with tf.name_scope('Layer_1'):
                self.kp_cl1 = tf.placeholder(tf.float32)
                self.out_l1 = tf.nn.dropout( self._conv_pool(image_batch=self.image_batch, num_channels = 96, ksize_conv = [5,5],
                                              stride_conv=[1,1], ksize_pool=[3,3], stride_pool=[2,2]),
                                             self.kp_cl1)
                print(tf.shape(self.out_l1))
        
            with tf.name_scope('Layer_2'):
                self.kp_cl2 = tf.placeholder(tf.float32)                
                self.out_l2 = tf.nn.dropout( self._conv_pool(self.out_l1, num_channels = 128, ksize_conv = [5,5],
                                                          stride_conv=[1,1], ksize_pool=[3,3], stride_pool=[2,2]),
                                             self.kp_cl2)
            with tf.name_scope('Layer_3'):
                self.kp_cl3 = tf.placeholder(tf.float32)                
                self.out_l3 = tf.nn.dropout( self._conv_pool(self.out_l2, num_channels = 256, ksize_conv = [5,5],
                                                          stride_conv=[1,1], ksize_pool=[3,3], stride_pool=[2,2]),
                                             self.kp_cl3)
                pl3_shape = list(self.out_l3.get_shape())
                self.flattened_layer_two = tf.reshape(
                self.out_l3,
                [
                    -1,  # Each image in the image_batch
                    int(pl3_shape[1] * pl3_shape[2] * pl3_shape[3])         # Every other dimension of the input
                ])

            with tf.name_scope('fully_connected_1'):

                self.kp_fcl1 = tf.placeholder(tf.float32)                    
                self.fully_connected_layer_1 = tf.nn.dropout( tf.contrib.layers.fully_connected(
                self.flattened_layer_two,
                2048,
                activation_fn=tf.nn.relu),
                                                             self.kp_fcl1)
                
            with tf.name_scope('fully_connected_2'):
                
                self.kp_fcl2 = tf.placeholder(tf.float32)                    
                self.fully_connected_layer_2 = tf.nn.dropout( tf.contrib.layers.fully_connected(
                                self.fully_connected_layer_1,
                            2048,
                            activation_fn=tf.nn.relu),
                                                                          self.kp_fcl2)
            with tf.name_scope('output_layer'):

                self.final_fully_connected = tf.contrib.layers.fully_connected(
                    self.fully_connected_layer_2,
                    num_classes
                )

            with tf.name_scope('loss'):
                self.loss_batch = tf.nn.softmax_cross_entropy_with_logits(logits=self.final_fully_connected,
                                                               labels = self.label_batch, name=None)
                self.valid_loss_batch = tf.Variable(0.)
                self.valid_loss_update = tf.assign(self.valid_loss_batch,
                                                   tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                                                       logits=self.final_fully_connected,labels = self.label_batch, name=None))
                                                   )
                tf.nn.softmax_cross_entropy_with_logits(logits=self.final_fully_connected,
                                                                                      labels = self.label_batch, name=None)
                
                self.total_loss_batch = tf.reduce_mean(self.loss_batch)
            with tf.name_scope('train'):
                self.learning_rate = tf.Variable(0., name='learning_rate')
        
            with tf.name_scope('summary'):
                self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
                self.increment_step = self.global_step.assign_add(1)
                total_loss = tf.Variable(0.)
                alpha = 0.9
                self.increment_loss = tf.assign(total_loss, tf.add(tf.multiply(alpha, total_loss),
                                                              tf.multiply(1-alpha, self.total_loss_batch)))
                total_loss_summary = tf.summary.scalar(name = 'loss', tensor = total_loss)
                total_loss_batch_summary = tf.summary.scalar(name = 'loss_batch', tensor = self.total_loss_batch)
                valid_accuracy = tf.Variable(0.)
                self.valid_accuracy_update = tf.assign(valid_accuracy, tf.contrib.metrics.accuracy(tf.argmax(self.label_batch, axis = 1),     tf.argmax(self.final_fully_connected, axis = 1)))
                training_accuracy = tf.Variable(0.)
                self.training_accuracy_update = tf.assign(training_accuracy, tf.contrib.metrics.accuracy(tf.argmax(self.label_batch, axis = 1), tf.argmax(self.final_fully_connected, axis = 1)))
        
                valid_accuracy_summary = tf.summary.scalar(name = 'valid_loss', tensor = self.valid_loss_batch)
                valid_loss_summary = tf.summary.scalar(name = 'valid_accuracy', tensor = valid_accuracy)

                training_accuracy_summary = tf.summary.scalar(name = 'training_accuracy', tensor = training_accuracy)
                learning_rate_summary = tf.summary.scalar(name = 'lr', tensor = self.learning_rate)
                self.merged_summaries = tf.summary.merge_all()

    def _getDataset(self, data, labels, name_scope, batch_size, rep = 1000000000):
        with tf.name_scope(name_scope):
            num_classes = self.num_classes            
            ds = tf.contrib.data.Dataset.from_tensor_slices((data, labels))
            ds = ds.shuffle(buffer_size=len(labels))
            ds = ds.map(input_parser)
            ds = ds.repeat(rep)
            ds = ds.batch(batch_size)
            iterator = ds.make_initializable_iterator()
        return ds, iterator

    def _conv_pool(self,image_batch, num_channels, ksize_conv, stride_conv, ksize_pool, stride_pool):
        float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

        conv2d_layer_one = tf.contrib.layers.convolution2d(
            float_image_batch,
            num_outputs= num_channels,
            kernel_size=ksize_conv,
            activation_fn = tf.nn.relu,
            stride = stride_conv,
            trainable = True)
        
        pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
            ksize=[1, ksize_pool[0], ksize_pool[1], 1],
            strides=[1, stride_pool[0], stride_pool[1], 1],
            padding='SAME')
        
        return pool_layer_one

    def training(self, lr_0, lr_inf, momentum = True, alpha = 0.9):
        self.lr_0 = lr_0
        self.lr_inf = lr_inf
        with self.graph.as_default():
            with tf.name_scope('train'):
                self.train_op = tf.train.MomentumOptimizer(learning_rate = self.learning_rate, momentum = alpha).minimize(self.total_loss_batch)
    
    def train(self, training_steps, saver = True, saver_step = 1000, saver_path = "saver", last_upgrade = -1):
        run_names = sorted(list(map(lambda st : int(st.split("/")[3]), glob.glob("./tensorboard/cnn/*"))))

        if len(run_names):
            j = run_names[len(run_names) - 1] + 1
        else:
            j = 0
        print("run n. %i" % (j))

        if last_upgrade < 0:
            last_upgrade = training_steps
        with self.graph.as_default(), tf.Session() as sess, withableWriter('tensorboard/cnn/'+str(j), self.graph)  as writer:
            saver = tf.train.Saver(max_to_keep = 10)
            sess.run([tf.global_variables_initializer(), self.train_iterator.initializer, self.valid_iterator.initializer])
            self.training_handle = sess.run(self.train_iterator.string_handle())
            self.validation_handle = sess.run(self.valid_iterator.string_handle())

            for step in range(training_steps):
                if step < last_upgrade:
                    sess.run(self.learning_rate.assign(self.lr_0 + step/last_upgrade * (self.lr_inf - self.lr_0)))
                if step % 1000 == 0:
                    sess.run([self.valid_accuracy_update, self.valid_loss_update], feed_dict={self.handle : self.validation_handle,
                                                                                              self.kp_in : 1,
                                                                                              self.kp_cl1 : 1,
                                                                                                self.kp_cl2 : 1,
                                                                                                self.kp_cl3 : 1,
                                                                                                self.kp_fcl1 : 1,
                                                                                                self.kp_fcl2 : 1})
                    #self.image_batch, self.label_batch = self.train_iterator.get_next()
                    sess.run([self.training_accuracy_update], feed_dict={self.handle : self.training_handle,
                                                                                              self.kp_in : 1,
                                                                                              self.kp_cl1 : 1,
                                                                                                self.kp_cl2 : 1,
                                                                                                self.kp_cl3 : 1,
                                                                                                self.kp_fcl1 : 1,
                                                                                                self.kp_fcl2 : 1})
                _, step_, __ , summary = sess.run([self.train_op,
                                                             self.increment_step,
                                                             self.increment_loss,
                                                             self.merged_summaries], feed_dict={self.handle : self.training_handle,
                                                                                              self.kp_in : 0.9,
                                                                                              self.kp_cl1 : 0.75,
                                                                                                self.kp_cl2 : 0.75,
                                                                                                self.kp_cl3 : 0.5,
                                                                                                self.kp_fcl1 : 0.5,
                                                                                                self.kp_fcl2 : 0.5})
                writer.add_summary(summary, global_step=step_)
                writer.flush()
                if step % 1000 == 0:
                    # Append the step number to the checkpoint name:
                    full_path = saver_path + "/" + str(j)
                    if not os.path.exists(full_path):
                        os.makedirs(full_path)
                    
                    saver.save(sess, full_path + "/my-model", global_step=step_)
            print('Training is done.')

    def predict(self, fn_batch):
            
        with self.graph.as_default(), tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, 'my-model-20001')            
            batch_list = []
            if not isinstance(fn_batch, list):
                fn_batch = [fn_batch]
            for i in range(len(fn_batch)):
                img, _ = input_parser(fn_batch[i], 0)
                batch_list.append(img)

            batch = sess.run(tf.stack(batch_list))                        
            rv = sess.run(tf.argmax(self.final_fully_connected, axis = 1), feed_dict={self.kp_cl1 : 1,
                                                                                                self.kp_cl2 : 1,
                                                                                                self.kp_cl3 : 1,
                                                                                                self.kp_fcl1 : 1,
                                                                                                self.kp_fcl2 : 1,
                                                                                                self.image_batch : batch})
        return rv

    def restore(self, path):
        with self.graph.as_default(), tf.Session() as sess:
            print('restoring...')
            saver = tf.train.Saver()
            saver.restore(sess, path)

parser = argparse.ArgumentParser()
parser.add_argument("--initial_learning_rate", type = float ,  help="The initial learning rate. The learning rate is then linearly reduced by a factor of 100. Default is 0.001.")
parser.add_argument("--training_steps", type = int ,  help="Number of training steps. Defalut is 100000.")
parser.add_argument("--batch_size", type = int ,  help="Batch size. Defalut is 96.")
parser.add_argument("--load_dataset_splits_from_pickles", type = int ,  help="Load train, validation and test set files from disk (yes, default) or create from scratch.")
args = parser.parse_args()
if args.initial_learning_rate:
    lr_0 = args.initial_learning_rate
else:
    lr_0 = 0.001
if args.training_steps:
    training_steps = args.training_steps
else:
    training_steps = 1000000
if args.batch_size:
    batch_size = args.batch_size
else:
    batch_size = 96
    
if args.load_dataset_splits_from_pickles:
    load_dataset_splits_from_pickles = args.load_dataset_splits_from_pickles
else:
    load_dataset_splits_from_pickles = 'yes'

if load_dataset_splits_from_pickles == 'yes':
    train_data, train_labels, test_data, test_labels, valid_data, valid_labels = get_dataset_from_backup()
else:
    train_data, train_labels, test_data, test_labels, valid_data, valid_labels = create_dataset()
 

train_mat = sio.loadmat('svhn/train_32x32.mat')
test_mat = sio.loadmat('svhn/test_32x32.mat')
extra_mat = sio.loadmat('svhn/extra_32x32.mat')

train_mat['X'] = np.rollaxis(train_mat['X'], 3)
test_mat['X'] = np.rollaxis(test_mat['X'],3)
extra_mat['X'] = np.rollaxis(extra_mat['X'], 3)
train_mat['y'] = list(np.reshape(train_mat['y'], -1))
test_mat['y'] = list(np.reshape(test_mat['y'], -1))
extra_mat['y'] = list(np.reshape(extra_mat['y'], -1))

train_data = np.concatenate((train_mat['X'][:(len(train_mat['X']) - 4000), :, :, :], extra_mat['X'][:(len(extra_mat['X'])-2000), :, :, :]))
valid_data = np.concatenate((train_mat['X'][(len(train_mat['X']) - 4000):len(train_mat['X']), :, :, :],  extra_mat['X'][(len(extra_mat['X'])-2000):len(extra_mat['X'])]))
test_data = test_mat['X']

train_labels = train_mat['y'][:(len(train_mat['y']) - 4000)] + extra_mat['y'][:(len(extra_mat['y'])-2000)]
valid_labels = train_mat['y'][(len(train_mat['y']) - 4000):len(train_mat['y'])] + extra_mat['y'][(len(extra_mat['y'])-2000):len(extra_mat['y'])]
test_labels = test_mat['y']

my_cnn = cnn(train_data, train_labels, valid_data, valid_labels, test_data, test_labels, batch_size)
my_cnn.training(lr_0, 0.01 * lr_0)
my_cnn.train(training_steps)