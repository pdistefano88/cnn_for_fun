
# coding: utf-8

# In[ ]:

import glob
import tensorflow as tf
import numpy as np
import os
from sklearn.utils import shuffle
from itertools import groupby
from collections import defaultdict

def parse_dict(dataset, shuff = True):
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
    # convert the label to one-hot encoding
    num_classes = len(set(label))
    one_hot = tf.one_hot(label, num_classes)

    # read the img from file
    img_file = tf.read_file(img)
    img_decoded = tf.image.decode_jpeg(img_file)
    grayscale_image = tf.image.rgb_to_grayscale(img_decoded)
    resized_image = tf.image.resize_images(grayscale_image, size = [250, 151])

    return resized_image, one_hot

def create_dicts(image_filenames, graph = tf.get_default_graph(), shuff = True):
    training_dataset = defaultdict(list)
    testing_dataset = defaultdict(list)
    valid_dataset = defaultdict(list)

    # Split up the filename into its breed and corresponding filename.
    # The breed is found by taking the directory name
    image_filename_with_breed = map(lambda filename: (filename.split("/")[2], filename), image_filenames)

    # Group each image by the breed which is the 0th element in the tuple returned above
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
        #training_data_shuffled, training_labels_shuffled = shuffle_dataset(training_dataset, shuff)
        #valid_data_shuffled, valid_labels_shuffled = shuffle_dataset(valid_dataset, shuff)
        return training_dataset, testing_dataset, valid_dataset
        


class withableWriter:
    def __init__(self, directory, graph = tf.get_default_graph()):
        self.directory = directory
        self.graph = graph
        
    def __enter__(self):
        self.writer = tf.summary.FileWriter(self.directory, self.graph)
        self.writer.flush()

        return(self.writer)
    
    def __exit__(self, type, value, traceback):
        self.writer.close()

class cnn:
        def __init__(train_dict, test_dict, valid_dict, graph = tf.get_default_graph()):
            self.graph = graph
            self.dataset = dataset
            
# In[2]:

image_filenames = glob.glob("./imagenet-dogs/n02*/*.jpg")



# In[3]:

NUM_CLASSES = 120



# In[ ]:


init = []
graph = tf.Graph()
batch_size = 96
valid_batch_size = batch_size

with graph.as_default():
    with tf.name_scope('input_pipeline'):
        #image_batch, label_batch = tf.train.shuffle_batch(
        #    [image, label], batch_size=batch_size, capacity=capacity, min_after_dequeue=min_after_dequeue)

        tr_data = tf.contrib.data.Dataset.from_tensor_slices((training_data_shuffled, training_labels_shuffled))
        tr_data = tr_data.map(input_parser)
        tr_data = tr_data.repeat(100000)
        tr_data = tr_data.batch(batch_size)
        tr_iterator = tr_data.make_initializable_iterator()
        image_batch, label_batch = iterator.get_next()

        valid_data = tf.contrib.data.Dataset.from_tensor_slices((valid_data_shuffled, valid_labels_shuffled))
        valid_data = valid_data.map(input_parser)
        valid_data = valid_data.repeat(100000)
        valid_data = valid_data.batch(valid_batch_size)
        valid_iterator = valid_data.make_initializable_iterator()
        #image_batch, label_batch = iterator.get_next()

    with tf.name_scope('Layer_1'):
        float_image_batch = tf.image.convert_image_dtype(image_batch, tf.float32)

        conv2d_layer_one = tf.contrib.layers.convolution2d(
            float_image_batch,
            num_outputs=32,     # The number of filters to generate
            kernel_size=(5,5),          # It's only the filter height and width.
            activation_fn=tf.nn.relu,
            #weight_init=tf.random_normal,
            stride=(2, 2),
            trainable=True)
        pool_layer_one = tf.nn.max_pool(conv2d_layer_one,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')

    with tf.name_scope('Layer_2'):
        conv2d_layer_two = tf.contrib.layers.convolution2d(
        pool_layer_one,
        num_outputs=64,        # More output channels means an increase in the number of filters
        kernel_size=(5,5),
        activation_fn=tf.nn.relu,
        #weight_init=tf.random_normal,
        stride=(1, 1),
        trainable=True)

        pool_layer_two = tf.nn.max_pool(conv2d_layer_two,
            ksize=[1, 2, 2, 1],
            strides=[1, 2, 2, 1],
            padding='SAME')
        pl2_shape = list(pool_layer_two.get_shape())
        pl2_shape[0] = None
        print(pl2_shape)
        #pool_layer_two.set_shape(pl2_shape)
    with tf.name_scope('output_Layer'):
        flattened_layer_two = tf.reshape(
        pool_layer_two,
        [
            -1,  # Each image in the image_batch
            int(pl2_shape[1] * pl2_shape[2] * pl2_shape[3])         # Every other dimension of the input
        ])
        hidden_layer_three = tf.contrib.layers.fully_connected(
        flattened_layer_two,
        512,
        activation_fn=tf.nn.relu
        )

        # Dropout some of the neurons, reducing their importance in the model
        hidden_layer_three = tf.nn.dropout(hidden_layer_three, 0.1)

        # The output of this are all the connections between the previous layers and the 120 different dog breeds
        # available to train on.
        final_fully_connected = tf.contrib.layers.fully_connected(
            hidden_layer_three,
            120 # Number of dog breeds in the ImageNet Dogs dataset
            #,weight_init=lambda i,dtype = tf.truncated_normal([512, 120], stddev=0.1)  
        )
        # Find every directory name in the imagenet-dogs directory (n02085620-Chihuahua, ...)
    with tf.name_scope('train'):
        loss_batch = tf.nn.softmax_cross_entropy_with_logits(logits=final_fully_connected,
                                                       labels = label_batch, name=None)
        total_loss_batch = tf.reduce_mean(loss_batch)
        lr_0 = 0.0005
        lr_inf = 0.000005
        learning_rate = tf.Variable(lr_0)
        alpha = 0.9
        train_op = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = alpha).minimize(total_loss_batch)

    with tf.name_scope('summary'):
        #logits_histogram = tf.summary.histogram(values = final_fully_connected, name = 'final_layer')
        #softmax_histogram = tf.summary.histogram(values= tf.nn.softmax(final_fully_connected), name = 'output')
        global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")
        increment_step = global_step.assign_add(1)
        total_loss = tf.Variable(0.)
        alpha = 0.9
        increment_loss = tf.assign(total_loss, tf.add(tf.multiply(alpha, total_loss),
                                                      tf.multiply(1-alpha, total_loss_batch)))
        total_loss_summary = tf.summary.scalar(name = 'loss', tensor = total_loss)
        total_loss_batch_summary = tf.summary.scalar(name = 'loss_batch', tensor = total_loss_batch)
        
        
        #valid_loss = tf.Variable(0.)
        #increment_valid_loss = tf.assign(valid_loss, total_loss_batch)
        #valid_loss_summary = tf.summary.scalar(name = 'valid_loss_batch', tensor = valid_loss)
        valid_accuracy = tf.Variable(0.)
        valid_accuracy_update = tf.assign(valid_accuracy, tf.contrib.metrics.accuracy(tf.argmax(label_batch, axis = 1),     tf.argmax(final_fully_connected, axis = 1)))
        training_accuracy = tf.Variable(0.)
        training_accuracy_update = tf.assign(training_accuracy, tf.contrib.metrics.accuracy(tf.argmax(label_batch, axis = 1), tf.argmax(final_fully_connected, axis = 1)))

        valid_accuracy_summary = tf.summary.scalar(name = 'valid_accuracy', tensor = valid_accuracy)
        training_accuracy_summary = tf.summary.scalar(name = 'training_accuracy', tensor = training_accuracy)
        learning_rate_summary = tf.summary.scalar(name = 'lr', tensor = learning_rate)
        merged_summaries =tf.summary.merge_all()


    with tf.name_scope('global_init'):
        init.append (tf.global_variables_initializer())

    training_steps = 300000

    run_names = list(map(lambda st : st.split("/")[3], glob.glob("./tensorboard/cnn/*")))
    j = int(run_names[len(run_names) - 1]) + 1
    print("run n. %i" % (j))
    
    with tf.Session() as sess, withableWriter('tensorboard/cnn/'+str(j), graph)  as writer:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        sess.run([tf.global_variables_initializer(), iterator.initializer, valid_iterator.initializer])
        #sess.run()
        out = [0 for i in range(training_steps)]
        for step in range(training_steps):
            if step % 100 == 0:
                image_batch, label_batch = valid_iterator.get_next()
                sess.run([valid_accuracy_update])
                image_batch, label_batch = iterator.get_next()
                sess.run([training_accuracy_update])
                #acc = sess.run([tf.contrib.metrics.accuracy(tf.argmax(label_batch, axis = 1),
                #tf.argmax(final_fully_connected, axis = 1))])
                #sess.run([increment_valid_loss, valid_accuracy_summary])
                #writer.add_summary(summary, global_step=step)
                #print(acc)

            _, step_, __ ,summary = sess.run([train_op,
                                                         increment_step,
                                                         increment_loss,
                                                         merged_summaries])
            writer.add_summary(summary, global_step=step_)
            # for debugging and learning purposes, see how the loss gets decremented thru training steps
            #if step % 10 == 0:
            writer.flush()
            
            if step < 200000:
                sess.run(learning_rate.assign(lr_0 + step/200000 * (lr_inf - lr_0)))
        print('qua')
        coord.request_stop()
        coord.join(threads)



# In[ ]:



