import random

import tensorflow as tf
import numpy as np

# Height of our input data.
HEIGHT = 10

# The size of each mini-batch.
BATCH_SIZE = 2

# Create 10x3 random array in numpy; this lives in main memory (*not* the GPU).
rand_data = np.random.randn(HEIGHT, 3)

print('numpy data (in main memory)')
print('---------------------------')
print(rand_data)
print()

# Number of epochs to train for.
EPOCHS = 100

# Shuffle the indexes of mini-batches, so that this epoch we process the
# mini-batches in a random order. This helps break locality in the structure
# of the training dataset, which can help with overfitting.
INDEXES = list(range(HEIGHT // BATCH_SIZE))
random.shuffle(INDEXES)

# Copy the numpy data into TF memory as a constant var; this will be copied
# exactly one time into the GPU (if one is available).
c_rand = tf.constant(rand_data, dtype=tf.float32)

# The index to use when generating our mini-batch.
ix = tf.placeholder(shape=(), dtype=tf.int32)

# The mini-batch of data we'll work on.
batch = tf.slice(c_rand, [ix, 0], [BATCH_SIZE, -1])

# The output of the calculation.
outp = tf.reduce_sum(tf.square(batch))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        for i in indexes:
            # Run the computation. The only data in the feed_dict is a single
            # 32-bit integer we supply here. All of the data needed for the
            # mini-batch already lives in GPU memory, and doesn't need to be
            # copied from main memory.
            b, o = sess.run([batch, outp], feed_dict={ix: i})
            print('epoch = {}, ix = {}'.format(epoch, i))
            print('batch: {}'.format(b))
            print('output: {}'.format(o))
