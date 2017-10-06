__author__ = 'r_tyler_mclaughlin'
import os
import time

# Load MNIST dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# Import Tensorflow and start a session
import tensorflow as tf
sess = tf.InteractiveSession()

def weight_variable(shape):
  '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):

  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
    '''
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    '''

    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

# this will compute all the required statistics for a single variable
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

def lrelu(x, alpha = -0.01):
   # easy implementation of the leaky ReLU function
   # parameter alpha SHOULD BE NEGATIVE
   return tf.maximum(x, alpha * x)

def main():
    # Specify training parameters
    result_dir = './results_part_2c-LReLU/' # directory where the results from the training are saved
    if tf.gfile.Exists(result_dir):
        tf.gfile.DeleteRecursively(result_dir)

    tf.gfile.MakeDirs(result_dir)

    max_step = 5500 #550 #5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

    start_time = time.time()
    x = tf.placeholder(tf.float32, [None, 784])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])


    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])

    x_image = tf.reshape(x, [-1, 28, 28, 1])

    input_1 = conv2d(x_image, W_conv1) + b_conv1

    h_conv1 = lrelu(input_1)
    h_pool1 = max_pool_2x2(h_conv1)

    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])

    input_2 = conv2d(h_pool1, W_conv2) + b_conv2
    h_conv2 = lrelu(input_2)
    h_pool2 = max_pool_2x2(h_conv2)

    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])

    input_fc1 = tf.matmul(h_pool2_flat, W_fc1) + b_fc1

    h_fc1 = lrelu(input_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    # SET UP TRAINING
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    val_accuracy_ = tf.placeholder(tf.float32,shape=())
    test_accuracy_ = tf.placeholder(tf.float32,shape=())

    with tf.Session() as sess:
      # Add a scalar summary for the snapshot loss.
      with tf.name_scope('cross_entropy'):
          variable_summaries(cross_entropy)


      # Create a saver for writing training checkpoints.
      saver = tf.train.Saver()

      # Instantiate a SummaryWriter to output summaries and the Graph.
      summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

      # Instantiate another SummaryWriter to output summaries and Graph for every epoch
      summary_test_writer = tf.summary.FileWriter(result_dir + '/test', sess.graph)
      # Make sure you change the result_dir (by appending '/val')!
      summary_val_writer = tf.summary.FileWriter(result_dir + '/val', sess.graph)
      sess.run(tf.global_variables_initializer())

      # FIRST CONVOLUTIONAL LAYER
      with tf.name_scope('W_conv1'):
          variable_summaries(W_conv1)
      with tf.name_scope('b_conv1'):
          variable_summaries(b_conv1)
      with tf.name_scope('input_1'):
          variable_summaries(input_1)
      with tf.name_scope('h_conv1'):
          variable_summaries(h_conv1)
      with tf.name_scope('h_pool1'):
          variable_summaries(h_pool1)

      # SECOND CONVOLUTIONAL LAYER

      with tf.name_scope('W_conv2'):
          variable_summaries(W_conv2)
      with tf.name_scope('b_conv2'):
          variable_summaries(b_conv2)
      with tf.name_scope('input_2'):
          variable_summaries(input_2)
      with tf.name_scope('h_conv2'):
          variable_summaries(h_conv2)
      with tf.name_scope('h_pool2'):
          variable_summaries(h_pool2)

      # FULLY CONNECTED LAYER

      with tf.name_scope('W_fc1'):
          variable_summaries(W_fc1)
      with tf.name_scope('b_fc1'):
          variable_summaries(b_fc1)
      with tf.name_scope('h_pool2_flat'):
          variable_summaries(h_pool2_flat)
      with tf.name_scope('input_fc1'):
          variable_summaries(input_fc1)
      with tf.name_scope('h_fc1'):
          variable_summaries(h_fc1)

      # READOUT LAYER
      with tf.name_scope('W_fc2'):
          variable_summaries(W_fc2)
      with tf.name_scope('b_fc2'):
          variable_summaries(b_fc2)
      with tf.name_scope('y_conv'):
          variable_summaries(y_conv)

      # Build the summary operation based on the TF collection of Summaries.
      summary_op = tf.summary.merge_all()

      # crucial:  add these additional "epoch" summaries after the merge_all step
      summary_op_test = tf.summary.scalar('test_accuracy',test_accuracy_)
      summary_op_val = tf.summary.scalar('val_accuracy', val_accuracy_)

      for i in range(max_step):
        batch = mnist.train.next_batch(50)
        if i % 100 == 0:
          train_accuracy = accuracy.eval(feed_dict={
              x: batch[0], y_: batch[1], keep_prob: 1.0})
          print('step %d, training accuracy %g' % (i, train_accuracy))
          # Update the events file which is used to monitor the training
          summary_str = sess.run(summary_op, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
          summary_writer.add_summary(summary_str, i)
          summary_writer.flush()

        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

        # save the checkpoints every 1100 iterations
        if i % 1100 == 0 or i == max_step:
            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=i)

            test_feed_dict = { x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            test_accuracy = accuracy.eval(feed_dict=test_feed_dict)
            print('step %d, test accuracy %g' % (i, test_accuracy))
            summary_str_test = sess.run(summary_op_test, feed_dict={test_accuracy_ : test_accuracy})
            summary_test_writer.add_summary(summary_str_test, i)
            summary_test_writer.flush()

            val_feed_dict = { x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}
            val_accuracy = accuracy.eval(feed_dict=val_feed_dict)
            print('step %d, validation accuracy %g' % (i, val_accuracy))
            summary_str_val = sess.run(summary_op_val, feed_dict={val_accuracy_ : val_accuracy})
            summary_val_writer.add_summary(summary_str_val, i)
            summary_val_writer.flush()


      print('test accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))

      print('validation accuracy %g' % accuracy.eval(feed_dict={
          x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0}))

      stop_time = time.time()
      print('The training takes %f second to finish' % (stop_time - start_time))

if __name__ == "__main__":
    main()
