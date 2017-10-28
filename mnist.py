import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data as _input_data_

data = _input_data_.read_data_sets("MNIST_data", one_hot=True)

with tf.variable_scope('input_data_placeholder'):
    _x_ = tf.placeholder(tf.float32, [None, 784])
with tf.variable_scope('output_data_placeholder'):
    _y_ = tf.placeholder(tf.float32, [None, 10])

def make_weight(layer_name, shape):
    with tf.variable_scope(layer_name):
        with tf.name_scope("Weights"):
            W = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
            tf.summary.histogram(layer_name+"_W_hist", W)            
            return W

def make_bias(layer_name, shape):
    with tf.variable_scope(layer_name):
        _init = tf.constant(0.1, shape = shape)
        with tf.name_scope("Biases"):            
            bias = tf.Variable(_init)
            tf.summary.histogram(layer_name+"_B_hist", bias)
            return bias;

def make_conv2d(layer_name, input_data, weights, strides=[1,1,1,1], padding = "SAME"):
    with tf.variable_scope(layer_name):
        return tf.nn.conv2d(input_data, weights, strides = strides, padding = padding)
    
def make_max_pool_2x2(layer_name, input_data, kernel_size=[1,2,2,1], strides=[1,2,2,1], padding = "SAME"):
    with tf.variable_scope(layer_name):
        return tf.nn.max_pool(input_data, ksize = kernel_size, strides = strides, padding = padding)

'''=================================================================================================='''
#resize input data flatten then resize to 28x28
X = tf.reshape(_x_, [-1, 28, 28, 1])

#28*28 -> convolutional 1 -> 38
W_conv1 = make_weight('convolutional_1', [5, 5, 1, 38])
b_conv1 = make_bias('convolutional_1', [38])
conv1 = make_conv2d('convolutional_1', X, W_conv1) + b_conv1
h_conv1 = tf.nn.relu(conv1)
h_maxpool1 = make_max_pool_2x2('convolutional_1_maxpool', h_conv1)

#38 -> convolutional 2 -> 64
W_conv2 = make_weight('convolutional_2', [5,5,38,64])
b_conv2 = make_bias('convolutional_2', [64])
conv2 = make_conv2d('convolutional_2', h_maxpool1, W_conv2) + b_conv2
h_conv2 = tf.nn.relu(conv2)
h_maxpool2 = make_max_pool_2x2('convolutional_2_maxpool', h_conv2)

# fully connected
W_fc1 = make_weight('fully_connected1', [7 * 7 * 64, 1024]) # 7 * 7 * 64
b_fc1 = make_bias('fully_connected1', [1024])
h_maxpool2_flatten = tf.reshape(h_maxpool2, [-1, 7 * 7 * 64])
fc1 = tf.matmul(h_maxpool2_flatten, W_fc1) + b_fc1
h_fc1 = tf.nn.relu(fc1)

#dropout layer for regularization
keep_prob = tf.placeholder(tf.float32)
h_fc1_dropout = tf.nn.dropout(h_fc1,keep_prob)

#final layer(output)
W_fc2 = make_weight('fully_connected_out', [1024, 10])
b_fc2 = make_bias('fully_connected_out', [10])
y_out = tf.matmul(h_fc1_dropout, W_fc2) + b_fc2

#calculate cross entropy aka error
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels = _y_, logits = y_out)    
)
tf.summary.scalar("cross_entropy",cross_entropy)
#define train step
train_step = tf.train.AdamOptimizer(0.0004).minimize(cross_entropy)

predict_correct = tf.equal(tf.argmax(y_out, 1), tf.argmax(_y_, 1))
accuracy = tf.reduce_mean(tf.cast(predict_correct, tf.float32))
tf.summary.scalar("accuracy",accuracy)

#initialize all necessary variable
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/2", sess.graph)
    #init
    sess.run(init)    
    #run
    for step in range(10000):
        #prev 60
        batch = data.train.next_batch(50)
        #prev 120
        if step % 100 == 0:
            summary, train_acc = sess.run([merged, accuracy], feed_dict = {_x_: batch[0], _y_: batch[1], keep_prob: 1.0})
            writer.add_summary(summary, step)
            print('step %d -> accuracy: %g'%(step, train_acc))
        else:
            summary, _ = sess.run([merged, train_step], feed_dict = {_x_: batch[0], _y_: batch[1], keep_prob: 0.75})#prev 0.5

        if step % 51 == 0:
            writer.add_summary(summary, step)
        
    print('test accuracy %g' % accuracy.eval(feed_dict={
      _x_: data.test.images, _y_: data.test.labels, keep_prob: 1.0}))




