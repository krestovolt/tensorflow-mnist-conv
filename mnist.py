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


with tf.name_scope('Convolutional_1') as scope:

# In this section, we visualize the filters of the first convolutional layers
# We concatenate the filters into one image
# Credits for the inspiration go to Martin Gorner
    W1_a = W_conv1                       # [5, 5, 1, 38]
    W1pad= tf.zeros([5, 5, 1, 1])        # [5, 5, 1, 11]  - four zero kernels for padding
    # We have a 6 by 6 grid of kernel visualizations. yet we only have 32 filters
    # Therefore, we concatenate 4 empty filters
    W1_b = tf.concat([W1_a, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad, W1pad], axis=3)   # [5, 5, 1, 49]
    W1_c = tf.split(W1_b, 49, axis=3)         # 49 x [5, 5, 1, 1]
    W1_row0 = tf.concat(W1_c[0:7], axis=0)    # [35, 5, 1, 1]
    W1_row1 = tf.concat(W1_c[7:14], axis=0)   # [35, 5, 1, 1]
    W1_row2 = tf.concat(W1_c[14:21], axis=0)  # [35, 5, 1, 1]
    W1_row3 = tf.concat(W1_c[21:28], axis=0)  # [35, 5, 1, 1]
    W1_row4 = tf.concat(W1_c[28:35], axis=0)  # [35, 5, 1, 1]
    W1_row5 = tf.concat(W1_c[35:42], axis=0)  # [35, 5, 1, 1]
    W1_row6 = tf.concat(W1_c[42:49], axis=0)  # [35, 5, 1, 1]

    W1_d = tf.concat([W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5, W1_row6], axis=1) # [35, 35, 1, 1]
    W1_e = tf.reshape(W1_d, [1, 35, 35, 1])
    W1_f = tf.split(W1_e, 1, 3)
    W1_g = tf.concat(W1_f[0:1], 0)    
    '''####################################################################################'''    
    Wtag = tf.placeholder(tf.string, None)
    tf.summary.image("images_conv", W1_g, 1)    
    '''===================================================================================='''

with tf.name_scope('Convolutional_2_a') as scope:
    W3_a = W_conv2                                           # [5,5,38,64]
    #W3pad= tf.zeros([5,5,38,n])             # [5,5,38,n]  - n zero kernels for padding
    #W3_b = tf.concat([W3_a, W3pad], 3)  # [5,5,38,64]
    # in my case I have exactly 64 output channels so I assign W3_b to W3_a
    # comment the assignment below and uncomment the tf.zeros and tf.concat statements above to pad
    # fewer than 64 output channels to exactly 64. You have to change "n" of course!
    W3_b = W3_a
    W3_c = tf.split(W3_b, 64, axis=3)                  # 64 x [5,5,38,64]
    W3_row0 = tf.concat(W3_c[0:8], axis=0)       # [5*8, 5, 38, 1]
    W3_row1 = tf.concat(W3_c[8:16], axis=0)     # [5*8, 5, 38, 1]
    W3_row2 = tf.concat(W3_c[16:24], axis=0)    # [5*8, 5, 38, 1]
    W3_row3 = tf.concat(W3_c[24:32], axis=0)    # [5*8, 5, 38, 1]
    W3_row4 = tf.concat(W3_c[32:40], axis=0)    # [5*8, 5, 38, 1]
    W3_row5 = tf.concat(W3_c[40:48], axis=0)    # [5*8, 5, 38, 1]
    W3_row6 = tf.concat(W3_c[48:56], axis=0)    # [5*8, 5, 38, 1]
    W3_row7 = tf.concat(W3_c[56:64], axis=0)    # [5*8, 5, 38, 1]
    W3_d = tf.concat([W3_row0, W3_row1, W3_row2, W3_row3, W3_row4, W3_row5, W3_row6, W3_row7], axis=1) # [64, 64, 38, 1]
    W3_e = tf.reshape(W3_d, [1, 5*8, 5*8, 38])
    W3_f = tf.split(W3_e, 38, axis=3)                  # 38 x [1, 64, 64, 1]    
    W3_g = tf.concat(W3_f[0:19], axis=0)             # [19, 64, 64, 1]        

    '''####################################################################################'''    
    Wtag = tf.placeholder(tf.string, None)    
    tf.summary.image("images_conv", W3_g, 19)    
    '''===================================================================================='''

with tf.name_scope('Convolutional_2_b') as scope:
    W3_a = W_conv2                                           # [5,5,38,64]
    #W3pad= tf.zeros([5,5,38,n])             # [5,5,38,n]  - n zero kernels for padding
    #W3_b = tf.concat([W3_a, W3pad], 3)  # [5,5,38,64]
    # in my case I have exactly 64 output channels so I assign W3_b to W3_a
    # comment the assignment below and uncomment the tf.zeros and tf.concat statements above to pad
    # fewer than 64 output channels to exactly 64. You have to change "n" of course!
    W3_b = W3_a
    W3_c = tf.split(W3_b, 64, axis=3)                  # 64 x [5,5,38,64]
    W3_row0 = tf.concat(W3_c[0:8], axis=0)       # [5*8, 5, 38, 1]
    W3_row1 = tf.concat(W3_c[8:16], axis=0)     # [5*8, 5, 38, 1]
    W3_row2 = tf.concat(W3_c[16:24], axis=0)    # [5*8, 5, 38, 1]
    W3_row3 = tf.concat(W3_c[24:32], axis=0)    # [5*8, 5, 38, 1]
    W3_row4 = tf.concat(W3_c[32:40], axis=0)    # [5*8, 5, 38, 1]
    W3_row5 = tf.concat(W3_c[40:48], axis=0)    # [5*8, 5, 38, 1]
    W3_row6 = tf.concat(W3_c[48:56], axis=0)    # [5*8, 5, 38, 1]
    W3_row7 = tf.concat(W3_c[56:64], axis=0)    # [5*8, 5, 38, 1]
    W3_d = tf.concat([W3_row0, W3_row1, W3_row2, W3_row3, W3_row4, W3_row5, W3_row6, W3_row7], axis=1) # [64, 64, 38, 1]
    W3_e = tf.reshape(W3_d, [1, 5*8, 5*8, 38])
    W3_f = tf.split(W3_e, 38, axis=3)                  # 38 x [1, 64, 64, 1]    
    W3_g = tf.concat(W3_f[19:38], axis=0)             # [19, 64, 64, 1]        

    '''####################################################################################'''    
    Wtag = tf.placeholder(tf.string, None)    
    tf.summary.image("images_conv", W3_g, 19)    
    '''===================================================================================='''

#initialize all necessary variable
merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/3", sess.graph)
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

    sess.close()



