"""
Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys
import keras


# Hyper Parameters
TIME_STEP = 10       # rnn time step
INPUT_SIZE = 1      # rnn input size
CELL_SIZE = 32      # rnn cell size
LR = 0.02           # learning rate

# show data
steps = np.linspace(0, np.pi*2, 100, dtype=np.float32)
x_np = np.sin(steps)    # float32 for converting torch FloatTensor
y_np = np.cos(steps)
print("x_np shape :", x_np.shape)
print("y_np shape :", y_np.shape)

#plt.plot(steps, y_np, 'r-', label='target (cos)')
#plt.plot(steps, x_np, 'b-', label='input (sin)')
#plt.legend(loc='best')
#plt.show()

# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # shape : (batch, 5, 1)
tf_y = tf.placeholder(tf.float32, [None, TIME_STEP, INPUT_SIZE])        # y : same shape

# RNN
#rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
rnn_cell = tf.keras.layers.LSTMCell(units=CELL_SIZE)
init_s = rnn_cell.zero_state(batch_size=1, dtype=tf.float32)    # very first hidden state
outputs, final_s = tf.keras.layers.RNN(
    rnn_cell,                   # cell you have chosen
    tf_x,                       # input
    initial_state=init_s,       # the initial hidden state
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
print('outputs :', outputs.shape)
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       # reshape 3D output to 2D for fully connected layer
print('outs2D :', outs2D.shape)
net_outs2D = keras.layers.dense(outs2D, INPUT_SIZE)                    # FC layer
print('net_outs2D :', net_outs2D.shape)
outs = tf.reshape(net_outs2D, [-1, TIME_STEP, INPUT_SIZE])          # reshape back to 3D
print('outs :', outs.shape)
sys.exit()



loss = tf.losses.mean_squared_error(labels=tf_y, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())     # initialize var in graph

plt.figure(1, figsize=(12, 5))
plt.ion()           # continuously plot

for step in range(10):
    start, end = step * np.pi, (step+1)*np.pi   # time range
    # use sin predicts cos
    steps = np.linspace(start, end, TIME_STEP)
    x = np.sin(steps)[np.newaxis, :, np.newaxis]    # shape (batch, time_step, input_size)
    y = np.cos(steps)[np.newaxis, :, np.newaxis]
    # x, y shape : (1, TIME_STEP, 1)
    if step == 0:
        print("x shape :", x.shape)
        print("y shape :", y.shape)
    if 'final_s_' not in globals():                 # first state, no any hidden state
        feed_dict = {tf_x: x, tf_y: y}
    else:                                           # has hidden state, so pass it to rnn
        feed_dict = {tf_x: x, tf_y: y, init_s: final_s_}
    A1, pred_, final_s_ = sess.run([train_op, outs, final_s], feed_dict)     # train
    # A1 is NoneType
    # pred_ shape : (1, TIME_STEP, 1)
    # final_s_ : (1, 32)
    if step == 1:
        print("A1 type :", type(A1))
        print("pred shape :", pred_.shape)
        print("final_s_ shape :", final_s_.shape)

    print('final_s_ : ', final_s_)
    # plotting
    plt.plot(steps, y.flatten(), 'r-')
    plt.plot(steps, pred_.flatten(), 'b-')
    plt.ylim((-1.2, 1.2))
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()


