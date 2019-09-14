
import tensorflow as tf
#from my_models import Classifier

for i in range(10):
    tf.reset_default_graph()
    # build the graph
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    #classifier = Classifier(global_step)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("do sth here.")