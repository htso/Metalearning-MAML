"""
Usage Instructions:
    10-shot sinusoid:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --metatrain_iterations=70000 --norm=None --update_batch_size=10

    10-shot sinusoid baselines:
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10 --baseline=oracle
        python main.py --datasource=sinusoid --logdir=logs/sine/ --pretrain_iterations=70000 --metatrain_iterations=0 --norm=None --update_batch_size=10

    5-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=32 --update_batch_size=1 --update_lr=0.4 --num_updates=1 --logdir=logs/omniglot5way/

    20-way, 1-shot omniglot:
        python main.py --datasource=omniglot --metatrain_iterations=40000 --meta_batch_size=16 --update_batch_size=1 --num_classes=20 --update_lr=0.1 --num_updates=5 --logdir=logs/omniglot20way/

    5-way 1-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=1 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet1shot/ --num_filters=32 --max_pool=True

    5-way 5-shot mini imagenet:
        python main.py --datasource=miniimagenet --metatrain_iterations=60000 --meta_batch_size=4 --update_batch_size=5 --update_lr=0.01 --num_updates=5 --num_classes=5 --logdir=logs/miniimagenet5shot/ --num_filters=32 --max_pool=True

    To run evaluation, use the '--train=False' flag and the '--test_set=True' flag to use the test set.

    For omniglot and miniimagenet training, acquire the dataset online, put it in the correspoding data directory, and see the python script instructions in that directory to preprocess the data.
"""
import csv
import numpy as np
import pickle
import random
import tensorflow as tf

from data_generator1 import FunGenerator
from maml2 import MAML
from tensorflow.python.platform import flags


def train(model, sess, data_generator, update_batch_size):
    
    for itr in range(3):
        print('G')
        # Do 'weights' change since last iteration ?
        # ans : yes
        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(model):
                training_scope.reuse_variables()
                wgh = sess.run(model.weights)
                print('weights :\n', wgh)
            else:
                print('weights not found in dir(self).')

        feed_dict = {}
        batch_x, batch_y, amp, phase, _, _, _ = data_generator.generate(0)
        # a : training data
        inputa = batch_x[:, :update_batch_size, :]
        labela = batch_y[:, :update_batch_size, :]
        # b: testing data, this is the last row in a batch
        inputb = batch_x[:, update_batch_size:, :] 
        labelb = batch_y[:, update_batch_size:, :]

        feed_dict = {model.inputa:inputa, model.inputb:inputb,  model.labela:labela, model.labelb:labelb}

        some_op = [model.metatrain_op] 

        # do one round of optimization
        result = sess.run(some_op, feed_dict) 
        
    
def main():    
    dg = FunGenerator(10*2, 25, 0)
    dim_output = dg.dim_output
    dim_input = dg.dim_input

    input_tensors = None

    print('MAML')
    model = MAML(dim_input, dim_output, test_num_updates=5)
    print('construct model')
    model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
    
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    print('start train')
    train(model, sess, dg, 10)


if __name__ == "__main__":
    main()
