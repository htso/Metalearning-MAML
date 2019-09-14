""" Code for the MAML algorithm and network definitions. """
from __future__ import print_function
import numpy as np
import sys
import tensorflow as tf
from tensorflow.python.platform import flags
from utils import mse, xent, normalize

FLAGS = flags.FLAGS

class MAML:
    def __init__(self, dim_input=1, dim_output=1, test_grad_steps=1):
        """ must call construct_model() after initializing MAML! """
        self.dim_input = dim_input
        self.dim_output = dim_output
        self.update_lr = FLAGS.update_lr
        self.meta_lr = tf.placeholder_with_default(FLAGS.meta_lr, ())
        self.classification = False
        self.test_grad_steps = test_grad_steps
        self.result = None # [HT added] expose the result of tf.map_fn() call below so i can sess.run it outside
        self.dim_hidden = [FLAGS.hidden1, FLAGS.hidden2]
        self.loss_func = mse # Loss is just mean squared error
        self.forward = self.forward_fc
        self.construct_weights = self.construct_fc_weights

    def construct_model(self, input_tensors=None, prefix='metatrain_'):
        # NOTE : the two input variables inputa, inputb need some explanation.
        #   suffix 'a': training data for inner gradient, 
        #   suffix 'b': test data for meta gradient
        if input_tensors is None:
            self.inputa = tf.placeholder(tf.float32)
            self.inputb = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
            self.labelb = tf.placeholder(tf.float32)
        else:
            self.inputa = input_tensors['inputa']
            self.inputb = input_tensors['inputb']
            self.labela = input_tensors['labela']
            self.labelb = input_tensors['labelb']

        #print('inputa shape :', self.inputa.shape)    

        with tf.variable_scope('model', reuse=None) as training_scope:
            if 'weights' in dir(self):
                # if 'weights' exists in the 'model' scope, then re-use it
                training_scope.reuse_variables()
                # assign it to python name 'weights'
                weights = self.weights
            else:
                # Define the weights
                self.weights = weights = self.construct_weights()

            # outputbs[i] and lossesb[i] is the output and loss after i+1 gradient updates
            lossesa, outputas = [], []
            accuraciesa, accuraciesb = [], []
            #grad_steps = max(self.test_grad_steps, FLAGS.grad_steps)
            grad_steps = FLAGS.grad_steps
            outputbs = [[]]*grad_steps
            lossesb = [[]]*grad_steps
            losses_a = [[]]*grad_steps
            accuraciesb = [[]]*grad_steps

            # ===========================================================================================
            def task_metalearn(inp, reuse=True):
                """ 
                inp : (1, # train pts, 1)

                where # train pts = K * % train split

                Perform gradient descent for one "task", or a dataset from one sine function in the meta-batch. 
                One such dataset is one row of inputa and inputb.

                Therefore, the neural net is trained on inputa only; whatever is in inputb do not affect the
                gradient and thus the weight update. 

                This function is run in parallel in tf.map_fn(..) below. Each row of inputa/inputb is 
                fed into this function to generate one task_outputa/bs, task_lossa/sb.

                NOTE : theta = 'weights', which is created by calling construct_weights() inside scope 'model'
                       so that it's visible in that scope. 
                """
                # NOTE : inputa <==> one row in x_train <==> one dataset <==> one sine function
                inputa, inputb, labela, labelb = inp

                # each element in the list represents one gradient update
                task_outputbs, task_lossesb, task_losses_a = [], [], []

                # ---- Inner loop first gradient update ----------------------------------------------------
                # In Finn's paper, Eq (1)
                #
                #     weights <==> theta
                #     fast_weights <==> theta_i
                #     
                #     theta_i(t) = theta_i(t-1) - beta * grad_i 
                #
                # Forward pass : input(A) --> loss(A) --> grad(A)
                # NOTE : Gradient and weight updates are calculated off from input(A)    
                task_outputa = self.forward(inputa, weights, reuse=reuse)  # only reuse on the first iter, see normalize() in utils.py
                
                task_lossa = self.loss_func(task_outputa, labela)
                #tf.Print(task_lossa, [task_lossa], message="task_lossa print")

                task_losses_a.append(task_lossa)
                # NOTE : gradients are calculated from inputa, thus it's learning from the training set
                # DATA NOTE : 'weights' is a dict, so weights.values() gets the values of the key-value pairs, and the 'values'
                #        are the symbolic variables like w1, b1, etc.
                grads = tf.gradients(task_lossa, list(weights.values()))

                if FLAGS.stop_grad:
                    grads = [tf.stop_gradient(grad) for grad in grads]
                # remake a dict from the keys of weights; grads is list of symbolic variables    
                gradients = dict(zip(weights.keys(), grads))
                # 1st update : theta_i = theta_0 - lr * grad_i
                fast_weights = dict(zip(weights.keys(), [weights[key] - self.update_lr*gradients[key] for key in weights.keys()]))

                # Prediction on the validation set based on one gradient update.
                # calc validation loss ==> loss(B)
                # Forward pass : input(B) --> output(B) --> loss(B)
                output = self.forward(inputb, fast_weights, reuse=True)
                task_outputbs.append(output)
                task_lossesb.append(self.loss_func(output, labelb))

                # Two or more gradient updates carried out in loop
                # NOTE : Gradients and therefore weight updates are based entirely on input(A)    
                for j in range(grad_steps - 1):
                    loss = self.loss_func(self.forward(inputa, fast_weights, reuse=True), labela)
                    task_losses_a.append(loss)
                    grads = tf.gradients(loss, list(fast_weights.values()))
                    if FLAGS.stop_grad:
                        grads = [tf.stop_gradient(grad) for grad in grads]
                    gradients = dict(zip(fast_weights.keys(), grads))
                    fast_weights = dict(zip(fast_weights.keys(), [(fast_weights[key] - self.update_lr*gradients[key]) for key in fast_weights.keys()]))
                    # Prediction on the validation set :
                    # NOTE : the following code has *no effect* on the model. it's purely for performance tracking purposes
                    output = self.forward(inputb, fast_weights, reuse=True)
                    task_outputbs.append(output)
                    task_lossesb.append(self.loss_func(output, labelb))

                in_sample_fit_A = self.forward(inputa, fast_weights, reuse=True)

                task_output = [task_outputa, task_outputbs, task_losses_a, task_lossa, task_lossesb, in_sample_fit_A]

                return task_output
            # =============================================================================================

            if FLAGS.norm is not 'None': # [????]
                # to initialize the batch norm vars, might want to combine this, and not run idx 0 twice. [????]
                unused = task_metalearn((self.inputa[0], self.inputb[0], self.labela[0], self.labelb[0]), reuse=False)

            # ----- run task_metalearn as many times as batch size in parallel ----------------------------------
            # NOTE : put the result into 'self.result' so I can see it from outside, i.e. by using 'model.result'
            out_dtype = [tf.float32, [tf.float32]*grad_steps, [tf.float32]*grad_steps, tf.float32, [tf.float32]*grad_steps, tf.float32]
            self.result = tf.map_fn(task_metalearn, elems=(self.inputa, self.inputb, self.labela, self.labelb), \
                               dtype=out_dtype, parallel_iterations=FLAGS.meta_batch_size)

            outputas, outputbs, task_losses_a, lossesa, lossesb, in_sample_fit_A = self.result

        # ==== Outer loop gradient update ============================================================================================
        # This is the meta learning part, which performs gradient descent on the validation loss (total_losses2[last step])
        if 'train' in prefix:
            # print('train is in prefix.')
            # 1 <==> 'A'. total_loss1 is one single value
            # 2 <==> 'B'. total_losses2 is a list of length = grad_steps
            self.total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32)
            self.total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32) for j in range(grad_steps)]
            # after the map_fn
            self.outputas, self.outputbs, self.in_sample_fit_A = outputas, outputbs, in_sample_fit_A
            #self.outputas, self.outputbs = outputas, outputbs

            # ===========================================================================
            # IMPORTNAT NOTE : pretrain optimization is performed on total_loss1, 
            #                  which is based on input(A). Why is ther a difference 
            #                  betwn pretrain and train ?
            self.pretrain_op = tf.train.AdamOptimizer(self.meta_lr).minimize(total_loss1)
            # ===========================================================================

            if FLAGS.metatrain_iterations > 0:
                optimizer = tf.train.AdamOptimizer(self.meta_lr)
                # ---->                                                        input(B)         last grad step
                #                                                                  |               |
                #                                                                  |               |
                self.gvs = gvs = optimizer.compute_gradients(loss=self.total_losses2[FLAGS.grad_steps-1], var_list=None)
                # NOTE : The only reason to use compute_gradients and follow with apply_gradients is to
                #        allow the following clipping operation on gradient. Otherwise, it should be written as 
                #        one line : optimizer.minimize(loss)
                # ====================================================================
                # IMPORTANT NOTE :
                #    Optimization is done on total_losses2, which is based on input(B)
                #    which has the losses from multiple grad steps, but this only uses
                #    the last update.
                self.metatrain_op = optimizer.apply_gradients(gvs)
                # ====================================================================
        else:
            self.metaval_total_loss1 = total_loss1 = tf.reduce_sum(lossesa) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32)
            self.metaval_total_losses2 = total_losses2 = [tf.reduce_sum(lossesb[j]) / tf.cast(FLAGS.meta_batch_size, dtype=tf.float32) for j in range(grad_steps)]
            # This line is added to check model.result during testing.
            self.outputas, self.outputbs, self.in_sample_fit_A = outputas, outputbs, in_sample_fit_A

        ## Summaries
        tf.summary.scalar(prefix+'Pre-update loss', total_loss1)
        for j in range(grad_steps):
            tf.summary.scalar(prefix+'Post-update loss, step ' + str(j+1), total_losses2[j])
            
    ### Construct FC networks
    def construct_fc_weights(self):
        # weights is a dictionary
        weights = {}
        weights['w1'] = tf.Variable(tf.truncated_normal([self.dim_input, self.dim_hidden[0]], stddev=0.01))
        weights['b1'] = tf.Variable(tf.zeros([self.dim_hidden[0]]))
        for i in range(1,len(self.dim_hidden)):
            weights['w'+str(i+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[i-1], self.dim_hidden[i]], stddev=0.01))
            weights['b'+str(i+1)] = tf.Variable(tf.zeros([self.dim_hidden[i]]))
        weights['w'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.truncated_normal([self.dim_hidden[-1], self.dim_output], stddev=0.01))
        weights['b'+str(len(self.dim_hidden)+1)] = tf.Variable(tf.zeros([self.dim_output]))
        return weights

    def forward_fc(self, inp, weights, reuse=False):
        hidden = normalize(tf.matmul(inp, weights['w1']) + weights['b1'], activation=tf.nn.relu, reuse=reuse, scope='0')
        for i in range(1,len(self.dim_hidden)):
            hidden = normalize(tf.matmul(hidden, weights['w'+str(i+1)]) + weights['b'+str(i+1)], activation=tf.nn.relu, reuse=reuse, scope=str(i+1))
        return tf.matmul(hidden, weights['w'+str(len(self.dim_hidden)+1)]) + weights['b'+str(len(self.dim_hidden)+1)]


