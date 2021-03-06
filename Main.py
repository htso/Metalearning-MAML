"""
Reproduce Finn et al 2017, Fig 3, Table 2, Fig 7.

To train:

    python Main.py --datasource=fun --Type=Finn --metatrain_iterations=15000 --hidden1=40 --hidden2=40 --grad_steps=5 --K=10 --n_val=10 --n_test=0 --train=True

After training is completed:

    python Main.py --datasource=fun --Type=periodic --meta_batch_size=8 --grad_steps=5 --K=20 --n_val=20 --n_test=20 --x_left=-1.59 --x_right=1.59 --w_lb=0.15915  --w_ub=0.15915 --train=False

NOTE : Set --train=False and --n_test=nn to do out-of-sample testing. Need to set x_right to something outside
of the original training range. For ex, if trained on [-5, 5] (x_left=-1.59 & x_right=1.59), then you may want to 
test on [5, 10]. Set x_right=3.18 ==> 3.18*3.14 ~ 10.
"""

import sys
import os
import csv
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.platform import flags
from tensorflow.python import debug as tf_debug
from matplotlib.backends.backend_pdf import PdfPages

from data_generator_ht import DataGenerator
from Maml import MAML

from FunGenerator import FunGenerator
from InitializationGenerator import *
from misc_utils import *

FLAGS = flags.FLAGS

## Dataset/method options
flags.DEFINE_string('datasource', 'fun', 'sinusoid or fun')
## Training options
flags.DEFINE_integer('K', 5, 'number of examples used for inner gradient update (the K in K-shot learning).')
flags.DEFINE_integer('n_val', 5, 'number of examples used for validation in inner gradient update.')
flags.DEFINE_integer('n_test', 0, 'number of examples used for out-of-sample testing.')
flags.DEFINE_integer('hidden1', 40, 'size of first hidden layer in FC net.')
flags.DEFINE_integer('hidden2', 40, 'size of second hidden layer in FC net.')
flags.DEFINE_integer('metatrain_iterations', 15000, 'number of metatraining iterations.') # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.') # default is no pretraining
flags.DEFINE_float('meta_lr', 0.001, 'beta, the base learning rate of the generator')
flags.DEFINE_float('update_lr', 0.01, 'alpha, or step size for inner gradient update.') # 0.1 for omniglot
flags.DEFINE_integer('grad_steps', 1, 'number of inner gradient updates during training.')
flags.DEFINE_integer('test_grad_steps', 20, 'number of gradient updates during test.')

# oracle means task id is input (only suitable for sinusoid)
flags.DEFINE_string('baseline', None, 'oracle, or None')
## Model options
flags.DEFINE_string('norm', 'None', 'batch_norm, layer_norm, or None')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')

## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if False, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', 'logs/fun', 'directory for summaries and checkpoints.')
flags.DEFINE_bool('resume', False, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_integer('test_iter', -1, 'iteration to load model (-1 for latest model)')
flags.DEFINE_bool('test_set', False, 'Set to True to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_K', -1, 'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1, 'value of inner gradient step during training. (use if you want to test with a different value)') # 0.1 for omniglot
flags.DEFINE_bool('DEBUG', False, 'Turn debug mode on. Default is False')

# parameters for data generator 
flags.DEFINE_float('n_sd', 0.1, 'std dev of noise for data generator.')
flags.DEFINE_float('x_left', -1.59, 'left end of x, in multiple of pi.')
flags.DEFINE_float('x_right', 1.59, 'right end of x, in multiple of pi.')
flags.DEFINE_float('amp_lb', 0.1, 'amplitude lower bound.')
flags.DEFINE_float('amp_ub', 5.0, 'amplitude upper bound.')
flags.DEFINE_float('w_lb', 0.15915, 'angular frequency, lower bound.')
flags.DEFINE_float('w_ub', 0.15915, 'angular frequency, upper bound.')
flags.DEFINE_float('ph_lb', 0.0, 'phase lower bound, in multiple of pi.')
flags.DEFINE_float('ph_ub', 1.0, 'phase upper bound, in multiple of pi.')
flags.DEFINE_float('offset_lb', -1.0, 'linear offset, lower bound.')
flags.DEFINE_float('offset_ub', 1.0, 'linear offset, upper bound.')
flags.DEFINE_float('slp_lb', -2.0, 'slope, lower bound.')
flags.DEFINE_float('slp_ub', 2.0, 'slope, upper bound.')
flags.DEFINE_integer('fType', 0, '{0,1,2}, type of periodic function, see explanation in FunGenerator.py')
flags.DEFINE_string('Type', 'periodic', "either 'periodic', 'linear', or 'Finn', see explanation in FunGenerator.py")



def train(model, saver, sess, exp_string, data_generator, resume_itr=0, gnm="Loss.jpg"):

    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 500
    if FLAGS.datasource == 'sinusoid':
        PRINT_INTERVAL = 500
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    elif FLAGS.datasource == 'fun':
        PRINT_INTERVAL = 500
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*2
    else:
        PRINT_INTERVAL = 100
        TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    if FLAGS.log:
        train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, sess.graph)

    print('\n\nDone initializing, start training.....')

    iter_ix, iter_ix1, prelosses, postlosses, test_total_loss1, test_total_losses2 = [], [], [], [], [], []
    
    # save the initial parameters (ie. random starts) before entering the loop
    Wval = sess.run(model.weights)
    W1_start = Wval['w1']
    W2_start = Wval['w2']
    W3_start = Wval['w3']
    b1_start = Wval['b1']
    b2_start = Wval['b2']
    b3_start = Wval['b3']
    W1_prev, W2_prev, W3_prev, b1_prev, b2_prev, b3_prev = W1_start, W2_start, W3_start, b1_start, b2_start, b3_start
      
    W1_angles, W2_angles, W3_angles, b1_angles, b2_angles = [], [], [], [], [] 
    W1_dist, W2_dist, W3_dist, b1_dist, b2_dist, b3_dist = [], [], [], [], [], []

    #multitask_weights, reg_weights = [], []  # what for, where used ?

    for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            if FLAGS.datasource == 'fun':
                res = data_generator.generate()
                x_train = res["x_train"]
                x_val = res["x_val"]
                x_test = res["x_test"]
                y_train = res["y_train"]
                y_val = res["y_val"]
                y_test = res["y_test"]
                x_eq = res["x_equal_spaced"]
                y_eq = res["y_equal_spaced"]
            elif FLAGS.datasource == 'sinusoid':
            	# batch_x, batch_y shape:
                #
                #     batch_x = [fun_i, x_i, 1]
                #     batch_y = [fun_i, f(x_i), 1]
                #
                # where fun_i identifies a specific functional form, x_i the x-coordinates on which f(x_i) is evaluated.
                batch_x, batch_y, amp, phase = data_generator.generate(train=False)
            else:
                raise ValueError('i don\'t know this datasource...')
                
            if FLAGS.baseline == 'oracle' and FLAGS.datasource == 'sinusoid':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(FLAGS.meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            if FLAGS.datasource == 'sinusoid':
            	# ---- maml data generator arrangement ----------------------------------------------------------------
                # A batch is splitted into two halves, inputa and inputb. These two parts inputa, inputb come from the same function.
                # They are evaluated at different x_i. The x values are not sorted, so those in inputa might be greater than those in inputb.
                # The goal is to make the submodels M({x}_i, theta_i) learn and test on data from the same function.
                inputa = batch_x[:, :FLAGS.K, :] # a is for few-shot training
                labela = batch_y[:, :FLAGS.K, :]
                inputb = batch_x[:, FLAGS.K:, :] # b is for validation
                labelb = batch_y[:, FLAGS.K:, :]
            elif FLAGS.datasource == 'fun':
            	# ---- FunGenerator data arrangement ----------------------------------
            	# The generator returns a train and a validation set. No need for the complicated split.
                inputa = x_train # a is for few-shot training
                labela = y_train
                inputb = x_val # b is for validation
                labelb = y_val

            feed_dict = {model.inputa:inputa, model.inputb:inputb, model.labela:labela, model.labelb:labelb}

        if itr < FLAGS.pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            # training op ======================
            # NOTE :
            # metatrain.op is a sym variable, 
            #
            #     metatrain_op = optimizer.minimize(loss)
            #
            # By running it in sess.run() below, it's doing one grad update  
            input_tensors = [model.metatrain_op]
            # ==================================

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            # ?????
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[FLAGS.grad_steps-1]])

        # ==== one grad update on theta (initial weights) =====
        # NOTE : This is just sess.run(metatrain_op, feed_dict) 
        result = sess.run(input_tensors, feed_dict)
        # =====================================================
        if itr % SUMMARY_INTERVAL == 0:
            if FLAGS.log:
                train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1]) # last element of the list (result[-1]) is task_lossesb, a list
            prelosses.append(result[-2]) # next to last element (result[-2]) is task_lossa
            iter_ix.append(itr)

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < FLAGS.pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iter' + str(itr - FLAGS.pretrain_iterations)
            print_str += ' preslosses : ' + str(np.mean(prelosses[-100:])) + ', postlosses :' + str(np.mean(postlosses[-100:]))
            print(print_str)

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

        # sinusoid is infinite data, so no need to test on meta-validation set. [????]
        if (itr!=0) and (itr % TEST_PRINT_INTERVAL) == 0:
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.grad_steps-1], model.summ_op]
            else:
                if FLAGS.datasource == 'sinusoid':
                    batch_x, batch_y, amp, phase= data_generator.generate(train=False)
                    inputa = batch_x[:, :FLAGS.K, :]
                    inputb = batch_x[:, FLAGS.K:, :]
                    labela = batch_y[:, :FLAGS.K, :]
                    labelb = batch_y[:, FLAGS.K:, :]
                elif FLAGS.datasource == 'fun':
                    res = data_generator.generate()
                    x_train = res["x_train"]
                    x_val = res["x_val"]
                    x_test = res["x_test"]
                    y_train = res["y_train"]
                    y_val = res["y_val"]
                    y_test = res["y_test"]
                    inputa = x_train # a is for few-shot training
                    labela = y_train
                    inputb = x_test # b is for validation
                    labelb = y_test
                else:
                    raise ValueError("i don't know this datasource.")

                feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}
                input_tensors = [model.total_loss1, model.total_losses2[FLAGS.grad_steps-1]]

            result = sess.run(input_tensors, feed_dict)
            test_total_loss1.append(result[0])
            test_total_losses2.append(result[1])
            iter_ix1.append(itr)
            print('\tTest set total_loss1: ' + str(result[0]) + ', total_losses2 : ' + str(result[1]))

            # ==== Parameter tracking and analysis ===================
            #print('dir(model):\n', dir(model))
            Wval = sess.run(model.weights)
            Wn_1 = Wval['w1']
            Wn_2 = Wval['w2']
            Wn_3 = Wval['w3']
            bn_1 = Wval['b1']
            bn_2 = Wval['b2']
            bn_3 = Wval['b3']

            # Angle between two adjacent parameter vectors            
            W1_angles.append(Step_CosineSimilarity(Wn_1, W1_prev)[0])
            W2_angles.append(Step_CosineSimilarity(Wn_2, W2_prev)[0])
            W3_angles.append(Step_CosineSimilarity(Wn_3, W3_prev)[0])
            b1_angles.append(Step_CosineSimilarity(bn_1, b1_prev)[0])
            b2_angles.append(Step_CosineSimilarity(bn_2, b2_prev)[0])
            W1_prev = Wn_1
            W2_prev = Wn_2
            W3_prev = Wn_3
            b1_prev = bn_1
            b2_prev = bn_2
            b3_prev = bn_3

            # Distance of current param from the initial parameter values
            # I use it as a measure of learning, ie. how far have they moved from the state of
            # complete ignorance. Euclidean distance is used; not sure if the choice
            # matters at all.
            W1_dist.append(Step_L2_Distance(Wn_1, W1_start))
            W2_dist.append(Step_L2_Distance(Wn_2, W2_start))
            W3_dist.append(Step_L2_Distance(Wn_3, W3_start))
            b1_dist.append(Step_L2_Distance(bn_1, b1_start))
            b2_dist.append(Step_L2_Distance(bn_2, b2_start))
            b3_dist.append(Step_L2_Distance(bn_3, b3_start))

    #pp = PdfPages(gnm)
    fig, ax = plt.subplots(4, 2, sharex=False, sharey=False, figsize=(10, 20))
    
    ax[0,0].plot(iter_ix, prelosses, linestyle='-', marker='o', markersize=6, color='blue', label="prelosses")            
    ax[0,0].grid(True)
    ax[0,0].set_title('Training Loss (prelosses)')
    ax[0,0].set_xlabel('meta iteration')
    
    ax[0,1].plot(iter_ix, postlosses, linestyle='-', marker='o', markersize=6, color='red', label="postlosses")            
    ax[0,1].grid(True)
    ax[0,1].set_title('Validation Loss (postlosses)')
    ax[0,1].set_xlabel('meta iteration')

    ax[1,0].plot(iter_ix1, test_total_loss1, linestyle='-', marker='o', markersize=6, color='red', label="test_total_loss1")            
    ax[1,0].grid(True)
    ax[1,0].set_title('Test Loss (test_total_loss1)')
    ax[1,0].set_xlabel('meta iteration')

    ax[1,1].plot(iter_ix1, test_total_losses2, linestyle='-', marker='o', markersize=6, color='red', label="test_total_losses2")            
    ax[1,1].grid(True)
    ax[1,1].set_title('Test Loss on out-of-sample data (test_total_losses2)')
    ax[1,1].set_xlabel('meta iteration')

    ax[2,0].plot(W1_angles, color='green', label='W1')
    ax[2,0].plot(W2_angles, color='red', label='W2')
    ax[2,0].plot(W3_angles, color='blue', label='W3')
    ax[2,0].set_title('W_i angles at test epoches', fontsize=10)
    ax[2,0].grid(True)
    ax[2,0].legend(loc="upper right")

    ax[2,1].plot(W1_dist, color='green', label='W1')
    ax[2,1].plot(W2_dist, color='red', label='W2')
    ax[2,1].plot(W3_dist, color='blue', label='W3')
    ax[2,1].set_title('W_i distance from start at test epoches', fontsize=8)
    ax[2,1].grid(True)
    ax[2,1].legend(loc="upper left")

    ax[3,0].plot(b1_dist, color='blue', label='b1')
    ax[3,0].plot(b2_dist, color='red', label='b2')
    ax[3,0].set_title('b_i distance from start at test epoches', fontsize=8)
    ax[3,0].grid(True)
    ax[3,0].legend(loc="upper left")

    #pp.savefig(fig, orientation = 'landscape')
    #pp.close()
    plt.savefig(gnm)    

    saver.save(sess, FLAGS.logdir + '/' + exp_string +  '/model' + str(itr))

    with open(FLAGS.logdir + '/' + exp_string + '_Param.pkl', 'wb') as f:
        pickle.dump({'prelosses': prelosses}, f)
        pickle.dump({'postlosses': postlosses}, f)
        pickle.dump({'test_total_loss1': test_total_loss1}, f)
        pickle.dump({'test_total_losses2': test_total_losses2}, f)
        pickle.dump({'W1_angles': W1_angles}, f)
        pickle.dump({'W2_angles': W2_angles}, f)
        pickle.dump({'W3_angles': W3_angles}, f)
        pickle.dump({'W1_dist': W1_dist}, f)
        pickle.dump({'W2_dist': W2_dist}, f)
        pickle.dump({'W3_dist': W3_dist}, f)
        pickle.dump({'b1_dist': b1_dist}, f)
        pickle.dump({'b2_dist': b2_dist}, f)
        pickle.dump({'W1_start': W1_start}, f)
        pickle.dump({'W2_start': W2_start}, f)
        pickle.dump({'W3_start': W3_start}, f)
        pickle.dump({'b1_start': b1_start}, f)
        pickle.dump({'b2_start': b2_start}, f)
        pickle.dump({'b3_start': b3_start}, f)



def test(model, saver, sess, exp_string, data_generator, test_grad_steps, gnm):
    print('\nK :', FLAGS.K)
    print('test_grad_steps :', test_grad_steps)
    print('exp_string :', exp_string)

    NUM_FUN = FLAGS.meta_batch_size

    OutAs = []
    OutBs = []

    #pp = PdfPages(gnm)
    fig, ax = plt.subplots(NUM_FUN, 2, sharex=True, sharey=False, figsize=(11, 20))

    for ii in range(NUM_FUN):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            if FLAGS.datasource == 'sinusoid':
                batch_x, batch_y, amp, phase = data_generator.generate(train=False, test_amp_rng=None, test_offset=None)
            elif FLAGS.datasource == 'fun':
                res = data_generator.generate()
                x_train = res["x_train"]
                x_val = res["x_val"]
                x_test = res["x_test"]
                y_train = res["y_train"]
                y_val = res["y_val"]
                y_test = res["y_test"]
                x_eq = res["x_equal_spaced"]
                y_eq = res["y_equal_spaced"]

                inputa = x_train # a is for few-shot training
                labela = y_train
                inputb = x_test # b is for validation
                labelb = y_test
            else:
                raise ValueError('datasource must be either sinusoid or fun.')                

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            if FLAGS.datasource == 'sinusoid':
                inputa = batch_x[:, :FLAGS.K, :]
                labela = batch_y[:, :FLAGS.K, :]
                inputb = batch_x[:, FLAGS.K:, :]
                labelb = batch_y[:, FLAGS.K:, :]
                x_eq = batch_x
                y_eq = batch_y

            # why feed a meta_lr, which is 0.0 ?
            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        # python : [value] + some_list = a list with value added to the top of the list
        tot_loss = sess.run([model.total_loss1] + model.total_losses2, feed_dict)

        # CHECK against what i got from model.result 
        outAs1, outBs1, in_sample_fit_A  = sess.run([model.outputas, model.outputbs, model.in_sample_fit_A], feed_dict)      
        outAs1 = np.array(outAs1)
        outBs1 = np.array(outBs1)
        in_sample_fit_A = np.array(in_sample_fit_A)
        # Fit on training data (inputa)
        x1 = inputa[ii,:,0]
        ix1 = np.argsort(x1) # sort the x values, matplotlib is really stupid !
        y1 = labela[ii,:,0]
        y_fit = in_sample_fit_A[ii,:,0] 
        ax[ii,0].plot(x_eq[ii,:,0], y_eq[ii,:,0], linewidth=1)
        ax[ii,0].scatter(x1[ix1], y1[ix1], s=20, color='black', label="Actual (inputa)")
        ax[ii,0].scatter(x1[ix1], y_fit[ix1], s=20, alpha=0.8, color='red', label="In-sample Fit")
        ax[ii,0].grid(True)
        if ii == 0:
            ax[ii,0].legend(loc="upper right")
            ax[ii,0].set_title('In-sample Actual vs Fit\ngrad_steps : %d\nK : %d, n_val : %d' % (FLAGS.grad_steps, FLAGS.K, FLAGS.n_val))
            
        # Predict on test data (inputb)
        ax[ii,1].plot(x_eq[ii,:,0], y_eq[ii,:,0], linewidth=1)
        ax[ii,1].plot(inputb[ii,:,0], labelb[ii,:,0], linestyle='-', marker='o', markersize=6, color='blue', label="Actual (inputb)")            
        ax[ii,1].plot(inputb[ii,:,0], outBs1[0,ii,:,0], linestyle='-', marker='o', markersize=6, color='green', label="Predict aft 1 grad step")
        ax[ii,1].plot(inputb[ii,:,0], outBs1[test_grad_steps-1,ii,:,0], linestyle='-', marker='o', markersize=6, color='red', label="Predict aft %d steps" % test_grad_steps)
        ax[ii,1].grid(True)          
        if ii == 0:
            ax[ii,1].legend(loc="lower left")
            ax[ii,1].set_title('Out-of-sample Actual vs Predict\nn_test : %d' % (FLAGS.n_test))
            
        OutAs.append(outAs1)
        OutBs.append(outBs1)
            
    #pp.savefig(fig, orientation = 'landscape')
    #pp.close()
    plt.tight_layout()
    plt.savefig(gnm)


def main():
    
    model_folder = 'ReproduceFinn_noNorm_mbs=25_K=10_grad_steps=5_updateLR=0.01Iter=15000_H=40-40'
    
    split = [FLAGS.K, FLAGS.n_val, FLAGS.n_test]
    print('split : ', split)

    # Data characteristics for FunGenerator
    x_rng = [FLAGS.x_left*np.pi, FLAGS.x_right*np.pi]
    amp_rng = [FLAGS.amp_lb, FLAGS.amp_ub]
    w = [FLAGS.w_lb, FLAGS.w_ub]
    ph = [FLAGS.ph_lb, FLAGS.ph_ub]
    n_sd = FLAGS.n_sd
    offset = [FLAGS.offset_lb, FLAGS.offset_ub]
    slp = [FLAGS.slp_lb, FLAGS.slp_ub]
    fType = [FLAGS.fType, FLAGS.fType]
    Type = FLAGS.Type
    param_rng = {'x':x_rng, 'amp':amp_rng, 'freq':w, 'phase':ph, "lin_slp":slp, "lin_offset":offset, \
                 'function':fType, "innovation_sd":n_sd, "Type":Type }
    print('param_rng : ', param_rng)    

    if Type == 'periodic':
        exp_string = 'Fun'
        if fType[0] == fType[1]:
            exp_string += '=' + str(fType[0])
        else:
            exp_string += '=' + str(fType[0]) + '-' + str(fType[1])
        if ph[0] == ph[1]:
            exp_string += '_ph=' + "{:.1f}".format(ph[0])
        else:
            exp_string += '_ph=' + "{:.1f}".format(ph[0]) + '-' + "{:.1f}".format(ph[1])   
        if w[0] == w[1]:
            exp_string += '_w=' + "{:.1f}".format(w[0])
        else:
            exp_string += '_w=' + "{:.1f}".format(w[0]) + '-' + "{:.1f}".format(w[1])
        if amp_rng[0] == amp_rng[1]:
            exp_string += '_amp=' + "{:.1f}".format(amp_rng[0])
        else:
            exp_string += '_amp=' + "{:.1f}".format(amp_rng[0]) + '-' + "{:.1f}".format(amp_rng[1])
    elif Type == 'linear':
        exp_string = 'Linear'
        if slp[0] == slp[1]:
            exp_string += '_slp=' + str(slp[0])
        else:
            exp_string += '_slp=' + str(slp[0]) + '-' + str(slp[1])
        if offset[0] == offset[1]:
            exp_string += '_ofs=' + str(offset[0])
        else:
            exp_string += '_ofs=' + str(offset[0]) + '-' + str(offset[1])    
    elif Type == 'Finn':
        exp_string = 'ReproduceFinn'
    else:
        raise ValueError('i have no idea what is this Type.')  

    if FLAGS.train == False:
        orig_meta_batch_size = FLAGS.meta_batch_size
        # always use meta batch size of 1 when testing.
        #FLAGS.meta_batch_size = 1

    if FLAGS.datasource == 'sinusoid':
        # Need a factor of 2 in num_samples_per_fun because half is for training, the other
        # half for testing. So first half is inputa, 2nd half inputb
        data_generator = DataGenerator(FLAGS.K*2, FLAGS.meta_batch_size)
    elif FLAGS.datasource == 'fun':
        data_generator = FunGenerator(train_test_split=split, \
        	                          batch_size=FLAGS.meta_batch_size, \
        	                          param_range=param_rng, \
        	                          dim_input=1, dim_output=1)
    else:
        raise ValueError('datasource must be either sinusoid or fun.')  

    dim_output = data_generator.dim_output
    if FLAGS.baseline == 'oracle':
        assert FLAGS.datasource == 'sinusoid'
        dim_input = 3
        FLAGS.pretrain_iterations += FLAGS.metatrain_iterations
        FLAGS.metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

    tf_data_load = False # what is this for?

    input_tensors = None

    # STEP 1. Create model ===================================================
    model = MAML(dim_input, dim_output, test_grad_steps=FLAGS.test_grad_steps)
    # ========================================================================

    # STEP 2. Contruct the model, define task_metalearn method in this call 
    if FLAGS.train or not tf_data_load:
        # build graph =========================================================
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        # =====================================================================
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')

    model.summ_op = tf.summary.merge_all()
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=100)
    sess = tf.InteractiveSession()
    
    if FLAGS.DEBUG is True:
        tf.logging.set_verbosity(tf.logging.ERROR)
        sess = tf_debug.LocalCLIDebugWrapperSession(sess)

    if FLAGS.train == False:
        # change to original meta batch size when loading model.
        FLAGS.meta_batch_size = orig_meta_batch_size

    if FLAGS.train_K == -1:
        FLAGS.train_K = FLAGS.K
    if FLAGS.train_update_lr == -1:
        FLAGS.train_update_lr = FLAGS.update_lr

    if FLAGS.stop_grad:
        exp_string += '_stopgrad'
    if FLAGS.baseline:
        exp_string += FLAGS.baseline
    if FLAGS.norm == 'batch_norm':
        exp_string += '_batchnorm'
    elif FLAGS.norm == 'layer_norm':
        exp_string += '_layernorm'
    elif FLAGS.norm == 'None':
        exp_string += '_noNorm'
    else:
        raise ValueError('Norm setting not recognized.')

    # Make the folder name as informative as possible
    exp_string += '_mbs='+str(FLAGS.meta_batch_size) + '_K=' + str(FLAGS.train_K) + '_grad_steps=' + str(FLAGS.grad_steps) + '_updateLR=' + str(FLAGS.train_update_lr) + 'Iter=' + str(FLAGS.metatrain_iterations)
    exp_string += '_H=' + str(FLAGS.hidden1) + '-' + str(FLAGS.hidden2)
    print('\nexp_string : ', exp_string)

    resume_itr = 0
    model_file = None    

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if FLAGS.resume or not FLAGS.train:
        #model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
        model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + model_folder)
        print('\nmodel_file : \n\t', model_file)
        if FLAGS.test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(FLAGS.test_iter)
        if model_file:
            ind1 = model_file.index('model')
            # print('index(\"model\") :', ind1)
            resume_itr = int(model_file[ind1+5:])
            # print('resume_itr :', resume_itr)
            # print('bef restore, does var weights exist?', )
            # if 'weights' in globals():
            #     print('yes')
            # else:
            #     print('noop')
            print("Restoring model weights from " + model_file)
            # ===============================
            saver.restore(sess, model_file)
            # ===============================
            # Write weights parameters to file for use in FitPerFun.py
            with tf.variable_scope('model', reuse=None) as old_scope:
                # print('dir(model):\n', dir(model))
                if 'weights' in dir(model):
                    # if 'weights' exists in the 'model' scope, then re-use it
                    old_scope.reuse_variables()
                    # model.weights is a dict
                    weights = model.weights
                else:
                    print('i don`t see any weights in model!')

    # IMPORTANT NOTE : use the same data generator for train and test, otherwise
    # it would be asking a model to predict something it's never seen before. 
    if FLAGS.train:
        print('.... call train() .............................')
        gnm = FLAGS.logdir + '/' + 'Loss_Angles_Distances_over_iter' + '.jpg'
        train(model, saver, sess, exp_string, data_generator, resume_itr, gnm=gnm)
    else:
        print('....call test() ...................')
        #gnm = FLAGS.logdir + '/' + 'Fit_Predict_Plots' + '.pdf'
        gnm = FLAGS.logdir + '/' + 'Fit_Predict_Plots' + '.jpg'
        print('gnm : ', gnm)
        test(model, saver, sess, model_folder, data_generator, FLAGS.grad_steps, gnm=gnm)
        
if __name__ == "__main__":
    main()


