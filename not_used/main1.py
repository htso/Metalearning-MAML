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

from data_generator1 import DataGenerator
from maml1 import MAML
from tensorflow.python.platform import flags






def train(model, saver, sess, exp_string, data_generator, resume_itr, meta_batch_size, update_batch_size, \
            pretrain_iterations, metatrain_iterations, baseline, num_updates, logdir):
    SUMMARY_INTERVAL = 100
    SAVE_INTERVAL = 1000
    PRINT_INTERVAL = 1000
    TEST_PRINT_INTERVAL = PRINT_INTERVAL*5
    
    train_writer = tf.summary.FileWriter(logdir + '/' + exp_string, sess.graph)
    print('Done initializing, starting training.')
    prelosses, postlosses = [], []

    num_classes = data_generator.num_classes # for classification, 1 otherwise
    multitask_weights, reg_weights = [], []

    for itr in range(resume_itr, pretrain_iterations + metatrain_iterations):
        feed_dict = {}
        if 'generate' in dir(data_generator):
            batch_x, batch_y, amp, phase = data_generator.generate()

            # oracle : just add two extra columns to X, and then use it as input to model,
            # i thought this won't give the model much hint, so ineffective as an aid.
            if baseline == 'oracle':
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                for i in range(meta_batch_size):
                    batch_x[i, :, 1] = amp[i]
                    batch_x[i, :, 2] = phase[i]

            # a : training data
            inputa = batch_x[:, :update_batch_size, :]
            labela = batch_y[:, :update_batch_size, :]
            # b: testing data, this is the last row in a batch
            inputb = batch_x[:, update_batch_size:, :] 
            labelb = batch_y[:, update_batch_size:, :]

            feed_dict = {model.inputa:inputa, model.inputb:inputb,  model.labela:labela, model.labelb:labelb}

        if itr < pretrain_iterations:
            input_tensors = [model.pretrain_op]
        else:
            # ==================================
            input_tensors = [model.metatrain_op] 
            # ==================================

        if (itr % SUMMARY_INTERVAL == 0 or itr % PRINT_INTERVAL == 0):
            input_tensors.extend([model.summ_op, model.total_loss1, model.total_losses2[num_updates-1]])
            if model.classification:
                input_tensors.extend([model.total_accuracy1, model.total_accuracies2[num_updates-1]])

        # =========================================
        result = sess.run(input_tensors, feed_dict) 
        # =========================================

        if itr % SUMMARY_INTERVAL == 0:
            prelosses.append(result[-2])
            train_writer.add_summary(result[1], itr)
            postlosses.append(result[-1])

        if (itr!=0) and itr % PRINT_INTERVAL == 0:
            if itr < pretrain_iterations:
                print_str = 'Pretrain Iteration ' + str(itr)
            else:
                print_str = 'Iteration ' + str(itr - pretrain_iterations)
            print_str += ': ' + str(np.mean(prelosses)) + ', ' + str(np.mean(postlosses))
            print(print_str)
            prelosses, postlosses = [], []

        if (itr!=0) and itr % SAVE_INTERVAL == 0:
            saver.save(sess, logdir + '/' + exp_string + '/model' + str(itr))

    saver.save(sess, logdir + '/' + exp_string +  '/model' + str(itr))


def test(model, saver, sess, exp_string, data_generator, test_num_updates=None):
    num_classes = data_generator.num_classes # for classification, 1 otherwise

    np.random.seed(1)
    random.seed(1)

    metaval_accuracies = []

    for _ in range(NUM_TEST_POINTS):
        if 'generate' not in dir(data_generator):
            feed_dict = {}
            feed_dict = {model.meta_lr : 0.0}
        else:
            batch_x, batch_y, amp, phase = data_generator.generate(train=False)

            if FLAGS.baseline == 'oracle': # NOTE - this flag is specific to sinusoid
                batch_x = np.concatenate([batch_x, np.zeros([batch_x.shape[0], batch_x.shape[1], 2])], 2)
                batch_x[0, :, 1] = amp[0]
                batch_x[0, :, 2] = phase[0]

            inputa = batch_x[:, :num_classes*FLAGS.update_batch_size, :]
            inputb = batch_x[:,num_classes*FLAGS.update_batch_size:, :]
            labela = batch_y[:, :num_classes*FLAGS.update_batch_size, :]
            labelb = batch_y[:,num_classes*FLAGS.update_batch_size:, :]

            feed_dict = {model.inputa: inputa, model.inputb: inputb,  model.labela: labela, model.labelb: labelb, model.meta_lr: 0.0}

        if model.classification:
            result = sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)
        else:  # this is for sinusoid
            result = sess.run([model.total_loss1] +  model.total_losses2, feed_dict)
        metaval_accuracies.append(result)

    metaval_accuracies = np.array(metaval_accuracies)
    means = np.mean(metaval_accuracies, 0)
    stds = np.std(metaval_accuracies, 0)
    ci95 = 1.96*stds/np.sqrt(NUM_TEST_POINTS)

    print('Mean validation accuracy/loss, stddev, and confidence intervals')
    print((means, stds, ci95))

    out_filename = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.csv'
    out_pkl = FLAGS.logdir +'/'+ exp_string + '/' + 'test_ubs' + str(FLAGS.update_batch_size) + '_stepsize' + str(FLAGS.update_lr) + '.pkl'
    with open(out_pkl, 'wb') as f:
        pickle.dump({'mses': metaval_accuracies}, f)
    with open(out_filename, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['update'+str(i) for i in range(len(means))])
        writer.writerow(means)
        writer.writerow(stds)
        writer.writerow(ci95)

def main():
    datasource = 'sinusoid'
    num_classes = 1
    baseline = None
    pretrain_iterations = 0
    metatrain_iterations = 70000
    meta_batch_size = 25
    meta_lr = 0.001
    update_batch_size = 10
    update_lr = 1e-3
    num_updates = 1
    norm = None
    stop_grad = False
    log_flag = True
    logdir = 'logs/sine'
    resume = True
    train_flag = True
    test_iter = -1
    test_set = False
    train_update_batch_size = -1
    train_update_lr = -1

    if train_flag:
        test_num_updates = 5
    else:
        test_num_updates = 10

    if train_flag == False:
        orig_meta_batch_size = 25
        # always use meta batch size of 1 when testing.
        meta_batch_size = 1

    # ==================================================================
    data_generator = DataGenerator(update_batch_size*2, 25)
    # ==================================================================
    
    dim_output = data_generator.dim_output
    if baseline == 'oracle':
        assert datasource == 'sinusoid'
        dim_input = 3
        pretrain_iterations += metatrain_iterations
        metatrain_iterations = 0
    else:
        dim_input = data_generator.dim_input

   
    tf_data_load = False
    input_tensors = None

    model = MAML(dim_input, dim_output, test_num_updates=test_num_updates)
    if train_flag or not tf_data_load:
        # =====================================================================
        model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        # =====================================================================
    if tf_data_load:
        model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
    model.summ_op = tf.summary.merge_all()

    saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

    sess = tf.InteractiveSession()

    if train_flag == False:
        # change to original meta batch size when loading model.
        meta_batch_size = orig_meta_batch_size

    if train_update_batch_size == -1:
        train_update_batch_size = update_batch_size
    if train_update_lr == -1:
        train_update_lr = update_lr

    exp_string = 'cls_'+str(num_classes)+'.mbs_'+str(meta_batch_size) + '.ubs_' + str(train_update_batch_size) + \
           '.numstep' + str(num_updates) + '.updatelr' + str(train_update_lr)

    
    if stop_grad:
        exp_string += 'stopgrad'
    if baseline:
        exp_string += baseline
    if norm == 'batch_norm':
        exp_string += 'batchnorm'
    elif norm == 'layer_norm':
        exp_string += 'layernorm'
    elif norm == 'None':
        exp_string += 'nonorm'
    else:
        print('Norm setting not recognized.')

    resume_itr = 0
    model_file = None

    tf.global_variables_initializer().run()
    tf.train.start_queue_runners()

    if resume or not train_flag:
        model_file = tf.train.latest_checkpoint(logdir + '/' + exp_string)
        if test_iter > 0:
            model_file = model_file[:model_file.index('model')] + 'model' + str(test_iter)
        if model_file:
            ind1 = model_file.index('model')
            resume_itr = int(model_file[ind1+5:])
            print("Restoring model weights from " + model_file)
            saver.restore(sess, model_file)

    if train_flag:
        train(model, saver, sess, exp_string, data_generator, resume_itr, meta_batch_size, update_batch_size, \
            pretrain_iterations, metatrain_iterations, baseline, num_updates, logdir)
    else:
        test(model, saver, sess, exp_string, data_generator, test_num_updates)

if __name__ == "__main__":
    main()
