import tensorflow as tf
import convnet2 as cv
import numpy as np
import pandas as pd
import time
import threading
from array import array
import math
import random
from tensorflow.python.client import timeline
from ANNA_train import interpolate_sn, read_param_file, read_known_binary, add_to_queue

parameters = read_param_file('ANNA.param')

def unnormalize_parameters(normed_parameters, minvals, maxvals):
    """takes in a list of parameters and undoes simple min/max normalization according to min/max values
    INPUTS
    normed_parameters: length n, containing parameters for a star
    minvals: length n, minimum parameter values
    maxvals: length n, max parameter values
    OUTPUTS
    unnormed_parameters: length n, unnormalized parameters
    """
    unnormed_parameters = normed_parameters*(maxvals-minvals) + minvals
    return unnormed_parameters

def test_neural_network(parameters):
    # import parameter values for normalizing known input parameters and preprocessing spectra
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    # read in the training data
    wavelengths, normed_outputs, pix_values = \
        read_known_binary(parameters['TESTING_DATA'], parameters, minvals, maxvals)
    # set to read in the entire test set when running inference and calculating cost
    test_size = np.size(normed_outputs[:, 0])
    # if preprocessing indicated, then read in appropriate parameters, otherwise set to dummy entries
    if parameters['PREPROCESS_TEST'] == 'YES':
        sn_range = np.fromstring(parameters['SN_RANGE_TEST'], sep=', ', dtype=np.float32)
        y_off_range = np.fromstring(parameters['REL_CONT_E_TEST'], sep=', ', dtype=np.float32)
        sn_template_file = parameters['SN_TEMPLATE_TEST']
        interp_sn = interpolate_sn(sn_template_file, wavelengths)
    else:
        sn_range, y_off_range, interp_sn = 0, 0, 0
        sn_template_file = 'none'
    # control flow for training - load in save location, etc. launch tensorflow session, prepare saver
    model_dir = parameters['MODEL_LOC']
    session = tf.Session()
    # load a previously trained model and corresponding graph. reset devices to enable moving between computers
    loader = tf.train.import_meta_graph(model_dir+'meta', clear_devices=True)
    loader.restore(session, model_dir+'save.ckpt')
    # launch coordinator for input queue
    coordinator = tf.train.Coordinator()
    randomize = False
    # if xval preprocessing desired, flip the correct flat in add_to_queue
    if parameters['PREPROCESS_TEST'] == 'YES':
        test_preprocess = True
    else:
        test_preprocess = False
    num_testthread = int(parameters['TEST_THREADS'])
    # force preprocessing to run on the cpu, utilizing the xval queue from ANNA_train
    with tf.device("/cpu:0"):
        test_threads = [threading.Thread(target=add_to_queue, args=(session, xval_op, coordinator, normed_outputs,
                                                                        pix_values, sn_range, interp_sn, y_off_range,
                                                                        test_size, randomize,
                                                                        xval_training)) for i in range(num_xvthread)]
        for i in test_threads:
            i.start()
    # these are basically all dummy parameters (except dropout)
    feed_dict_test = {learning_rate: 5e-5, dropout: 1.0, batch_size: 10, select_queue: 1}
    test_cost = session.run(cost, feed_dict=feed_dict_test)
    # close the queue and join threads
    coordinator.request_stop()
    session.run(xval_queue.close(cancel_pending_enqueues=True))
    xvcoordinator.join(xval_threads)
    # return cost
    print(round(test_cost, 2))
    session.close()

test_neural_network(parameters)