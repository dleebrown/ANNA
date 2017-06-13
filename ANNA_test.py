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
from ANNA_train import interpolate_sn, read_param_file, read_known_binary, preprocess_spectra

parameters = read_param_file('ANNA.param')

def load_frozen_model(parameters):
    model_dir = parameters['MODEL_LOC']
    frozen_dir = model_dir+'frozen.pb'
    with open(frozen_dir, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, input_map=None, return_elements=None, name='prefix', op_dict=None, producer_op_list=None)
    return graph

def add_to_queue(session, queue_operation, coordinator, normed_outputs, pix_values, sn_range, interp_sn, y_off_range,
                 fetch_size, randomize, preprocess):
    """basically a wrapper function that takes in fluxes and raw params and adds a random preprocessed example to queue
    INPUTS
    session: tensorflow session
    queue operation: operation to queue...
    coordinator: tensorflow thread coordinator
    known_outputs: output from read_known_binary
    pixel_values: output from read_known_binary
    sn_range: sn_range to sample
    y_offset_range: continuum error range to sample
    randomize: randomly draw fetch_size examples if true, if not true then draw the first fetch_size examples
    OUTPUTS
    N/A, this just wraps the preprocessing and enqueue ops into a threadable function
    """
    while not coordinator.should_stop():
            np.random.seed()
            num_stars = np.size(pix_values[:, 0])
            if randomize:
                select_star = np.random.randint(0, num_stars-1, size=(fetch_size, 1))
            if not randomize:
                select_star = np.arange(0, fetch_size)
                select_star = np.reshape(select_star, (fetch_size, 1))
            fluxes = pix_values[select_star[:, 0], :]
            if preprocess:
                sn_values = np.random.uniform(sn_range[0], sn_range[1], size=(fetch_size, 1))
                y_offsets = np.random.uniform(y_off_range[0], y_off_range[1], size=(fetch_size, 1))
                #  strip off star num, radial velocity info
                known = normed_outputs[select_star[:, 0], 1:-1]
                proc_fluxes = preprocess_spectra(fluxes, interp_sn, sn_values, y_offsets)
            else:
                known = normed_outputs[select_star[:, 0], 1:-1]
                proc_fluxes = fluxes
            try:
                session.run(queue_operation, feed_dict={image_data: proc_fluxes, known_params: known})
            except tf.errors.CancelledError:
                if randomize:
                    print('Input queue closed, exiting training')

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

def test_fancy_frozen(parameters):
    # import parameter values for normalizing known input parameters and preprocessing spectra
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    num_outputs = int(parameters['NUM_OUTS'])
    num_px = int(parameters['NUM_PX'])
    # read in the training data
    wavelengths, normed_outputs, pix_values = \
        read_known_binary(parameters['TESTING_DATA'], parameters, minvals, maxvals)
    print(np.shape(pix_values))
    normed_outputs = normed_outputs[:, 1:-1]
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
    with tf.Graph().as_default() as graph:
        input_px = tf.placeholder(tf.float32, shape=[None, num_px], name='input_px')
        input_known = tf.placeholder(tf.float32, shape=[None, num_outputs], name='input_known')
        model_dir = parameters['MODEL_LOC']
        frozen_dir = model_dir + 'frozen.pb'
        with open(frozen_dir, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, input_map={"MASTER_QUEUE/testpx:0": input_px, "MASTER_QUEUE/testoutput:0":input_known}, name='blabla')
    for op in graph.get_operations():
        print(op.name)
    cost = graph.get_tensor_by_name('blabla/COST/testcost:0')
    dropout = graph.get_tensor_by_name('blabla/NETWORK_HYPERPARAMS/dropout:0')
    batch_size = graph.get_tensor_by_name('blabla/NETWORK_HYPERPARAMS/batch_size:0')
    with tf.Session(graph=graph) as sess:
        testpleasework = sess.run(cost, feed_dict={input_px: pix_values, input_known: normed_outputs, dropout:1.0, batch_size:test_size})
        print(testpleasework)







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
    # load into a new graph to avoid tflow op duplication issues due to importing train function
    session = tf.Session()
    # load a previously trained model and corresponding graph. reset devices to enable moving between computers
    loader = tf.train.import_meta_graph(model_dir+'meta', clear_devices=True)
    loader.restore(session, model_dir+'save.ckpt')
    # restore the necessary placeholder variables and operations
    global image_data
    image_data = tf.get_collection('image_data')[0]
    global known_params
    known_params = tf.get_collection('known_params')[0]
    cost = tf.get_collection('cost')[0]
    xval_op = tf.get_collection('xval_op')[0]
    xval_queue=tf.get_collection('xval_queue')[0]
    learning_rate = tf.get_collection('learning_rate')[0]
    dropout = tf.get_collection('dropout')[0]
    batch_size = tf.get_collection('batch_size')[0]
    select_queue = tf.get_collection('select_queue')[0]
    pixels_shaped = tf.get_collection('pixels_shaped')[0]
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
                                                                        test_preprocess)) for i in range(num_testthread)]
        for i in test_threads:
            i.start()
    print('hello')
    # these are basically all dummy parameters (except dropout)
    feed_dict_test = {learning_rate: 5e-5, dropout: 1.0, batch_size: 1, select_queue: 1}
    print('hello')
    coordinator.request_stop()
    print('hello')
    session.run(xval_queue.close(cancel_pending_enqueues=True))
    coordinator.join(test_threads)
    test_cost = session.run(xval_op, feed_dict=feed_dict_test)
    print('hello')

    test_cost = session.run(learning_rate, feed_dict=feed_dict_test)
    print('hello')
    # close the queue and join threads
    

    # return cost
    print(round(test_cost, 2))
    session.close()

#test_neural_network(parameters)
test_fancy_frozen(parameters)