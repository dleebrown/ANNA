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


def read_param_file(paramfile):
    """reads in a parameter file containing all the info needed to execute tensorflow session (training or inference)
    assumes parameters have correct names and does basic line splitting before dumping them in the output dict
    ignores lines/strings that start with #
    INPUTS
    paramfile: txt file with one parameter per line. format should be param_name: value1, value2
    OUTPUTS
    parameters: dictionary containing all the extracted parameters keyed to the names given in the paramfile
    """
    parameters = {}
    raw_params = open(paramfile, 'r')
    param_lines = raw_params.readlines()
    for entry in param_lines:
        if entry[0] != '#' and entry[0] != '\n':
            tempstring = entry.split(':')
            param_name = str(tempstring[0])
            param_value = str(tempstring[1].strip())
            parameters[param_name] = param_value
    raw_params.close()
    return parameters


def get_set_stats(input_array):
    num_pixels = int(input_array[0])
    num_params = int(input_array[1])
    # each star consists of pixels, parameters, and name, and SN
    star_length = num_pixels+num_params+1+1
    total_array_size = np.size(input_array)
    # trim off the wavelength and npix, nparams lengths
    total_array_size -= (2+num_pixels)
    total_num_stars = int(total_array_size / star_length)
    # throw an error if division doesn't result in integer:
    if total_array_size % star_length != 0:
        print('warning, looks like input array contains extra entries!')
        print('length of star name + num_pixels + num_params = '+str(star_length))
        print('length of array with star names, pixels, and params = '+str(total_array_size))
        print('remainder after dividing the array length by star length = '+str(total_array_size % star_length))
    return total_num_stars, num_pixels, num_params


def normalize_parameters(raw_parameters, minvals, maxvals):
    """takes in a list of parameters and does simple min/max normalization according to min/max values
    INPUTS
    raw_parameters: length n, containing parameters for a star
    minvals: length n, minimum parameter values
    maxvals: length n, max parameter values
    OUTPUTS
    normed_parameters: length n, normalized parameters
    """
    normed_parameters = (raw_parameters - minvals) / (maxvals-minvals)
    return normed_parameters


def read_known_binary(binaryfile, parameters, minvals, maxvals):
    """reads in a binary file of standard form containing spectra with known parameter values and returns np array
    INPUTS
    binaryfile: float64 binary file where first entry is number of pixels, second entry is the number of parameters,
    followed by wavelengths (length = num_pixels), followed by spectra, where each spectrum consists of entries
    star params, star pixels
    OUTPUTS
    wavelengths: float32 1D array containing wavelength values
    known_outputs: float32 2D array with dims [n_stars, n_params] containing parameters for each example
    pixel_values: float32 2D array with dims [n_star, n_pixels] containing flux values for each example
    """
    input_binary = open(binaryfile, 'rb')
    floatarray = array('d')
    floatarray.fromstring(input_binary.read())
    floatarray = np.array(floatarray, dtype=np.float32)
    #  checks to make sure dimensions check out before proceeding and num pix, params match the param file values
    num_pixels = int(floatarray[0])
    if num_pixels != int(parameters['NUM_PX']):
        print('warning, num_pixels in binary file and parameter file do not match!')
        print('num_pixels specified in binary file = '+str(num_pixels))
        print('num_pixels specified in parameter file = '+str(parameters['NUM_PX']))
        return
    num_params = int(floatarray[1])
    if num_params != int(parameters['NUM_OUTS']):
        print('warning, num_params in binary file and parameter file do not match!')
        print('num_params specified in binary file = '+str(num_params))
        print('num_params specified in parameter file = '+str(parameters['NUM_OUTS']))
        return
    # each star consists of name, parameters, (trained), rotv (not trained), and pixels
    star_length = num_pixels+num_params+1+1
    total_array_size = np.size(floatarray)
    # trim off the wavelength and npix, nparams lengths
    total_array_size -= (2+num_pixels)
    # throw an error if division doesn't result in integer:
    if total_array_size % star_length != 0:
        print('warning, looks like input array contains extra entries!')
        print('length of star name + num_pixels + num_params = '+str(star_length))
        print('length of array with star names, pixels, and params = '+str(total_array_size))
        print('remainder after dividing the array length by star length = '+str(total_array_size % star_length))
        return
    num_stars = int(total_array_size / star_length)
    wavelengths = floatarray[2:num_pixels + 2]
    # now slice out the parameter, pixel data and reshape into n rows
    spectra = floatarray[num_pixels+2:]
    spectra = np.reshape(spectra, (num_stars, star_length))
    # further slice into known outputs and pixel values
    known_outputs = spectra[:, 0:num_params+1+1]
    pixel_values = spectra[:, num_params+1+1:]
    for i in range(num_stars):
        known_outputs[i, 1:-1] = normalize_parameters(known_outputs[i, 1:-1], minvals, maxvals)
    return wavelengths, known_outputs, pixel_values


def interpolate_sn(sn_template_file, wavelengths):
    """takes in a SN template with columns wavelength, flux, and interpolates into wavelengths
    INPUTS
    sn_template_file: file containing relative sn template
    wavelengths: wavelengths to interpolate relative sn onto
    OUTPUTS
    interpolated_sn: 1D array size [wavelengths] containing interpolated sn values
    """
    relative_sn = np.genfromtxt(sn_template_file)
    interpolated_sn = np.interp(wavelengths, relative_sn[:, 0], relative_sn[:, 1])
    return interpolated_sn


def preprocess_spectra(fluxes, interpolated_sn, sn_array, y_offset_array):
    """preprocesses a batch of spectra, adding noise according to specified sn profile, and applies continuum error
    INPUTS
    fluxes: length n 2D array with flux values for a spectrum
    interpolated_sn: length n 1D array with relative sn values for each pixel
    sn_array: 2d array dims (num examples, 1) with sn selected for each example
    y_offset_array: same as sn array but with y_offsets
    OUTPUTS
    fluxes: length n 2D array with preprocessed fluxes for a spectrum
    """
    n_pixels = np.size(fluxes[0, :])
    n_stars = np.size(fluxes[:, 1])
    base_stddev = 1.0 / sn_array[:, 0]
    for i in range(n_stars):
        noise_array = np.random.normal(0.0, scale=base_stddev[i], size=n_pixels)
        fluxes[i, :] += noise_array*interpolated_sn
    fluxes += y_offset_array
    return fluxes


def add_to_queue(session, queue_operation, coordinator, normed_outputs, pix_values, sn_range, interp_sn, y_off_range,
                 fetch_size):
    """basically a wrapper function that takes in fluxes and raw params and adds a random preprocessed example to queue
    INPUTS
    session: tensorflow session
    queue operation: operation to queue...
    coordinator: tensorflow thread coordinator
    known_outputs: output from read_known_binary
    pixel_values: output from read_known_binary
    sn_range: sn_range to sample
    y_offset_range: continuum error range to sample
    OUTPUTS
    N/A, this just wraps the preprocessing and enqueue ops into a threadable function
    """
    while not coordinator.should_stop():
            np.random.seed()
            num_stars = np.size(pix_values[:, 0])
            select_star = np.random.randint(0, num_stars-1, size=(fetch_size, 1))
            fluxes = pix_values[select_star[:, 0], :]
            sn_values = np.random.uniform(sn_range[0], sn_range[1], size=(fetch_size, 1))
            y_offsets = np.random.uniform(y_off_range[0], y_off_range[1], size=(fetch_size, 1))
            #  strip off star num, radial velocity info
            known = normed_outputs[select_star[:, 0], 1:-1]
            proc_fluxes = preprocess_spectra(fluxes, interp_sn, sn_values, y_offsets)
            session.run(queue_operation, feed_dict={image_data: proc_fluxes, known_params: known})


parameters = read_param_file('ANNA.param')


with tf.name_scope('NETWORK_HYPERPARAMS'):
    batch_size = int(parameters['BATCH_SIZE1'])
    num_px = int(parameters['NUM_PX'])
    cl1_shape = np.fromstring(parameters['CONV1_SHAPE'], sep=', ', dtype=np.int32)
    drop_prob = float(parameters['KEEP_PROB1'])
    fc1_num_weights = int(parameters['FC1_OUTDIM'])
    fc2_num_weights = int(parameters['FC2_OUTDIM'])
    num_outputs = int(parameters['NUM_OUTS'])
    known_params = tf.placeholder(tf.float32, shape=[None, num_outputs])
    image_data = tf.placeholder(tf.float32, shape=[None, num_px])

# the input queue - handles building random batches and contains wrapper for preprocessing
with tf.name_scope('INPUT_QUEUE'):
    input_queue = tf.FIFOQueue(capacity=200000, dtypes=[tf.float32, tf.float32],
                                            shapes=[[num_outputs], [num_px]])
    queue_operation = input_queue.enqueue_many([known_params, image_data])
    batch_outputs, batch_pixels = input_queue.dequeue_many(batch_size)
    pixels_shaped = tf.reshape(batch_pixels, [batch_size, 1, num_px, 1])
    outputs_shaped = tf.reshape(batch_outputs, [batch_size, num_outputs])


# initialize weights and biases for conv layer 1
with tf.name_scope('CONV1_TENSORS'):
    stdev1 = math.sqrt(2.0 / cl1_shape[1])
    cl1_weight = cv.generate_weights(cl1_shape, type='custom',
                                         function=tf.random_normal(cl1_shape, mean=0.00, stddev=stdev1))
    cl1_bias = cv.generate_bias(length=cl1_shape[3], value=0.00)

# Convolutional layer 1 with sigmoid activation, 1x3 max pooling, batch normalization
with tf.name_scope('CONV1_LAYER'):
    cl1 = cv.layer_conv(input=pixels_shaped, weights=cl1_weight)
    cl1_activate = cv.layer_activate(input=cl1, bias=cl1_bias, type='relu')
# Flatten output of convolutional network to prepare for fully-connected network
with tf.name_scope('Flatten'):
    flat_cl2, num_params = cv.layer_flatten(cl1_activate)

# initialize weights and biases for fully connected layer 1
with tf.name_scope('FC1_TENSORS'):
    fc1_shape = [num_params, fc1_num_weights]
    stdev3 = math.sqrt(2.0 / (num_px * cl1_shape[3]))
    fc1_weight = cv.generate_weights(fc1_shape, type='custom',
                                         function=tf.random_normal(fc1_shape, mean=0.00, stddev=stdev3))
    fc1_bias = cv.generate_bias(fc1_num_weights, value=0.00)

# fully connected layer 1 with sigmoid activation
with tf.name_scope('FC1_LAYER'):
    fc1 = cv.layer_fc(input=flat_cl2, weights=fc1_weight)
    fc1_activate = cv.layer_activate(input=fc1, bias=fc1_bias, type='relu')
    fc1_dropout = cv.layer_dropout(input=fc1_activate, keep_probability=drop_prob)

# initialize weights for fully connected layer 2
with tf.name_scope('FC2_TENSORS'):
    fc2_shape = [fc1_num_weights, fc2_num_weights]
    stdev4 = math.sqrt(2.0 / fc1_num_weights)
    fc2_weight = cv.generate_weights(fc2_shape, type='custom',
                                         function=tf.random_normal(fc2_shape, mean=0.00, stddev=stdev4))
    fc2_bias = cv.generate_bias(fc2_num_weights, value=0.00)

# fully connected layer 2 with sigmoid activation
with tf.name_scope('FC2_LAYER'):
    fc2 = cv.layer_fc(input=fc1_dropout, weights=fc2_weight)
    fc2_activate = cv.layer_activate(input=fc2, bias=fc2_bias, type='relu')
    fc2_dropout = cv.layer_dropout(input=fc2_activate, keep_probability=drop_prob)

# initialize weights for output layer of network
with tf.name_scope('FC3_TENSORS'):
    fc3_shape = [fc2_num_weights, num_outputs]
    stdev5 = math.sqrt(2.0 / fc2_num_weights)
    fc3_weight = cv.generate_weights(fc3_shape, type='custom',
                                         function=tf.random_normal(fc3_shape, mean=0.00, stddev=stdev5))

# compute output of the network using fully connected layer 2 with no activation function or biases
with tf.name_scope('FC3_LAYER'):
    fc_output = cv.layer_fc(input=fc2_dropout, weights=fc3_weight)

# cost function for the network
with tf.name_scope('COST'):
    cost = cv.network_cost(predicted_vals=fc_output, known_vals=outputs_shaped, type='l2')

# optimization function
with tf.name_scope('OPTIMIZE'):
    optimize = cv.network_optimize(cost, learn_rate=5e-5, optimizer='adam')


def train_neural_network(parameters):
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    sn_range = np.fromstring(parameters['SN_RANGE_TRAIN'], sep=', ', dtype=np.float32)
    y_off_range = np.fromstring(parameters['REL_CONT_E_TRAIN'], sep=', ', dtype=np.float32)
    sn_template_file = parameters['SN_TEMPLATE']
    fetch_size = int(parameters['NUM_FETCH'])
    wavelengths, normed_outputs, pix_values = read_known_binary('allspec_test_200_600', parameters, minvals, maxvals)
    interp_sn = interpolate_sn(sn_template_file, wavelengths)
    begin_time = time.time()

    session = tf.Session()

    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()

    session.run(tf.global_variables_initializer())
    coordinator = tf.train.Coordinator()

    with tf.device("/cpu:0"):
        num_threads = 1
        enqueue_threads = [threading.Thread(target=add_to_queue, args=(session, queue_operation, coordinator,
                                                                 normed_outputs, pix_values, sn_range, interp_sn,
                                                                 y_off_range, fetch_size)) for i in range(num_threads)]
        for i in enqueue_threads:
            i.start()
    time.sleep(1)
    for i in range(int(parameters['NUM_TRAIN_ITERS1'])):
        if i % 1000 == 0:

            session.run(optimize, options=options, run_metadata=run_metadata)
            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_01.json', 'w') as f:
                f.write(chrome_trace)

            #session.run(optimize)
            print('done with batch ' + str(int(i)) + '/' + str(int(parameters['NUM_TRAIN_ITERS1'])))
            test_cost = session.run(cost)
            print('current cost: ' + str(round(test_cost, 2)))
        elif i % 100 == 0:
            session.run(optimize)
            # Create the Timeline object, and write it to a json file
            print('done with batch '+str(int(i))+'/'+str(int(parameters['NUM_TRAIN_ITERS1'])))
            test_cost = session.run(cost)
            print('current cost: '+str(round(test_cost, 2)))
        else:
            session.run(optimize)
    coordinator.request_stop()
    coordinator.join(enqueue_threads)
    end_time = time.time()
    print('execute time: '+str(round(end_time-begin_time, 2)))

train_neural_network(parameters)