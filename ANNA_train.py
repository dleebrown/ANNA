import tensorflow as tf
import convnet2 as cv
import numpy as np
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
        if entry[0] != '#' and entry[0] != '\n' and entry[0] != '\r':
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
    star ID, star params, dummy param, star pixels
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
    # each star consists of name, parameters, (trained), dummy parameter (not trained), and pixels
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
                 fetch_size, randomize, preprocess, queuetype):
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
                #  strip off star num, dummy param info
                known = normed_outputs[select_star[:, 0], 1:-1]
                proc_fluxes = preprocess_spectra(fluxes, interp_sn, sn_values, y_offsets)
            else:
                known = normed_outputs[select_star[:, 0], 1:-1]
                proc_fluxes = fluxes
            if queuetype == 'train_q':
                try:
                    session.run(queue_operation, feed_dict={image_data: proc_fluxes, known_params: known})
                except tf.errors.CancelledError:
                    if randomize:
                        print('Input queue closed, exiting training')
            if queuetype == 'xval_q':
                xval_enqueued = session.run(xval_queue.size(), feed_dict={xval_data: proc_fluxes, xval_params: known})
                if xval_enqueued <= int(parameters['MAX_PP_DEPTH']) - int(parameters['XVAL_SIZE']):
                    try:
                        session.run(queue_operation, feed_dict={xval_data: proc_fluxes, xval_params: known},
                                    options=tf.RunOptions(timeout_in_ms=5000))
                    except tf.errors.DeadlineExceededError:
                        sizes = session.run(xval_queue.size(), feed_dict={xval_data: proc_fluxes, xval_params: known})
                        print('Cross-validation enqueue error, current queue size: '+str(sizes))
                    except tf.errors.CancelledError:
                        if randomize:
                            print('Input queue closed, exiting training')


# freezes a tensorflow model to be reloaded later
def freeze_model(parameters):
    model_dir = parameters['SAVE_LOC']
    # the op to save - this results in all the tboard options being discarded from the frozen graph
    output_op = 'COST/cost'
    frozen_dir = model_dir+'frozen.model'
    saver = tf.train.Saver()
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    session = tf.Session()
    # restore the trained model
    saver.restore(session, model_dir+'save.ckpt')
    # convert all useful variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(session, input_graph_def, [output_op])
    # open the specified file and write the model
    with open(frozen_dir, 'wb') as frozen_model:
        frozen_model.write(output_graph_def.SerializeToString())
    print('Model frozen as '+frozen_dir)

# read in the parameter file
parameters = read_param_file('ANNA.param')

# the network architecture, only called if this file is run as main
if __name__ == '__main__':
    with tf.name_scope('NETWORK_HYPERPARAMS'):
        batch_size = tf.placeholder(tf.int32, name='batch_size')
        num_px = int(parameters['NUM_PX'])
        cl1_shape = np.fromstring(parameters['CONV1_SHAPE'], sep=', ', dtype=np.int32)
        fc1_num_weights = int(parameters['FC1_OUTDIM'])
        fc2_num_weights = int(parameters['FC2_OUTDIM'])
        num_outputs = int(parameters['NUM_OUTS'])
        known_params = tf.placeholder(tf.float32, shape=[None, num_outputs], name='known_params')
        image_data = tf.placeholder(tf.float32, shape=[None, num_px], name='image_data')
        xval_params = tf.placeholder(tf.float32, shape=[None, num_outputs], name='xval_params')
        xval_data = tf.placeholder(tf.float32, shape=[None, num_px], name='xval_data')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        dropout = tf.placeholder(tf.float32, name='dropout')
        pp_depth = int(parameters['MAX_PP_DEPTH'])
        # adds various variables to collection in order to recall later
        tf.add_to_collection('learning_rate', learning_rate)
        tf.add_to_collection('dropout', dropout)
        tf.add_to_collection('batch_size', batch_size)
        tf.add_to_collection('image_data', image_data)
        tf.add_to_collection('known_params', known_params)

    # the input queue - handles building random batches and contains wrapper for preprocessing
    with tf.name_scope('INPUT_QUEUE'):
        input_queue = tf.FIFOQueue(capacity=pp_depth, dtypes=[tf.float32, tf.float32],
                                                shapes=[[num_outputs], [num_px]])
        queue_op = input_queue.enqueue_many([known_params, image_data])

    # adding a second queue to handle xval data - this is just unused if xval option is unused
    with tf.name_scope('XVAL_QUEUE'):
        xval_queue = tf.FIFOQueue(capacity=pp_depth, dtypes=[tf.float32, tf.float32],
                                  shapes=[[num_outputs], [num_px]])
        xval_op = xval_queue.enqueue_many([xval_params, xval_data])
        tf.add_to_collection('xval_op', xval_op)
        tf.add_to_collection('xval_queue', xval_queue)

    with tf.name_scope('MASTER_QUEUE'):
        select_queue = tf.placeholder(tf.int32, [])
        master_queue = tf.QueueBase.from_list(select_queue, [input_queue, xval_queue])
        batch_outputs, batch_pixels = master_queue.dequeue_many(batch_size)
        # add some redundant identity ops to add in naming (convnet2 doesn't support names)
        batch_pixels = tf.identity(batch_pixels, name='px')
        batch_outputs = tf.identity(batch_outputs, name='output')
        pixels_shaped = tf.reshape(batch_pixels, [batch_size, 1, num_px, 1])
        outputs_shaped = tf.reshape(batch_outputs, [batch_size, num_outputs])
        # add more variables to collection
        tf.add_to_collection('select_queue', select_queue)
        tf.add_to_collection('pixels_shaped', pixels_shaped)

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
        fc1 = cv.layer_fc(input=flat_cl2, weights=fc1_weight, name='test2')
        fc1_activate = cv.layer_activate(input=fc1, bias=fc1_bias, type='relu')
        fc1_dropout = cv.layer_dropout(input=fc1_activate, keep_probability=dropout)

    # initialize weights for fully connected layer 2
    with tf.name_scope('FC2_TENSORS'):
        fc2_shape = [fc1_num_weights, fc2_num_weights]
        stdev4 = math.sqrt(2.0 / fc1_num_weights)
        fc2_weight = cv.generate_weights(fc2_shape, type='custom',
                                             function=tf.random_normal(fc2_shape, mean=0.00, stddev=stdev4))
        fc2_bias = cv.generate_bias(fc2_num_weights, value=0.00)

    # fully connected layer 2 with sigmoid activation
    with tf.name_scope('FC2_LAYER'):
        fc2 = cv.layer_fc(input=fc1_dropout, weights=fc2_weight, name='test')
        fc2_activate = cv.layer_activate(input=fc2, bias=fc2_bias, type='relu')
        fc2_dropout = cv.layer_dropout(input=fc2_activate, keep_probability=dropout)

    # initialize weights for output layer of network
    with tf.name_scope('FC3_TENSORS'):
        fc3_shape = [fc2_num_weights, num_outputs]
        stdev5 = math.sqrt(2.0 / fc2_num_weights)
        fc3_weight = cv.generate_weights(fc3_shape, type='custom',
                                             function=tf.random_normal(fc3_shape, mean=0.00, stddev=stdev5))

    # compute output of the network using fully connected layer 2 with no activation function or biases
    with tf.name_scope('FC3_LAYER'):
        fc_output = cv.layer_fc(input=fc2_dropout, weights=fc3_weight, name='fc_output')
        # add this to collection for inference
        tf.add_to_collection('fc_output', fc_output)

    # cost function for the network
    with tf.name_scope('COST'):
        cost = cv.network_cost(predicted_vals=fc_output, known_vals=outputs_shaped, type='l2')
        # again use identity op to name cost
        cost = tf.identity(cost, name='cost')
        # add to tboard and collection
        batch_cost_sum = tf.summary.scalar("BATCH COST", cost)
        xval_cost_sum = tf.summary.scalar("XVAL COST", cost)
        tf.add_to_collection('cost', cost)

    with tf.name_scope('OPTIMIZE'):
        optimize = cv.network_optimize(cost, learn_rate=learning_rate, optimizer='adam')

    with tf.name_scope('VISUALS'):
        # add in various histograms to the summary for tboard output
        cv1w_sum = tf.summary.histogram('CONV1_WEIGHTS', cl1_weight)
        cv1b_sum = tf.summary.histogram('CONV1_BIASES', cl1_bias)
        cv1a_sum = tf.summary.histogram('CONV1_ACTIVATIONS', cl1_activate)
        fc1w_sum = tf.summary.histogram('FC1_WEIGHTS', fc1_weight)
        fc1b_sum = tf.summary.histogram('FC1_BIASES', fc1_bias)
        fc1a_sum = tf.summary.histogram('FC1_ACTIVATIONS', fc1_activate)
        fc2w_sum = tf.summary.histogram('FC2_WEIGHTS', fc2_weight)
        fc2b_sum = tf.summary.histogram('FC2_BIASES', fc2_bias)
        fc2a_sum = tf.summary.histogram('FC2_ACTIVATIONS', fc2_activate)
        fc3w_sum = tf.summary.histogram('FC3_WEIGHTS', fc3_weight)
        fc3a_sum = tf.summary.histogram('FC3_ACTIVATIONS', fc_output)


def train_neural_network(parameters):
    # import parameter values for normalizing known input parameters and preprocessing spectra
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    # read in the training data
    wavelengths, normed_outputs, pix_values = \
        read_known_binary(parameters['TRAINING_DATA'], parameters, minvals, maxvals)
    if parameters['PREPROCESS_TRAIN'] == 'YES':
        sn_range = np.fromstring(parameters['SN_RANGE_TRAIN'], sep=', ', dtype=np.float32)
        y_off_range = np.fromstring(parameters['REL_CONT_E_TRAIN'], sep=', ', dtype=np.float32)
        sn_template_file = parameters['SN_TEMPLATE']
        interp_sn = interpolate_sn(sn_template_file, wavelengths)
    else:
        sn_range, y_off_range, interp_sn = 0, 0, 0
    fetch_size = int(parameters['NUM_FETCH'])
    bsize_train1 = int(parameters['BATCH_SIZE1'])
    # if separate xval set specified, read that in
    if parameters['TRAINING_XVAL'] == 'YES':
        xv_wave, xv_norm_out, xv_px_val = read_known_binary(parameters['XVAL_DATA'], parameters, minvals, maxvals)
        xv_size = int(parameters['XVAL_SIZE'])
        if parameters['PREPROCESS_XVAL'] == 'YES':
            # this could reload parameters, but allows for case of not preprocessing training but preprocess xval
            sn_template_file = parameters['SN_TEMPLATE']
            interp_sn = interpolate_sn(sn_template_file, wavelengths)
            sn_range = np.fromstring(parameters['SN_RANGE_TRAIN'], sep=', ', dtype=np.float32)
            y_off_range = np.fromstring(parameters['REL_CONT_E_TRAIN'], sep=', ', dtype=np.float32)

    # subloop definition to build a separate queue for xval data if it exists and calculate the cost
    def xval_subloop(learn_rate, bsize, step, inherit_iter_count):
        queuetype = 'xval_q'
        xvcoordinator = tf.train.Coordinator()
        randomize = False
        # if xval preprocessing desired, flip the correct flat in add_to_queue
        if parameters['PREPROCESS_XVAL'] == 'YES':
            xval_training = True
        else:
            xval_training = False
        num_xvthread = int(parameters['XV_THREADS'])
        # force preprocessing to run on the cpu - this is actually not optimal since the threads will be respawned and
        # every time xval subloop run but for smaller xval sizes this shouldn't matter much
        with tf.device("/cpu:0"):
            xval_threads = [threading.Thread(target=add_to_queue, args=(session, xval_op, xvcoordinator, xv_norm_out,
                                                                        xv_px_val, sn_range, interp_sn, y_off_range,
                                                                        xv_size, randomize, xval_training,
                                                                        queuetype)) for i in range(num_xvthread)]
            for i in xval_threads:
                i.start()
        feed_dict_xval = {learning_rate: learn_rate, dropout: 1.0, batch_size: bsize, select_queue: 1}
        xval_cost, xval_sum = session.run([cost, xval_cost_sum], feed_dict=feed_dict_xval)
        if parameters['TBOARD_OUTPUT'] == 'YES':
            writer.add_summary(xval_sum, step + inherit_iter_count)
        # close the queue and join threads
        xvcoordinator.request_stop()
        xvcoordinator.join(xval_threads)
        # return cost
        return round(xval_cost, 2)

    # definition of the training loop in order to allow for multistage training
    def train_loop(iterations, learn_rate, keep_prob, bsize, inherit_iter_count):
        begin_time = time.time()
        coordinator = tf.train.Coordinator()
        # will always randomize batch selection for training
        randomize = True
        # if training preprocessing desired, flips appropriate flag
        if parameters['PREPROCESS_TRAIN'] == 'YES':
            pp_training = True
        else:
            pp_training = False
        # force preprocessing to run on the cpu
        with tf.device("/cpu:0"):
            num_threads = int(parameters['PP_THREADS'])
            queuetype = 'train_q'
            enqueue_threads = [threading.Thread(target=add_to_queue, args=(session, queue_op, coordinator,
                                                                           normed_outputs, pix_values, sn_range,
                                                                           interp_sn, y_off_range, fetch_size,
                                                                           randomize, pp_training,
                                                                           queuetype)) for i in range(num_threads)]
            for i in enqueue_threads:
                i.start()
        # delay running training by 1 second in order to prefill the queue
        #time.sleep(1)
        feed_dict_train = {learning_rate: learn_rate, dropout: keep_prob, batch_size: bsize, select_queue: 0}
        # controls early stopping threshold
        early_stop_counter = 0
        # stores completed iterations to pass to second stage of training for tensorboard visualization
        completed_iterations = 0
        # stores best cost in order to control early stopping
        best_cost = 0.0
        # fetches early stop threshold if early stopping is enabled, otherwise just sets early stop iters to total iters
        if parameters['EARLY_STOP'] == 'YES':
            early_stop_threshold = int(parameters['ES_SAMPLE'])
        else:
            early_stop_threshold = iterations
        # main training loop
        for i in range(iterations):
            # only continues if early stop threshold has not been met
            if early_stop_counter <= early_stop_threshold:
                completed_iterations += 1
                # first iteration will store the cost under best_cost for xval if xval specified, current batch if not
                if i == 0:
                    if parameters['TRAINING_XVAL'] == 'YES':
                        init_cost = xval_subloop(learn_rate, xv_size, i, inherit_iter_count)
                        best_cost = init_cost
                        print('Initial xval cost: '+str(round(init_cost, 2)))
                        session.run(optimize, feed_dict=feed_dict_train)
                    else:
                        init_cost = session.run(cost, feed_dict=feed_dict_train)
                        best_cost = init_cost
                        print('Initial batch cost: '+str(round(init_cost, 2)))
                        session.run(optimize, feed_dict=feed_dict_train)
                # if not first iteration but iteration corresponding to sample interval, runs diagnostics
                elif (i+1) % int(parameters['SAMPLE_INTERVAL']) == 0:
                    test_cost = session.run(cost, feed_dict=feed_dict_train)
                    session.run(optimize, feed_dict=feed_dict_train)
                    if parameters['TRAINING_XVAL'] == 'YES':
                        xvcost = xval_subloop(learn_rate, xv_size, i, inherit_iter_count)
                        print('done with batch '+str(int(i+1))+'/'+str(iterations)+', current cost: '
                          + str(round(test_cost, 2))+', xval cost: '+str(xvcost))
                    else:
                        # if xval set not specified, will just calculate cost for current batch and print
                        print('done with batch ' + str(int(i + 1)) + '/' + str(iterations) + ', current cost: '
                          + str(round(test_cost, 2)))
                    if parameters['EARLY_STOP'] == 'YES':
                        # if early stopping desired, will compare xval or current batch cost to the best cost
                        if parameters['TRAINING_XVAL'] == 'YES':
                            if float(xvcost) >= best_cost:
                                early_stop_counter += 1
                            else:
                                # reset early stopping counter if current cost is better than previous best cost
                                best_cost = xvcost
                                early_stop_counter = 0
                        else:
                            if float(test_cost) >= best_cost:
                                early_stop_counter += 1
                            else:
                                best_cost = test_cost
                                early_stop_counter = 0
                    if parameters['TBOARD_OUTPUT'] == 'YES':
                        # if tensorboard logging enabled, stores visualization data
                        outlog = session.run(merged_summaries, feed_dict=feed_dict_train)
                        writer.add_summary(outlog, i+1+inherit_iter_count)
                else:  # just runs optimize if none of the above criteria are met
                    session.run(optimize, feed_dict=feed_dict_train)
            if early_stop_counter == early_stop_threshold or (i == (iterations-1)
                                                              and early_stop_counter < (early_stop_threshold+1)):
                # if end of training reached, print a message and optionally save timeline
                if parameters['TIMELINE_OUTPUT'] == 'YES':
                    # if timeline desired, prints to json file for most recent iteration
                    fetched_timeline = timeline.Timeline(run_metadata.step_stats)
                    chrome_trace = fetched_timeline.generate_chrome_trace_format()
                    with open(parameters['LOG_LOC'] + 'timeline_01.json', 'w') as f:
                        f.write(chrome_trace)
                    print('Timeline saved as timeline_01.json in folder ' + parameters['LOG_LOC'])
                print('Early stop threshold or specified iterations met')
                early_stop_counter += 1
        # close the preprocessing queues and join threads
        coordinator.request_stop()
        session.run(input_queue.close(cancel_pending_enqueues=True))
        if parameters['TRAINING_XVAL'] == 'YES':
            session.run(xval_queue.close(cancel_pending_enqueues=True))
        coordinator.join(enqueue_threads)
        end_time = time.time()
        # return the run time and total completed iterations
        return str(round(end_time-begin_time, 2)), completed_iterations

    # control flow for training - load in save location, etc. launch tensorflow session, prepare saver
    model_dir = parameters['SAVE_LOC']
    session = tf.Session(config=tf.ConfigProto())
    options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
    session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # if tboard output desired start a writer
    if parameters['TBOARD_OUTPUT'] == 'YES':
        writer = tf.summary.FileWriter(parameters['LOG_LOC']+'logs', session.graph)
        merged_summaries = tf.summary.merge([cv1w_sum, cv1b_sum, cv1a_sum, fc1w_sum, fc1b_sum, fc1a_sum, fc2w_sum,
                                             fc2b_sum, fc2a_sum, fc3w_sum, fc3a_sum, batch_cost_sum])
    # train the network on input training data
    execute_time, finished_iters = train_loop(int(parameters['NUM_TRAIN_ITERS1']), float(parameters['LEARN_RATE1']),
                                              float(parameters['KEEP_PROB1']), bsize_train1, inherit_iter_count=0)
    # save model and the graph and close session
    save_path = saver.save(session, model_dir+'save.ckpt')
    session.close()
    print('Training stage 1 finished in '+execute_time+'s, model and graph saved in '+save_path)
    if parameters['DO_TRAIN2'] == 'YES':
        # if multistage training specified, repeat above process except for the metagraph saving
        bsize_train2 = int(parameters['BATCH_SIZE2'])
        print('Training stage 2 beginning, loading model...')
        session = tf.Session()
        options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        saver = tf.train.Saver()
        saver.restore(session, model_dir+'save.ckpt')
        if parameters['TBOARD_OUTPUT'] == 'YES':
            writer = tf.summary.FileWriter(parameters['LOG_LOC'] + 'logs', session.graph)
            merged_summaries = tf.summary.merge([cv1w_sum, cv1b_sum, cv1a_sum, fc1w_sum, fc1b_sum, fc1a_sum, fc2w_sum,
                                                 fc2b_sum, fc2a_sum, fc3w_sum, fc3a_sum, batch_cost_sum])
        print('Model loaded, beginning training...')
        execute_time, _ = train_loop(int(parameters['NUM_TRAIN_ITERS2']), float(parameters['LEARN_RATE2']),
                                  float(parameters['KEEP_PROB2']), bsize_train2,
                                  inherit_iter_count=finished_iters)
        save_path = saver.save(session, model_dir)
        session.close()
        print('Training stage 2 finished in ' + execute_time + 's, model saved in ' + save_path)
    # freeze the model and save it to disk after training
    freeze_model(parameters)

if __name__ == '__main__':
    train_neural_network(parameters)
