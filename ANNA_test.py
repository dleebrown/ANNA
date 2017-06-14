import tensorflow as tf
import convnet2 as cv
import numpy as np
from array import array
from ANNA_train import interpolate_sn, read_param_file, read_known_binary, preprocess_spectra

"""This is the 'test neural network' program. Assumes a model has already been trained and frozen, and runs 
inference on a specified test set of the same binary form as the training data. Outputs parameter error on a 
per-star basis and prints summary statistics to the console to allow for quick checking of model accuracy
"""

# read in the usual parameter file
parameters = read_param_file('ANNA.param')


# loads a frozen tensorflow model, and remaps the inputs originally fed from a queue to be placeholders
def load_frozen_model(parameters, input_px, input_known):
    model_dir = parameters['MODEL_LOC']
    frozen_model = model_dir+'frozen.model'
    model_file = open(frozen_model, 'rb')
    load_graph = tf.GraphDef()
    load_graph.ParseFromString(model_file.read())
    model_file.close()
    tf.import_graph_def(load_graph, input_map={"MASTER_QUEUE/px:0": input_px,
                                               "MASTER_QUEUE/output:0": input_known}, name='test')
    print('frozen model loaded successfully from '+model_dir)


# preprocesses input data, if desired, otherwise just slice into id, flux, and parameter arrays
def preproc_data(normed_outputs, pix_values, sn_range, interp_sn, y_off_range, preprocess):
    num_stars = np.size(pix_values[:, 0])
    if preprocess:
        sn_values = np.random.uniform(sn_range[0], sn_range[1], size=(num_stars, 1))
        y_offsets = np.random.uniform(y_off_range[0], y_off_range[1], size=(num_stars, 1))
        known = normed_outputs[:, 1:-1]
        starnum = normed_outputs[:, 0]
        proc_fluxes = preprocess_spectra(pix_values, interp_sn, sn_values, y_offsets)
    else:
        known = normed_outputs[:, 1:-1]
        starnum = normed_outputs[:, 0]
        proc_fluxes = pix_values
    return starnum, proc_fluxes, known


# undoes min/max normalization
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


# takes in inferred, true parameters and star names and outputs file with per-star error and prints summary stats
def summary_statistics(parameters, star_names, normed_inferred, normed_true, minvals, maxvals):
    save_location = parameters['TEST_SAVE_LOC']
    save_file = save_location + 'test_stats.out'
    star_names = np.reshape(star_names, (np.size(star_names), 1))
    unnormed_inferred = unnormalize_parameters(normed_inferred, minvals, maxvals)
    unnormed_true = unnormalize_parameters(normed_true, minvals, maxvals)
    dtrueinf = unnormed_true - unnormed_inferred
    mean_stats = np.mean(dtrueinf, axis=0)
    std_stats = np.std(dtrueinf, axis=0)
    concat_results = np.concatenate((unnormed_true, dtrueinf), axis=1)
    concat_results = np.concatenate((star_names, concat_results), axis=1)
    np.savetxt(save_file, concat_results, delimiter=',',
               header='star_id,true_temp,true_grav,true_met,true_vt,true_rot,dtemp,dgrav,dmet,dvt,drot')
    print('Test summary output in '+save_file)
    print('Average temp offset:   '+str(round(mean_stats[0], 2)) + "+/-" + str(round(std_stats[0], 2))+' K')
    print('Average grav offset:   '+str(round(mean_stats[1], 2)) + "+/-" + str(round(std_stats[1], 2))+' dex')
    print('Average [Fe/H] offset: '+str(round(mean_stats[2], 2)) + "+/-" + str(round(std_stats[2], 2))+' dex')
    print('Average vt offset:     '+str(round(mean_stats[3], 2)) + "+/-" + str(round(std_stats[3], 2))+' km/s')
    print('Average rotv offset:   '+str(round(mean_stats[4], 2)) + "+/-" + str(round(std_stats[4], 2))+' km/s')


# controls loading a frozen model and some test data, runs inference, and outputs statistics
def test_frozen_model(parameters):
    # import parameter values for normalizing known input parameters and preprocessing spectra
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    num_outputs = int(parameters['NUM_OUTS'])
    num_px = int(parameters['NUM_PX'])
    # read in the training data
    wavelengths, normed_outputs, pix_values = \
        read_known_binary(parameters['TESTING_DATA'], parameters, minvals, maxvals)
    # set to read in the entire test set when running inference and calculating cost
    total_num_stars = np.size(normed_outputs[:, 0])
    # if preprocessing indicated, then read in appropriate parameters, otherwise set to dummy entries
    if parameters['PREPROCESS_TEST'] == 'YES':
        sn_range = np.fromstring(parameters['SN_RANGE_TEST'], sep=', ', dtype=np.float32)
        y_off_range = np.fromstring(parameters['REL_CONT_E_TEST'], sep=', ', dtype=np.float32)
        sn_template_file = parameters['SN_TEMPLATE_TEST']
        interp_sn = interpolate_sn(sn_template_file, wavelengths)
        preprocess = True
    else:
        preprocess = False
        sn_range, y_off_range, interp_sn = 0, 0, 0
    star_ids, spectra, known_outputs = preproc_data(normed_outputs, pix_values, sn_range, interp_sn, y_off_range,
                                                    preprocess)
    # define a few new ops to use as inputs instead of a queue
    with tf.Graph().as_default() as graph:
        input_px = tf.placeholder(tf.float32, shape=[None, num_px], name='input_px')
        input_known = tf.placeholder(tf.float32, shape=[None, num_outputs], name='input_known')
        load_frozen_model(parameters, input_px, input_known)
    # snag the ops needed to actually run inference and cost
    cost = graph.get_tensor_by_name('test/COST/cost:0')
    outputs = graph.get_tensor_by_name('test/FC3_LAYER/fc_output:0')
    dropout = graph.get_tensor_by_name('test/NETWORK_HYPERPARAMS/dropout:0')
    batch_size = graph.get_tensor_by_name('test/NETWORK_HYPERPARAMS/batch_size:0')
    # launch a session and calculate costs
    session = tf.Session(graph=graph)
    cost, outputs = session.run([cost, outputs], feed_dict={input_px: spectra, input_known: known_outputs, dropout:1.0, batch_size: total_num_stars})
    summary_statistics(parameters, star_ids, outputs, known_outputs, minvals, maxvals)
    session.close()

if __name__ == '__main__':
    test_frozen_model(parameters)
