import tensorflow as tf
import convnet2 as cv
import numpy as np
from array import array
import readmultispec as rmspec
from scipy import interpolate as ip
from ANNA_train import interpolate_sn, read_param_file, read_known_binary

"""This is the inference program. Assumes a model has already been trained and frozen, and runs 
inference on a specified test set of the same binary form as the training data. Outputs parameter error on a 
per-star basis and prints summary statistics to the console to allow for quick checking of model accuracy

The multispec reader 'readmultispec.py' comes from 
Kevin Gullikson, based originally on a script by Rick White (2012). The script can be found at:
Kevin Gullikson. (2014, May 20). General-Scripts_v1.0. Zenodo. http://doi.org/10.5281/zenodo.10013
"""

# read in the usual parameter file
parameters = read_param_file('ANNA.param')


# loads a frozen tensorflow model, and remaps the inputs originally fed from a queue to be placeholders
def load_frozen_model_infer(parameters, input_px):
    model_dir = parameters['INFER_MODEL_LOC']
    frozen_model = model_dir+'frozen.model'
    model_file = open(frozen_model, 'rb')
    load_graph = tf.GraphDef()
    load_graph.ParseFromString(model_file.read())
    model_file.close()
    tf.import_graph_def(load_graph, input_map={"MASTER_QUEUE/px:0": input_px}, name='infer')
    print('frozen model loaded successfully from '+model_dir)


# reads in a multispec file and parses it, returning 2 arrays - ids and fluxes interpolated on wavelength grid
def read_multispec_file(parameters):
    image_name = parameters['MS_FITS_IMAGE']
    wavelength_template = parameters['WAVE_TEMPLATE']
    wavelength_grid = np.loadtxt(wavelength_template)[:, 0]
    image = rmspec.readmultispec(image_name)
    wavelengths = image['wavelen']
    flux = image['flux']
    # for now just make ids a np arange until I figure out how to hook the WOCS IDs
    num_stars = len(flux)
    ids = np.arange(num_stars)
    ids = np.reshape(ids, (num_stars, 1))
    output_array = np.zeros((num_stars, np.size(wavelength_grid)), dtype=np.float32)
    for entry in range(num_stars):
        current_fluxes = flux[entry]
        current_wavelengths = wavelengths[entry]
        function = ip.interp1d(current_wavelengths, current_fluxes, kind='linear')
        interpolated_spectrum = function(wavelength_grid)
        output_array[entry, :] = interpolated_spectrum
    return ids, output_array


# this is just a wrapper to slice input binaries of the form used for training - only used in debug mode
def debug_mode_input_handler(normed_outputs, pix_values):
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


# takes in inferred parameters and star names and outputs file with per-star parameters
def save_output(parameters, star_names, normed_inferred, minvals, maxvals):
    save_location = parameters['INFER_SAVE_LOC']
    save_file = save_location + 'infer_stats.out'
    star_names = np.reshape(star_names, (np.size(star_names), 1))
    unnormed_inferred = unnormalize_parameters(normed_inferred, minvals, maxvals)
    concat_results = np.concatenate((star_names, unnormed_inferred), axis=1)
    np.savetxt(save_file, concat_results, delimiter=',',
               header='fits_row,infer_temp,infer_grav,infer_met,infer_vt,infer_rot')
    print('Inference summary output in '+save_file)


# controls loading a frozen model and some data, runs inference, and outputs results
def run_inference(parameters):
    # import parameter values for unnormalizing inferred parameters
    minvals = np.fromstring(parameters['MIN_PARAMS'], sep=', ', dtype=np.float32)
    maxvals = np.fromstring(parameters['MAX_PARAMS'], sep=', ', dtype=np.float32)
    num_px = int(parameters['NUM_PX'])
    if parameters['DEBUG_MODE'] == 'YES':
        # debug mode assumes inputs of the binary form (with parameters) used for training and testing
        wavelengths, normed_outputs, pix_values = \
            read_known_binary(parameters['DEBUG_DATA'], parameters, minvals, maxvals)
        # set to read in the entire test set when running inference in debug mode
        total_num_stars = np.size(normed_outputs[:, 0])
        star_ids, spectra, _ = debug_mode_input_handler(normed_outputs, pix_values)
    elif parameters['SINGLE_MS_MODE'] == 'YES':
        star_ids, spectra = read_multispec_file(parameters)
        total_num_stars = np.size(star_ids)
    # define a few new ops to use as inputs instead of a queue
    with tf.Graph().as_default() as graph:
        input_px = tf.placeholder(tf.float32, shape=[None, num_px], name='input_px')
        load_frozen_model_infer(parameters, input_px)
    # snag the ops needed to actually run inference
    outputs = graph.get_tensor_by_name('infer/FC3_LAYER/fc_output:0')
    dropout = graph.get_tensor_by_name('infer/NETWORK_HYPERPARAMS/dropout:0')
    batch_size = graph.get_tensor_by_name('infer/NETWORK_HYPERPARAMS/batch_size:0')
    # launch a session and run inference
    session = tf.Session(graph=graph)
    outputs = session.run(outputs, feed_dict={input_px: spectra, dropout: 1.0, batch_size: total_num_stars})
    # save the outputs to specified file
    save_output(parameters, star_ids, outputs, minvals, maxvals)
    session.close()

if __name__ == '__main__':
    run_inference(parameters)
