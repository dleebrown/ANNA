import tensorflow as tf
import numpy as np

'''
Hopefully everything needed to train and run a CNN for relatively arbitrary inputs.

Some nice little helper functions:
    Layers to string together to generate NN architecture:
        layer_conv(input,weights,stride=[1,1,1,1],padding='SAME')
            computes the convolution of an input tensor with appropriately sized weight tensor
            stride is [1 example, 1 height, 1 width, 1 channel], padding is symmetric by default
            input: [num_examples,input_height,input_width,input_channels]
            weights:[filter_height,filter_width,input_channels,output_channels]
            output: [num_examples,input_height,input_width,output_channels] ###(CHECK THIS)###
        layer_activate(input,bias,type={'sigmoid','relu','tanh'})
            nonlinear activation function of specified type
            input: tensor
            output: tensor of same dimension
            defaults to linear (identity) activation if no type specified
        layer_dropout(input,keep_probability=0.0)
            performs dropout i.e. sets random activations to 0, and scales other activations accordingly
            input: tensor
            output: tensor of same dimension
        layer_pool(input,size,type={'max','average'})
            performs specified pooling operation.
            uses disjoint pools, i.e., each parameter is assigned to a unique pool
            input: [num_examples,height,width,channels]
            output: ~[num_examples/size[0],height/size[1],width/size[2],channels/size[3]]
            defaults to no pooling if type not specified
        layer_flatten(input)
            flattens a 4d tensor to a 2d tensor, holding first axis constant
            input: [num_examples, height, width, channels]
            output: [num_examples, height*width*channels]
        layer_fc(input,weights)
            computes a fully-connected convolution - all inputs are connected through weight tensor
            requires 2d input: [num_examples,num_parameters]
            weights need to be: [num_parameters,num_outputs]
            returns a tensor: [num_examples,num_outputs]

    Optimization functions:
        network_cost(predicted_vals,known_vals,type={'x_entropy_sig','x_entropy_soft','l2'}, combine={'mean','sum'})
            computes the cost associated with output values
            softmax cross entropy for exclusive classification, sigmoid for non-exclusive (multi-class)
            note that input assumes logits (raw values) for predicted_vals for the cross entropy cost functions,
            inputs: 2 tensors of identical dimension (generally [# examples,[output parameters]]
            output: a number calculated according to the 'combine' parameter
        network_optimize(cost_function, learn_rate, optimizer={'grad_desc','adadelta', 'adam'})
            returns selected optimization function with indicated learning rate and cost function
            assumes standard graph flow where weights are stored as variables and updated as cost is optimized
            input: cost function that will also initiate a forward feed step
            output: function to optimize
            recommend learning rates of 0.001 for adadelta (Zeiler 2012) and adam (Kingma 2014)
            grad_desc will need tuning as usual

    Parameter initialization:
        generate_weights(shape,type={'normal','uniform','p_uniform'})
            initializes a tensor variable with dimension [shape]
            values initialized to specified function:
                'normal' gives normal distribution, mean 0, stdev 0.2
                'uniform' gives uniform distro, mean 0, min/max -0.2/0.2
                'p_uniform' gives uniform distro, mean 0.2, min/max 0/0.2
            defaults to normal if no distro specified
            returns a tensor: [shape]
        generate_bias(length,value=0.05):
            initializes a 1D bias tensor variable with entries initialized to constant value
            returns a tensor: [length]

    IO functions:
        save_weights_txt(weights, name)
            saves weights in the current directory with given name as a .txt file
            converts to a numpy array first, then calls np.savetext
            input: tensor
            output: none
        load_weights_txt(name)
            loads .txt file
            returns a tensor version of loaded file in float32 format
assumes everything is NHWC format throughout; cue cheers from the fortran lovers
'''

### NEURAL NETWORK LAYER FUNCTIONS ###

def layer_conv(input,weights,stride=[1,1,1,1]):
    output=tf.nn.conv2d(input=input,filter=weights,strides=stride,padding='SAME')
    return output

def layer_activate(input,bias,type='linear'):
    output=input+bias
    if type=='sigmoid':
        output=tf.nn.sigmoid(output)
        return output
    elif type=='relu':
        output=tf.nn.relu(output)
        return output
    elif type=='tanh':
        output=tf.nn.tanh(output)
        return output
    else:
        print('no activation function selected, returning linear activation')
        return output

def layer_dropout(input,keep_probability):
    output=tf.nn.dropout(input,keep_probability)
    return output

def layer_pool(input,stride,type='none',padding='SAME'):
    if type=='max':
        output=tf.nn.max_pool(value=input,ksize=stride,strides=stride,padding=padding)
        return output
    elif type=='average':
        output=tf.nn.avg_pool(value=input,ksize=stride,strides=stride,padding=padding)
        return output
    else:
        print('no pooling specified, returning input')
        return input

def layer_flatten(input):
    shape=input.get_shape()
    nparams=shape[1:4].num_elements()
    output=tf.reshape(input,[-1,nparams])
    return output,nparams

def layer_fc(input,weights):
    output=tf.matmul(input,weights)
    return output

### OPTIMIZATION FUNCTIONS ###
def network_cost(predicted_vals,known_vals, type='l2',combine='mean'):
    if type=='l2':
        difference=known_vals-predicted_vals
        output=2*tf.nn.l2_loss(difference)
        return output
    elif type=='x_entropy_sig':
        cost=tf.nn.sigmoid_cross_entropy_with_logits(logits=predicted_vals,targets=known_vals)
        if combine=='mean':
            output=tf.reduce_mean(cost)
            return output
        elif combine=='sum':
            output=tf.reduce_sum(cost)
            return output
        else:
            print('no combine operation selected, returning mean cost')
            output = tf.reduce_mean(cost)
            return output
    elif type == 'x_entropy_soft':
        cost = tf.nn.softmax_cross_entropy_with_logits(logits=predicted_vals, labels=known_vals)
        if combine == 'mean':
            output = tf.reduce_mean(cost)
            return output
        elif combine == 'sum':
            output = tf.reduce_sum(cost)
            return output
        else:
            print('no combine operation selected, returning mean cost')
            output = tf.reduce_mean(cost)
            return output
    else:
        print('no cost function selected, returning l2')
        difference = known_vals - predicted_vals
        output = 2 * tf.nn.l2_loss(difference)
        return output

def network_optimize(cost_function, learn_rate, optimizer='grad_desc'):
    if optimizer=='grad_desc':
        optim_function = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost_function)
    elif optimizer=='adadelta':
        optim_function=tf.train.AdadeltaOptimizer(learning_rate=learn_rate).minimize(cost_function)
    elif optimizer=='adam':
        optim_function=tf.train.AdamOptimizer(learning_rate=learn_rate).minimize(cost_function)
    else:
        print('no optimizer selected, using grad_descent')
        optim_function=tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(cost_function)
    return optim_function

### INITIALIZATION FUNCTIONS ###

def generate_weights(shape,type='normal', function='None'):
    if type=='normal':
        output = tf.Variable(tf.random_normal(shape, mean=0.01, stddev=0.003))
        return output
    elif type=='uniform':
        output = tf.Variable(tf.random_uniform(shape,minval=-0.2,maxval=0.2))
        return output
    elif type=='p_uniform':
        output = tf.Variable(tf.random_uniform(shape, minval=0.0, maxval=0.01))
        return output
    elif type=='custom':
        output = tf.Variable(function)
        return output
    else:
        print('no initialization specified, returning normal initialization')
        output=tf.Variable(tf.random_normal(shape,mean=0.0,stddev=0.2))
        return output

def generate_bias(length,value=0.05):
    output=tf.Variable(tf.constant(value,shape=[length]))
    return output

### IO FUNCTIONS ###

def save_weights_txt(weights,name='saved_weights'):
    output=weights.eval()
    np.savetxt(name,output)

def load_weights_txt(name):
    input = np.loadtxt(name)
    output = tf.convert_to_tensor(value=input, dtype=tf.float32)
    return output