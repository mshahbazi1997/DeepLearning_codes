import tensorflow as tf
import numpy as np


class DenseLayer(object):
    def __init__(self, input_dim, output_dim, act,
                 weight_initializer, bias_initializer, stddev=None):
        super(DenseLayer, self).__init__()
        # save input output dimensions
        self.input_dim = input_dim
        self.output_dim = output_dim

        # saving activation function
        self.act = act
        
        # save initializer
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer

        # weights and biases dictionary
        self.vars = {}

        # standard deviation
        self.stddev = stddev

    def __call__(self, inputs):
        x = inputs
        
        input_dim = int(self.input_dim)
        output_dim= int(self.output_dim)
        
        # bias initialization
        self.vars['bias'] = self.bias_initializer(shape=output_dim)
        #self.vars['bias']  = tf.Variable(tf.zeros(output_dim))

        # weight initialization
        if self.stddev is not None:
            self.vars['weight'] = self.weight_initializer(shape = [input_dim,output_dim], stddev = self.stddev)
            #self.vars['weight'] = tf.Variable(tf.random_normal(shape=(input_dim, output_dim), stddev = self.stddev))
        else:
            self.vars['weight'] = self.weight_initializer(shape=[input_dim, output_dim])
            #self.vars['weight'] = tf.Variable(tf.random_normal(shape=(input_dim, output_dim), stddev = self.stddev))
        
        b = self.vars['bias']
        W = self.vars['weight']
        
        print(b,W,x)
        
        #y = tf.nn.bias_add(tf.matmul(x, W), b)
        #output = self.act(y)
        #################################################################################################
        # TODO: Apply weights and biases over inputs                                                    #
        #   1. Use tf.matmul to multiply inputs by weights                                              #
        #   2. Add bias                                                                                 #
        #   3. Apply activation function                                                                #
        #   4. Save final result in transformed variable                                                #
        #################################################################################################
        transformed = self.act(tf.matmul(x,self.vars['weight'])+self.vars['bias'])

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return transformed
