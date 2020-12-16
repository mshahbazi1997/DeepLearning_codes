import tensorflow as tf
from utils import normal_initializer, zero_initializer
from layers import ConvLayer, ConvPoolLayer, DeconvLayer
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS


class AutoEncoder(object):
    def __init__(self):
        # placeholder for storing rotated input images
        self.input_rotated_images = tf.placeholder(dtype=tf.float32,
                                                   shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))
        # placeholder for storing original images without rotation
        self.input_original_images = tf.placeholder(dtype=tf.float32,
                                                    shape=(None, FLAGS.height, FLAGS.width, FLAGS.num_channel))

        # self.output_images: images predicted by model
        # self.code_layer: latent code produced in the middle of network
        # self.reconstruct: images reconstructed by model
        self.code_layer, self.reconstruct, self.output_images = self.build()
        self.loss = self._loss()
        self.opt = self.optimization()

    def optimization(self):
        
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        
        return optimizer.minimize(self.loss)
    
    
    
    def encoder(self, inputs):

        #############################################################################################################
        # TODO: Build Convolutional Part of Encoder                                                                 #
        # Put sequential layers:                                                                                    #
        #       ConvLayer1 ==> ConvPoolLayer1 ==> ConvLayer2 ==> ConvPoolLayer2 ==> ConvLayer3 ==> ConvPoolLayer3   #
        # Settings of layers:                                                                                       #
        # For all ConvLayers: filter size = 3, filter stride = 1, padding type = SAME                               #
        # For all ConvPoolLayers:                                                                                   #
        #   Conv    : filter size = 3, filter stride = 1, padding type = SAME                                       #
        #   Pooling :   pool size = 3,   pool stride = 2, padding type = SAME                                       #
        # Number of Filters:                                                                                        #
        #       num_channel defined in FLAGS (input) ==> 8 ==> 8 ==> 16 ==> 16 ==> 32 ==> 32                        #
        #############################################################################################################
 
        # convolutional layer
        
        cl1 = ConvLayer(input_filters = FLAGS.num_channel, output_filters = 8, act = tf.nn.relu,
                 kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
        conv1 = cl1(self.input_rotated_images)
        print(conv1.shape)


        # convolutional and pooling layer

#         clp1_1 = ConvLayer(input_filters = 8, output_filters = 8, act = tf.nn.relu,
#                          kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
#         clp1_2 = clp1_1(conv1)
        clp1_1 = ConvPoolLayer(input_filters = 8, output_filters = 8, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME',
                         pool_size = 3 , pool_stride = 2, pool_padding='SAME')
        conv_pool1 = clp1_1(conv1)

        print(conv_pool1.shape)
        
        # convolutional layer


        cl2 = ConvLayer(input_filters = 8, output_filters = 16, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
        conv2 = cl2(conv_pool1)
        print(conv2.shape)


        # convolutional and pooling layer


#         clp2_1 = ConvLayer(input_filters = 16, output_filters = 16, act = tf.nn.relu,
#                          kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
#         clp2_2 = clp2_1(conv2)
        clp2_1 = ConvPoolLayer(input_filters = 16, output_filters = 16, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME',
                         pool_size = 3 , pool_stride = 2, pool_padding='SAME')
        conv_pool2 = clp2_1(conv2)

        print(conv_pool2.shape)


        # convolutional layer


        cl3 = ConvLayer(input_filters = 16, output_filters = 32, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
        conv3 = cl3(conv_pool2)
        print(conv3.shape)


        # convolutional and pooling layer


#         clp3_1 = ConvLayer(input_filters = 32, output_filters = 32, act = tf.nn.relu,
#                          kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME')
#         clp3_2 = clp3_1(conv3)
        clp3_1 = ConvPoolLayer(input_filters = 32, output_filters = 32, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 1, kernel_padding = 'SAME',
                         pool_size = 3 , pool_stride = 2, pool_padding='SAME')
        conv_pool3 = clp3_1(conv3)
        
        print(conv_pool3.shape)

        dimentions = conv_pool3.shape
        a1 = int(dimentions[1])
        a2 = int(dimentions[2])
        a3 = int(dimentions[3])
        
        last_conv_dims = (a1,a2,a3)
        
        ##########################################################################
        #                           END OF YOUR CODE                             #
        ##########################################################################

        ##########################################################################
        # TODO: Make Output Flatten and Apply Transformation                     #
        # Please save the last three dimensions of output of the above code      #
        # Save these numbers in a variable called last_conv_dims                 #
        # Multiply all these dimensions to find num of features if flatten       #
        # Use tf.reshape to make a tensor flat                                   #
        # Define some weights and bias and apply linear transformation           #
        # Use normal and zero initializer for weights and bias respectively      #
        # Please store output of transformation in a variable called dense       #
        # Num of features for dense is defined by code_size in FLAG              #
        # Note that there is no need apply any kind of activation function       #
        ##########################################################################

        # make output of pooling flatten
        flatten = tf.reshape(conv_pool3, [tf.shape(conv_pool3)[0], a1 * a2 * a3], name= 'flatten')
        print(flatten.shape)
        
        
        # apply fully connected layer
        
        W_FC = normal_initializer((a1 * a2 * a3, FLAGS.code_size))
        b_FC = zero_initializer(FLAGS.code_size)
        
        dense = tf.add(tf.matmul(flatten, W_FC), b_FC)
        print(dense.shape)

        ##########################################################################
        #                           END OF YOUR CODE                             #
        ##########################################################################

        return dense, last_conv_dims

    def decoder(self, inputs, last_conv_dims):
        
        #########################################################################################
        # TODO: Apply Transformation and Reshape to Original                                    #
        # Define some weights and biases and apply linear transformation                        #
        # Num of output features in this transformation can be calculated using last_conv_dims  #
        # Multiply all the numbers in last_conv_dims to find num of output features             #
        # Please note that number of input features is code_size stored in FLAGS                #
        # Use normal and zero initializer for weights and biases respectively                   #
        # Apply tf.nn.relu activation function                                                  #
        # Finally use last_conv_dims to reshape output of transformation                        #
        #########################################################################################

        # apply fully connected layer
        a1 = last_conv_dims[0]
        a2 = last_conv_dims[1]
        a3 = last_conv_dims[2]
        
        
        
        W_FC = normal_initializer((FLAGS.code_size, a1 * a2 * a3))
        b_FC = zero_initializer(a1 * a2 * a3)
        decode_layer = tf.nn.relu(tf.add(tf.matmul(inputs, W_FC), b_FC))
        
        print(decode_layer.shape)
        
        # reshape to send as input to transposed convolutional layer
        
        deconv_input = tf.reshape(decode_layer, [-1, a1 , a2 , a3], name= 'deconv_input')

        print(deconv_input.shape)

        ###################################################################################
        # TODO: Apply 3 Transposed Convolution Sequentially                               #
        # Put sequential layers:                                                          #
        #       DeconvLayer ==> Deconv Layer ==> Deconv Layer                             #
        # For all layers use:                                                             #
        #       filter size = 3, filter stride = 2, padding type = SAME                   #
        # Apply tf.nn.relu as activation function for first two layers                    #
        # Note that use linear activation function for last layer                         #
        # Multiply all the numbers in last_conv_dims to find num of output features       #
        # Number of filters:                                                              #
        #       num_channel defined in FLAGS (input of first deconv) ==> 16 ==> 8 ==> 1   #
        # Save final output in deconv3                                                    #
        ###################################################################################

        # transpose convolutional layer

        dcl1 = DeconvLayer(input_filters = 32, output_filters = 16, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 2 , kernel_padding = 'SAME')
        deconv1 = dcl1(deconv_input)

        print(deconv1.shape)

        # transpose convolutional layer

        dcl2 = DeconvLayer(input_filters = 16, output_filters = 8, act = tf.nn.relu,
                         kernel_size = 3, kernel_stride = 2 , kernel_padding = 'SAME')
        deconv2 = dcl2(deconv1)
        print(deconv2.shape)

        
        # transpose convolutional layer

        dcl3 = DeconvLayer(input_filters = 8, output_filters = 1, act = tf.identity,
                         kernel_size = 3, kernel_stride = 2 , kernel_padding = 'SAME')
        deconv3 = dcl3(deconv2)
        print(deconv3.shape)
        

        ###################################################################################
        #                                  END OF YOUR CODE                               #
        ###################################################################################
        return deconv3

    def _loss(self):
        ###########################################################################
        # TODO: Loss function                                                     #
        # First flatten reconstruction of images by tf.reshape                    #
        # Flatten shape should be [number of instances, height * width * channel  #
        # Make original input images flat as well                                 #
        # Apply tf.nn.sigmoid_cross_entropy_with_logits to produce loss           #
        #       logits: flatten output                                            #
        #       labels: flatten input                                             #
        # Use tf.reduce_mean for evaluating mean of loss over all instances       #
        # Store mean loss in mean_loss                                            #
        ###########################################################################
        
        labels = tf.reshape(self.input_original_images, [tf.shape(self.input_original_images)[0],-1])
        
        logits = tf.reshape(self.reconstruct, [tf.shape(self.input_original_images)[0],-1])
        
        mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits))
        
        ###########################################################################
        #                              END OF YOUR CODE                           #
        ###########################################################################
        return mean_loss

    def build(self):
        # evaluate encoding of images by self.encoder
        code_layer, last_conv_dims = self.encoder(self.input_rotated_images)

        # evaluate reconstructed images by self.decoder
        reconstruct = self.decoder(code_layer, last_conv_dims)

        # apply tf.nn.sigmoid to change pixel range to [0, 1]
        output_images = tf.nn.sigmoid(reconstruct)

        return code_layer, reconstruct, output_images
