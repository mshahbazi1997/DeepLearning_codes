import torch
import torch.nn as nn

class Flags:
    def __init__(self):
        self.learning_rate = 0.0005
        self.width = 32
        self.height = 32
        self.num_channel = 1
        self.batch_size = 10
        self.num_epochs = 5
        self.code_size = 256

FLAGS = Flags()


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        # placeholder for storing rotated input images
        self.input_rotated_images = nn.Parameter(torch.empty((None, FLAGS.height, FLAGS.width, FLAGS.num_channel)))
        # placeholder for storing original images without rotation
        self.input_original_images = nn.Parameter(torch.empty((None, FLAGS.height, FLAGS.width, FLAGS.num_channel)))

        # self.output_images: images predicted by model
        # self.code_layer: latent code produced in the middle of network
        # self.reconstruct: images reconstructed by model
        self.code_layer, self.reconstruct, self.output_images = self.build()
        self.loss = self._loss()
        self.opt = self.optimization()

    def optimization(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=FLAGS.learning_rate)
        return optimizer
    
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
        cl1 = nn.Conv2d(FLAGS.num_channel, 8, kernel_size=3, stride=1, padding=1)
        conv1 = nn.ReLU()(cl1(inputs))
        print(conv1.shape)

        # convolutional and pooling layer
        clp1_1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        clp1_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        conv_pool1 = nn.ReLU()(clp1_2(clp1_1(conv1)))
        print(conv_pool1.shape)

        # convolutional layer
        cl2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        conv2 = nn.ReLU()(cl2(conv_pool1))
        print(conv2.shape)

        # convolutional and pooling layer
        clp2_1 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1)
        clp2_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        conv_pool2 = nn.ReLU()(clp2_2(clp2_1(conv2)))
        print(conv_pool2.shape)
