import numpy as np

class MLP:
    def __init__(self):
        self.layers = []
        self.mode = 'TRAIN'
        
    def add(self, layer):
        '''
        add a new layer to the layers of model.
        '''
        self.layers.append(layer)
    
    def set_mode(self, mode):
        if mode == 'TRAIN' or mode == 'TEST':
            self.mode = mode
        else:
            raise ValueError('Invalid Mode')
    
    def forward(self, x, y):
        loss, scores = None, None
        ############################################################################
        # TODO: Implement the forward pass of MLP model.                           #
        # Note that you have the layers of the model in self.layers and each one   #
        # of them has a forward() method.                                          #
        # The last layer is always a LossLayer which in this assignment is only    #
        # SoftmaxCrossEntropy.                                                     #
        # You have to compute scores (output of model before applying              #
        # SoftmaxCrossEntropy) and loss which is the output of SoftmaxCrossEntropy #
        # forward pass.                                                            #
        # Do not forget to pass mode=self.mode to forward pass of the layers! It   #
        # will be used later in Dropout and Batch Normalization.                   #
        # Do not forget to add the L2-regularization loss term to loss. You can    #
        # find whether a layer has regularization_strength by using get_reg()      #
        # method. Note that L2-regularization is only used for weights of fully    #
        # connected layers in this assignment.                                     #
        ############################################################################
        pass
        L2loss = 0
        for i, layer in enumerate(self.layers):
            
            if ('FullyConnected') in layer.__str__():
                
                # we have to sum-up all squered weights together and add it to final loss
                
                regularization_parameter = layer.get_reg()
                weights = layer.get_params()['w'].data
                L2loss = L2loss + regularization_parameter*np.sum(weights**2)
            
            if i == 0:
                
                # first layer
                
                output = layer.forward(x, mode=self.mode)
                
            elif i == len(self.layers)-1 :
                
                # last layer
                
                scores = np.copy(output)
                loss = layer.forward(output, y)
                loss = loss + L2loss
                
            else:
                
                # hidden layers
                
                output = layer.forward(output, mode=self.mode)
                
        return scores, loss
        
        
    def backward(self):
        ############################################################################
        # TODO: Implement the backward pass of the model. Use the backpropagation  #
        # algorithm.                                                               #                         
        # Note that each one of the layers has a backward() method and the last    #
        # layer would always be a SoftmaxCrossEntropy layer.                       #
        ############################################################################
        pass
        for i in range (len(self.layers)-1,-1,-1):
            
            layer = self.layers[i]
            
            if i == len(self.layers)-1:
                dout = layer.backward()
            else:
                dout = layer.backward(dout)
            
    def __str__(self):
        '''
        returns a nice representation of model
        '''
        splitter = '===================================================='
        return splitter + '\n' + '\n'.join('layer_{}: '.format(i) + 
                                           layer.__str__() for i, layer in enumerate(self.layers)) + '\n' + splitter
