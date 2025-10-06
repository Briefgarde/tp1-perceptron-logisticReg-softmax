import numpy as np
from random import shuffle
from classifier import Classifier


class Softmax(Classifier):
    """A subclass of Classifier that uses the Softmax to classify."""
    def __init__(self, random_seed=0):
        super().__init__('softmax')
        if random_seed:
            np.random.seed(random_seed)

    def loss(self, X, y=None, reg=0):
        scores = None
        # Initialize the loss and gradient to zero.
        loss = 0.0
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]
        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        pass

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        if y is None:
            return scores

        # loss
        #############################################################################
        # TODO: Compute the softmax loss and store the loss in loss.                #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        pass

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        
        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        pass   

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        return loss, dW


    def predict(self, X):
        y_pred = np.zeros(X.shape[1])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        pass

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

