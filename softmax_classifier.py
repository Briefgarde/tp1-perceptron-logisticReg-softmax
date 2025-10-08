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
        def softmax(z):
            exp_z = np.exp(z)
            sum_z = np.sum(exp_z, axis=-1, keepdims=True)            
            probabilities = exp_z / sum_z
            return probabilities
        
        z = X @ self.W
        z = z - np.max(z, axis=1, keepdims=True) # this recenter the array so it's more stable
        scores = softmax(z)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores

        # loss
        #############################################################################
        # TODO: Compute the softmax loss and store the loss in loss.                #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!     
        #############################################################################
        y_OHE = np.zeros((y.size, y.max() + 1)) # this comes from StackOverFlow
        y_OHE[np.arange(y.size), y] = 1
            # y is initially (n,1), it's a column vector with a number like K to indentify its class
            # meanwhile the loss function require we sort out the correct class probability
            # in z, for one instance, we have a row vector like (1, K) with K-1 of those number being the false probabilities, 
            # and the K probability being the correct one
            # if we OHE the y vector, we can instantly single out the correct one without doing a mask by doing a mult 
            # with z since only the correct probability (encoded to 1) will actually go through (the other will be multi by 0)
            # and we can tell collapse it with np.sum(axis=1) to keep only this correct z per row
        z_correct = np.sum(z * y_OHE, axis=1)
        # print(score_correct.shape) # Nx1
        log_sum_exp_score = np.log(np.sum(np.exp(z), axis=1))
        # print(log_sum_exp_score.shape) # Nx1
        
        loss = - np.sum(z_correct - log_sum_exp_score) / num_train + (0.5*reg) * np.sum(self.W**2)

        # I got this mostly correctly myself, but used scores instead of z the first time around. 

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        
        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        dW = X.T @ (scores - y_OHE) / num_train   + reg*self.W 

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
        y_pred = self.loss(X=X) # thank you David for showing me this shortcut
        y_pred = np.argmax(y_pred, axis=1)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

