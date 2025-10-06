import numpy as np
from random import shuffle
from classifier import Classifier


class Logistic(Classifier):
    """A subclass of Classifier that uses the logistic function to classify."""
    def __init__(self, random_seed=0):
        super().__init__('logistic')
        if random_seed:
            np.random.seed(random_seed)



    def loss(self, X, y=None, reg=0):
        """
        Softmax loss function, vectorized version.

        Inputs and outputs are the same as softmax_loss_naive.
        """
        # Initialize the loss and gradient to zero.
        scores = None
        loss = None
        dW = np.zeros_like(self.W)
        num_classes = self.W.shape[1]
        num_train = X.shape[0]

        #scores
        #############################################################################
        # TODO: Compute the scores and store them in scores.                        #
        #############################################################################
        a = X @ self.W 

        scores = 1 / (1 + np.exp(-a)) # this could be done with self.predict(X)
        eps = 1e-15
        scores = np.clip(scores, eps, 1 - eps)

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        if y is None:
            return scores


        # loss
        #############################################################################
        # TODO: Compute the logistic loss and store the loss in loss.               #
        # If you are not careful here, it is easy to run into numeric instability.  #
        # Don't forget the regularization!                                          #
        #############################################################################
        loss = (- np.dot(y, np.log(scores)) - np.dot((1-y), np.log(1-scores)))/num_train + (0.5*reg) * np.dot(self.W.T, self.W)

        
        # loss = - np.sum(y.T.dot(np.log(scores)) + (1 - y).T.dot(np.log(1-scores))) / num_train + reg * np.linalg.norm(self.W)**2

        loss = loss.item() 
        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################

        # grad
        #############################################################################
        # TODO: Compute the gradients and store the gradients in dW.                #
        # Don't forget the regularization!                                          #
        #############################################################################     
        
        y = y.reshape(-1, 1) 
        dW = np.dot(X.T, (scores - y)) / num_train + reg*self.W 

        #############################################################################
        #                          END OF YOUR CODE                                 #
        #############################################################################
        return loss, dW

    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        a = np.dot(X, self.W)
        y_pred = 1 / (1 + np.exp(-a)) # this contain the probabilities themselves
        # We then adjust the probabilities to predict a classification. 
        y_pred = [1 if pred>0.5 else 0 for pred in y_pred]

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

