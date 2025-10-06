import numpy as np


class Classifier(object):
    def __init__(self, classifier_type):
        self.classifier_type = classifier_type
        self.W = None

    def train(
        self,
        X,
        y,
        X_val=None,
        y_val=None,
        learning_rate=1e-3,
        reg=1e-5,
        num_iters=100,
        batch_size=200,
        verbose=False,
    ):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
        training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
        means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.
        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = (
            np.max(y) + 1
        )  # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            # lazily initialize W
            if self.classifier_type == "logistic":
                self.W = 0.001 * np.random.randn(dim, 1)
            else:
                self.W = 0.001 * np.random.randn(dim, num_classes)
        # Run stochastic gradient descent to optimize W
        loss_history = []
        acc_history = []

        loss_val_history = []
        acc_val_history = []

        for it in range(1, num_iters + 1):
            X_batch = None
            y_batch = None

            indices = np.random.choice(num_train, batch_size)
            X_batch = X[indices]
            y_batch = y[indices]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            if X_val is not None:
                loss_val, _ = self.loss(X_val, y_val, reg)
                loss_val_history.append(loss_val)
                acc_val_history.append(np.mean(self.predict(X_val) == y_val))

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.W = self.W - learning_rate * grad 

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            acc_history.append(np.mean(self.predict(X_batch) == y_batch))

            if verbose and it % 100 == 0:
                print(
                    "iteration {} / {} : loss {}".format(it, num_iters, loss), end="\r"
                )
        if verbose:
            print("".ljust(70), end="\r")

        return loss_history, acc_history, loss_val_history, acc_val_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.
        Inputs:
        - X: D x N array of training data. Each column is a D-dimensional point.
        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        pass

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.
        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.
        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass
