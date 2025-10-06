# TP1 : Perceptron, Logistic regression, Softmax

In order to simplify the reading of this rapport, I've chosen to talks about things in order in which they appear within the lessons I follow, and in the order in which the notebooks ask the questions. 

## Disclaimer : ChatGPT 

I've used ChatGPT as an education tool for this TP, in both the code sections and this rapport.

For all code sections, I've made sure to avoid directly copying anything ChatGPT suggested and prompted it such that it avoided directly giving the solution, but only provided intuition or hints. When I did finally find a correct result, I've sometime asked ChatGPT to refactor my code to be cleaner and/or faster. 

For the rapport, similarly, I've always written the first draft integraly myself, including all of the formulas. Afterwards, I have asked ChatGPT to check for mistakes, imcomprehension, and grammatical errors. The final output you see here is sometimes copied from ChatGPT after this last step. 


## Perceptron

The Perceptron is a linear model for binary classification that's part of the Linear discriminant family. Because of how it works, the Perceptron does not produce a probability of an instance belonging to a given class since it never learned the probability of the results. Instead, it predict a direct 0/1 result for an instance. 

### Implementation of the scoring function

In a Perceptron model, the computation of the prediction score is very similar to a simple linear regression model, and is defined as:

$$
\hat{y}_i = \text{sign}(\mathbf{x}_i^\top \mathbf{w}_t)
$$

Where:
- $\mathbf{x}_i$ is the feature vector for instance $i$
- $\mathbf{w}_t$ is the current weight vector of the model
- $\text{sign}()$ is a function that maps a real number to either $-1$ or $+1$ depending on its sign

Formally:
$$
\text{sign}(z) = 
\begin{cases}
+1 & \text{if } z > 0 \\
-1 & \text{if } z < 0 \\
0 & \text{if } z = 0
\end{cases}
$$

In practice, if the score happens to be exactly $0$ (which is rare), we can arbitrarily assign the prediction to one of the two labels.  

Thus, the role of $\text{sign}()$ is to transform the scalar product $\mathbf{x}_i^\top \mathbf{w}_t$ into a binary classification decision.  

While this looks similar to linear regression at the prediction stage, the similarities end here: the Perceptron does **not** use the same loss function nor the same optimization procedure.  

In matrix form, if $X \in \mathbb{R}^{n \times d}$ is the design matrix and $w \in \mathbb{R}^d$ the weight vector, the predictions for all $n$ instances are:

$$
\hat{y} = \text{sign}(Xw)
$$

### Implementation of the loss function

As mentioned earlier, a Perceptron model does not use the same loss function as linear regression, which minimizes the squared error between $y$ and $\hat{y}$.  
If we tried to adapt squared error directly to classification, we would effectively be counting the number of misclassified points. This leads to a piecewise constant loss: its gradient is $0$ almost everywhere (and undefined at the jumps), making it useless for optimization.  

Instead, the perceptron loss focuses only on the misclassified instances:

$$
E(\mathbf{w}) = - \sum_{\mathbf{x}_i \in M} \big(\mathbf{x}_i^\top \mathbf{w}\big) y_i
$$

Where:
- $\mathbf{x}_i$ is the feature vector of instance $i$
- $\mathbf{w}$ is the weight vector
- $y_i \in \{-1, +1\}$ is the label
- $M$ is the set of misclassified instances

Intuitively, if an instance is misclassified, then $(\mathbf{x}_i^\top \mathbf{w}) y_i < 0$, and the loss grows with the magnitude of the mistake. Minimizing this loss encourages the model to push misclassified points across the decision boundary with a larger margin.

In matrix form, this can be expressed as:

$$
E(\mathbf{w}) = - (\mathbf{Xw})^\top \left(\frac{\mathbf{y} - \hat{\mathbf{y}}}{2}\right)
\qquad \text{with} \quad \hat{\mathbf{y}} = \text{sign}(\mathbf{Xw})
$$

Here the term $\tfrac{\mathbf{y} - \hat{\mathbf{y}}}{2}$ acts as a selector:  
- if an instance is correctly classified, $y = \hat{y}$, the term is $0$  
- if an instance is misclassified, $y \neq \hat{y}$, the term is $\pm 1$  

Thus, only misclassified samples contribute to the loss, exactly as in the summation form.


### Implementation of the gradient and weight updates. 

From the loss function, we can then compute the gradient to direct our weight update. The gradient is defined like so : 

$$ 
\begin{array}{c} \nabla E({\bf w})=\sum_{{\bf x_i} \in M} {\bf x_i} y_i = -\mathbf{x}^T(y - sign(\mathbf{x}\mathbf{w})) \end{array} $$ 

To minimize the loss, we move in the direction of the negative gradient : 

$$ 
\begin{array}{c} -\nabla E({\bf w})= \mathbf{x}^T(y - sign(\mathbf{x}\mathbf{w})) \end{array} 
$$ 

And we can then proceed with a standard weight update step with a learning rate $η$: 

$$ 
\mathbf w_{n+1} = \mathbf w_n + η \cdot (-\nabla E({\bf w_n})) = \mathbf w_n + η \cdot (\mathbf{x}^T(y - sign(\mathbf{x}\mathbf{w}))) 
$$ 

Where $n$ indicate the current iteration step





### Question 1 : In the figures above we have iterations on $x$-axis. What is an iteration and why are the quantities so different from a plot to another ? What is different from the epochs ?

Iterations is a separate terms to epoch. Epoch design one full pass of training the model where, at the end of the epoch, it will have seen all of the training instance, no matter the batch size. 

An iteration is different from an epoch. An iteration designate one modification of the weights of the model based on the instances seen, which is dictated by the batch size. Multiple iterations can happen in one epoch if the batch size is smaller than the number of instance. 

For example, if n=100, and 'the batch_size'=10, for each epoch, the model will go through 10 iterations. In each iteration, the model will update its weights based on the 10 instances it's currently seeing, then move on to the next iteration. When it'll have done those 10 iteration (and thus seen every instances), it'll move on to the next epoch. 

This explain why the number of epoch is so different per model. The online model use a batchsize equal to the num of instance, so in each iteration, it sees one instance only. In full batch, it uses batchsize = number of instance, so it sees all instance at once. In this model, the num of epoch is equivalent to the num of iteration. Finally, in the mini batch model, the model sees some (10) instances per iteration, and thus need a few iteration to go to the next epoch. 

The num of iteration is given by this formula : 

$$
\text{num of iteration} = \text{num of epoch} * \lceil\frac{\text{num of instances}}{\text{batch size}}\rceil
$$

Where we round up the number of instance divided by the batch size, to make sure to include everything. 


### Question 2 : Comment this plots in your report ! Why are they so different ? What is according to you the best model (full-batch, online or mini-batch) ? Explain your answer !!!

The first thing we can notice with all of those plots is how jagged they are. Notably present in the online history, all of the accuracy curves constantly jump up and down even after they've reached what could be considered their best performances. This is due to how this model (Perceptron) makes its updates based solely on misclassfication, rather than by trying to minimize an overall loss. 

The difference comes from how often the updates are made (directly related to the batch size) and how this influence the model. 

In online batch where updates are made after each instances, each iterations provide only a small amount of information since it sees only one instance. This leads to a lot of noise, a lot of un-optimal updates that slows the progress of the training. 

In full batch, the model makes update at a much slower pace, seeing every instance per iterations. This makes the initial progress slower, but once the weights have slowly been shifted in the correct direction, the accuracy suddenly jumps when the models figures out the correct direction. 

The mini batch is the middle between those two mode. It doesn't have the same noiseness of the online batching method and is faster compared to the full batch mode, meaning it can quickly adjust its weights. The early full swings that happen at the starts can be caused by the model only seeing "strange" batch  of data (all correctly classified, all misclassified) during one iteration, leading to wild swings. Once it starts to get trained enough, the method quickly finds its optimal weights. 

The best model is, in my opinion, a tie between the full or mini batch version. The full model shows a smoother progress with less sudden drops in performances, but its convergence happened slower. Meanwhile, the minibatch model very quickly figured out a solution, but its iteration method mean it is still prone to a variation in performance if it sees bad data. 



### Question 3 : Comment this plots in your report ! What are the surfaces ? What is the line between the surfaces? What are the bullets points ? What are the colors for?

The bullet points are each of the instance within our dataset, with their coordinate being given by their respective features. The color of the bullet point is their class label, the target we're looking for. In both training and test data, we can see that the two group are very well separated. Even in the training data, only one red point is anywhere close to the blue points, they are otherwise really well separated, which means they are easy (or at least possible) to learn for our models. 

The surfaces indicates the area where the models will predict a particular class (decision regions). Any points in the blue area, for example, will be predicted as class 1 while any points in the orange area will be class 2. More mathematically, those areas indicate the ranges of features values where the model will predict the class. With the online model for example, an instance with features sepal_length=6 and sepal_width=4 land in the orange area, and the model will predict the relevant class. 

The line between the surfaces is the decision boundary. Its slope is determined by the two first weights, and the third weight indicate the bias it has, shifting the line up or down. 

Is it's defined as 

$$
w_1x_1 + w_2x_2 + w_0 = 0
$$

Since the dataset is pretty simple and linearly separated, all three models came up with relatively similar decision regions and boundaries. 

### Question 4 : What can you say about the losses of the three perceptrons we visualize above? (Non linearly separable data)

A Perceptron model is explicitly made to handle linearly separable data. When faced with non-linearly separable data, the Perceptron is not able to piece together a result. As such, it's performances fall off a cliff, it cannot converge on a good solution and keep making updates that leads to nowhere. 

Here in the loss plots, we can see that the loss never goes particularly down. Depending on the batch size and the number of iterations, the updates are more or less noisy (online vs full batch size) resulting in varying jaggedness but none of them converge to anything. 

This is because the Perceptron is looking for a linear decision boundary between the instances of the dataset, and can not find any since there are simply none. Without more advanced techniques (suggested by ChatGPT : kernelized perceptrons), a Perceptron can not do better on this kind of data. 

## Logistic Regression

The logistic regression model is an evolution of the Perceptron. Where as the later only produced a direct 0/1 class output, the Logistic Regression output an actual probability of an instance belonging to a given class. 

### Implementation of the score and prediction function

Just like the Perceptron model, a Logistic Regression model begins its prediction by computing the dot product between the input features and its weight vector. The difference lies in the **activation function** that follows this linear combination. 

- In the Perceptron, the **sign** function was used to produce a hard decision (class -1 or +1, or equivalently 0 or 1).  
- In Logistic Regression, the **sigmoid** function is used instead, which produces a *probability-like* output.

The sigmoid function is defined as:

$$
\hat{\mathbf y} = \sigma(a) = \frac{1}{1 + e^{-a}}
$$

where:

- \( a = \mathbf{Xw} \) (the linear combination of features and weights).

The effect of the sigmoid function is to **squash** the unbounded linear score \( a \) into the range \([0,1]\), following an S-shaped curve. This allows us to interpret the output as the probability of the instance belonging to the positive class.

- For very negative values of \( a = \mathbf{x}^T \mathbf{w} \), the sigmoid output approaches 0.  
- For very positive values of \( a \), the output approaches 1.  
- For values of \( a \) near 0, the sigmoid outputs values close to 0.5, indicating high uncertainty in the classification.  

This contrasts with the Perceptron’s sign function, which always outputs a strict 0 or 1 and cannot express uncertainty. Logistic Regression therefore provides a **probabilistic prediction** rather than a hard decision, making it more flexible and informative.


### Implementation of the loss function

To compute the loss of a logistic regression model, we typically use the Negative Log Likelihood. As the classification we're doing follows a Bernoulli distribution, the two concepts are linked. 

The base likelihood of the Bernoulli distribution can be defined as : 



$$
\begin{array}{rcl}
p(\mathbf{y|X}) & = & \prod_{i=1}^N p(y_i|\mathbf{x_i}) \\
 & = & \prod_{i=1}^N p(y_i = C_1| \mathbf{x_i})^{y_i} (1-p(y_i = C_1| \mathbf{x_i}))^{1-y_i} \\
\end{array}
$$

With this set, we can move to the negative log likelihood like so : 

$$
\begin{array}{rcl}
E(\mathbf{w}) & = & - ln(p(\mathbf{y} | \mathbf{X}))  \\
 & = & - ln \prod_{i=1}^N p(y_i = C_1| \mathbf{x_i})^{y_i} (1-p(y_i = C_1| \mathbf{x_i}))^{1-y_i} \\
 & = & - \sum_{i=1}^{N} y_i \ln p(y_i = C_1| \mathbf{x_i}) - \sum_{i=1}^{N} (1-y_i)\ln(1-p(y_i = C_1| \mathbf{x_i})) \\
 & = & - \sum_{i=1}^{N} y_i \ln(\sigma (w^Tx_i)) - \sum_{i=1}^{N} (1-y_i)\ln(1-(\sigma (w^Tx_i))) \\
\end{array}
$$

Where : 
- $\sigma (w^Tx_i) = \frac{1}{1 + e^{-a}}$ and $a = w^Tx_i$

**The negative log likelihood is the cost function we use for the Logistic regression.**

All of which we can finally translate into matrix form like so : 

$$
\begin{array}{rcl}
E(\mathbf{w}) & = & - \sum_{i=1}^{N} y_i \ln(\sigma (\mathbf{w^Tx_i})) - \sum_{i=1}^{N} (1-y_i)\ln(1-(\sigma (\mathbf{w^Tx_i}))) \\
& = & - \frac{1}{N}[\mathbf{y^T} \ln(\sigma (\mathbf{Xw})) - (\mathbf{1-y})^T \ln(\mathbf{1}- \sigma (\mathbf{Xw}))] \\
& = & - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})] \\
\end{array}
$$

Where $\hat{y} = \sigma(\mathbf{Xw})$ (for the final row).

#### Adding the L2 regularization

Now that we've defined the cost function as the negative log likelihood, adding a L2 regularization is easy. 

If : 

$$
E(\mathbf{w}) = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})]
$$

Then the cost function with L2 regularization is : 

$$
E_{\text{l2}}(\mathbf{w}) = \underbrace{E(\mathbf{w})}_{nll} + \underbrace{\lambda \mathbf{w^Tw}}_{l2} = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})] + \lambda \mathbf{w^Tw}
$$

Where : 
- $\lambda$ is an hyperparameter that control the strength of the L2 regularization. 

This is the base cost function with regularization. Depending on the sources consulted, some recommend to divide the L2 part by $\frac{1}{2}$ or by $\frac{1}{2N}$, which allows for an easier derivation of the gradient. Other also recommend to not include the intercept term ($w_0$) in the L2 regularization, as it is not a standard weight and simply shift the general result up or down. Various professional libraries (sci-kit learn, tensorflow, pytorch, etc...) sometimes vary on this aspect. 

### Implementation of the gradient function : TODO

With the cost function defined above, we can now take the derivative to guide our optimization process. 

$$
E(\mathbf{w}) = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})]
$$

Before doing the full derivation, let's already find the derivation of the core sigmoid function : 

$$
\begin{array}{rcl}
\sigma(x) &  = &  \frac{1}{1+e^{-x}} \\
\nabla_x \sigma(x) & = & \frac{d(\sigma(x))}{x}\\
 & = & \frac{1}{1+e^{-x}} \cdot (1 - \frac{1}{1+e^{-x}}) \\
 & = & \sigma(x)(1-\sigma(x))
\end{array}
$$

Using the chain rule we can start working on the main derivation. 

$$
\begin{array}{rcl}
\nabla_{\mathbf{w}} E(\mathbf{w}) & = & \frac{d(E(w))}{d(w)} \\
& = & \frac{1}{N}\mathbf{X^T}(\sigma_w(\mathbf{X}) - \mathbf{y})
\end{array}
$$




#### Gradient with L2 regularization

Thanks to the addition rule of derivation, the gradient with L2 regularization is straight forward. 

If : 

$$
\nabla_{\mathbf{w}} E(\mathbf{w}) = \frac{1}{N}\mathbf{X^T}(\sigma_w(\mathbf{X}) - \mathbf{y})
$$

And : 

$$
E_{\text{l2}}(\mathbf{w}) = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})] + \lambda \mathbf{w^Tw}
$$

Then the gradient of the cost function with L2 regularization is defined as : 

$$
\begin{array}{rcl}
\nabla E_{\text{l2}}(\mathbf{w}) & = & \nabla_{\mathbf{w}} E(\mathbf{w}) + \nabla(\lambda \mathbf{w^Tw}) \\
& = & \nabla_{\mathbf{w}} E(\mathbf{w}) + 2\lambda\mathbf{w}
\end{array}
$$

Giving this as the gradient when using L2 regularization : 

$$
\nabla E_{\text{l2}}(\mathbf{w}) = \frac{1}{N}\mathbf{X^T}(\sigma_w(\mathbf{X}) - \mathbf{y}) + 2\lambda\mathbf{w}
$$


### Fine tuning of the Logistic Regression : Learning Rate and Regularization : TODO

When testing our multiple different hyperparameter like the learning rate and the strength of the regulation, a few different thing arise that are rather odd. 

The 

## Softmax : TODO
