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

- $ a = \mathbf{Xw} $ (the linear combination of features and weights).

The effect of the sigmoid function is to **squash** the unbounded linear score $ a $ into the range $[0,1]$, following an S-shaped curve. This allows us to interpret the output as the probability of the instance belonging to the positive class.

- For very negative values of $ a = \mathbf{x}^T \mathbf{w} $, the sigmoid output approaches 0.  
- For very positive values of $ a $, the output approaches 1.  
- For values of $ a $ near 0, the sigmoid outputs values close to 0.5, indicating high uncertainty in the classification.  

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

### Implementation of the Gradient Function

With the cost function defined above, we can now take its derivative with respect to the weights to guide our optimization process.  
Recall that the binary logistic regression cost (negative log-likelihood) is defined as:

$$
E(\mathbf{w}) = - \frac{1}{N}\left[\mathbf{y}^T \ln(\hat{\mathbf{y}}) + (\mathbf{1 - y})^T \ln(\mathbf{1} - \hat{\mathbf{y}})\right]
$$

To work out this derivative, we’ll use the **chain rule**, which allows us to break this large expression into smaller, simpler parts that are easier to differentiate.  
We define the intermediate functions as follows:

- \(h(\mathbf{w}) = \mathbf{Xw} = \mathbf{z}\)
- \(g(\mathbf{z}) = \sigma(\mathbf{z}) = \frac{1}{1 + e^{-\mathbf{z}}} = \hat{\mathbf{y}}\)
- \(f(\hat{\mathbf{y}}) = E(\mathbf{w}) = - \frac{1}{N}\left[\mathbf{y}^T \ln(\hat{\mathbf{y}}) + (\mathbf{1 - y})^T \ln(\mathbf{1} - \hat{\mathbf{y}})\right]\)

Applying the chain rule gives:

$$
\nabla_{\mathbf{w}} E(\mathbf{w}) = \frac{df}{d\hat{\mathbf{y}}} \cdot \frac{dg}{d\mathbf{z}} \cdot \frac{dh}{d\mathbf{w}}
$$

Let’s now derive each term individually.

---

#### Step 1: Derivative of \(h(\mathbf{w}) = \mathbf{Xw}\)

This first part is straightforward:

$$
\frac{dh}{d\mathbf{w}} = \mathbf{X}
$$

---

#### Step 2: Derivative of \(g(\mathbf{z}) = \sigma(\mathbf{z}) = (1 + e^{-\mathbf{z}})^{-1}\)

We differentiate with respect to \(\mathbf{z}\):

$$
\frac{dg}{d\mathbf{z}} = \frac{d}{d\mathbf{z}}(1 + e^{-\mathbf{z}})^{-1}
$$

We can apply the chain rule again here by defining:
- \(r(\mathbf{z}) = 1 + e^{-\mathbf{z}}\)
- \(t(u) = u^{-1}\)

Then:

$$
\frac{dg}{d\mathbf{z}} = t'(r(\mathbf{z})) \cdot r'(\mathbf{z})
$$

Compute each derivative:

$$
t'(u) = -u^{-2} = -(1 + e^{-\mathbf{z}})^{-2}, \qquad r'(\mathbf{z}) = -e^{-\mathbf{z}}
$$

Combining:

$$
\frac{dg}{d\mathbf{z}} = (-(1 + e^{-\mathbf{z}})^{-2})(-e^{-\mathbf{z}}) = \frac{e^{-\mathbf{z}}}{(1 + e^{-\mathbf{z}})^2}
$$

We can rewrite this in a more compact and interpretable form using \(\sigma(\mathbf{z})\):

$$
\frac{dg}{d\mathbf{z}} = \sigma(\mathbf{z})(1 - \sigma(\mathbf{z}))
$$

As **the derivative of the sigmoid function**, this is one of the most important identities in logistic regression.

---

#### Step 3: Derivative of \(f(\hat{\mathbf{y}})\)

Now, for the derivative of the loss with respect to the predicted probabilities \(\hat{\mathbf{y}}\):

$$
\frac{df}{d\hat{\mathbf{y}}} 
= -\left( \frac{\mathbf{y}}{\hat{\mathbf{y}}} - \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} \right)
$$

This comes directly from the logarithmic derivative rule 
\(\frac{d}{dx} \ln(x) = \frac{1}{x}\), applied to each term in the cost.

---

#### Step 4: Putting it all together

Now, combine all three derivatives according to the chain rule:

$$
\begin{array}{rcl}
\nabla_{\mathbf{w}} E(\mathbf{w}) 
&=& \frac{df}{d\hat{\mathbf{y}}} \cdot \frac{dg}{d\mathbf{z}} \cdot \frac{dh}{d\mathbf{w}} \\[8pt]
&=& \underbrace{-\left( \frac{\mathbf{y}}{\hat{\mathbf{y}}} - \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} \right)}_{\frac{df}{d\hat{\mathbf{y}}}}
\underbrace{\hat{\mathbf{y}}(1 - \hat{\mathbf{y}})}_{\frac{dg}{d\mathbf{z}}}
\underbrace{\mathbf{X}}_{\frac{dh}{d\mathbf{w}}}
\end{array}
$$

Simplifying:

$$
\nabla_{\mathbf{w}} E(\mathbf{w}) 
= -\mathbf{X}^T [(\mathbf{y} - \hat{\mathbf{y}})]
$$

or equivalently (reversing the sign to match gradient descent convention):

$$
\nabla_{\mathbf{w}} E(\mathbf{w}) = \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})
$$

Finally, reintroducing the normalization over \(N\) samples:

$$
\nabla_{\mathbf{w}} E(\mathbf{w}) = \frac{1}{N} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y})
$$

---

This final expression is the gradient of the logistic regression loss.  
It tells us that, on average, the weights should be updated proportionally to the difference between the predicted probabilities and the true labels, scaled by the input features \(\mathbf{X}\).





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

Firstly, a lot of combination of parameters end up with surprisingly the exact same accuracy values. For example, a learning rate of $1e-06$ and any of 0.0001, 0.01 or 0 regulation end up with the same accuracy (0.651 the last time I ran the code). This hint that, no matter the combination, the model end up in the exact same minima it find, and the hyperparameters do not manage to change the result. 

The second thing that comes to mind is finding of the best regulation strength, which is 10'000. Most of the time, we're used to hyperparameters being rather small, seeing how the learning rate is in the range 0.0000001-0.000000001 for example. Yet, here, the best regulation strength is not just well above 0, but outright 10K ! Such a regulation strength means the model seem to perform the best when weights are *extremely* small. 

At lower learning rate ($1e-07, 1e-08$), we see that the model barely learn anything comparated to the $1e-06$ learning rate. All validation accuracy remain in the zone of ~0.42, with little to no impact from the regulation strength. This seem to hint that a learning rate this small is unable to escape the initial zone of initialization of its weights, and the model is unable to take steps to avoid this trap. 

## Softmax classifier / Multi class logistic regression

### Implementation of the Scores

<sub>I know this part really sounds like it got written entierely by ChatGPT, but I did really go through the trouble of planning out the dimensions of every thing myself first, and figuring out the per instance and per class prediction. ChatGPT just rewrote those info into pretty $W \in \mathbb{R}^{(\text{dim}, K)}$ notations. </sub>


As an extension of logistic regression, the softmax classifier follows a very similar structure to its binary predecessor.

We start by taking the dot product between an input instance and the model’s weight parameters. In binary logistic regression, the **sigmoid function** is used to squash the resulting score into a probability between 0 and 1. In the multiclass setting, we instead use the **softmax function**, which performs a similar transformation but ensures that *all* class probabilities are represented and that they sum to 1.

A key structural difference lies in the **weight matrix**.  
Instead of a single weight vector, the softmax classifier maintains a weight matrix of shape  : 

$$
W \in \mathbb{R}^{(\text{dim}, K)},
$$

where $K$ is the number of classes.  
This means that the output of the dot product for a single instance $x_i$ is:

$$
(1, \text{dim}) \cdot (\text{dim}, K) = (1, K).
$$

#### Score Function

The scores for one instance are defined as:

$$
\text{score}_i = \text{softmax}(z_i)
$$

where:

$$
z_i = x_i W
$$

and the softmax function is:

$$
\text{softmax}(z_i)_k = \frac{e^{z_{ik}}}{\sum_{j=1}^{K} e^{z_{ij}}}.
$$

#### Per-Class and Per-Instance Forms

For a specific class $k$:

$$
z_{ik} = x_i w_k,
$$

where:
- $x_i \in \mathbb{R}^{(1, \text{dim})}$ is the input instance (a row vector),
- $w_k \in \mathbb{R}^{(\text{dim}, 1)}$ is the weight column corresponding to class $k$.

Thus, $z_{ik}$ is a scalar — the score of instance $i$ for class $k$

To compute the scores for all classes at once:

$$
z_i = x_i W
$$

where:
- $x_i \in \mathbb{R}^{(1, \text{dim})}$
- $W \in \mathbb{R}^{(\text{dim}, K)}$
- $z_i \in \mathbb{R}^{(1, K)}$ is the vector of scores across all classes.

Finally, the predicted probability for class $k$ given instance $x_i$ is:

$$
\hat{y}_{ik} = \frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}}.
$$

### Implementation of the Loss Function

The process to obtain the loss function for a Softmax classifier is conceptually very similar to that used for Logistic Regression (and, more broadly, for many probabilistic models).  
We start by defining the **Negative Log-Likelihood (NLL)** of the model, which we will use as our cost function.

#### 1. The Categorical Distribution

The Softmax classifier is based on the **Categorical distribution**, which extends the Bernoulli distribution to multiple classes instead of a binary setup.  
It is defined as:

$$
P(y = i) = p_i
$$

This simply means that the probability of an instance $x$ belonging to class $i$ is $p_i$.  
All probabilities must satisfy:

$$
\sum_{i=1}^{K} p_i = 1, \quad 0 \le p_i \le 1
$$

---

#### 2. The Softmax Function

The **Softmax** function converts raw scores (logits) into such a valid probability distribution:

$$
p(y = k \mid x; W) = p_k = \frac{e^{x_i^\top w_k}}{\sum_{j=1}^{K} e^{x_i^\top w_j}}
$$

where:
- $x_i$ is the feature vector of the $i$-th sample,  
- $w_k$ is the weight vector for class $k$,  
- and $K$ is the total number of classes.



#### 3. Likelihood of the Dataset

Assuming independence between samples, the likelihood of observing the entire dataset is:

$$
L(W) = \prod_{i=1}^{N} P(y_i \mid x_i, W)
     = \prod_{i=1}^{N} \frac{e^{x_i^\top w_{c_i}}}{\sum_{j=1}^{K} e^{x_i^\top w_j}}
$$

where $c_i$ denotes the correct class for sample $i$.

#### 4. Negative Log-Likelihood (Cost Function)

Taking the negative logarithm gives the **Negative Log-Likelihood (NLL)**:

$$
\begin{array}{rcl}
NLL(W) & = & - \ln L(W) \\[6pt]
& = & - \sum_{i=1}^{N} \ln \left( \frac{e^{x_i^\top w_{c_i}}}{\sum_{j=1}^{K} e^{x_i^\top w_j}} \right) \\[6pt]
& = & - \sum_{i=1}^{N} \left( x_i^\top w_{c_i} - \ln \left( \sum_{j=1}^{K} e^{x_i^\top w_j} \right) \right)
\end{array}
$$

Here:
- $x_i^\top w_{c_i}$ is the **logit score** (before the softmax) for the correct class.  
- The second term, $\ln(\sum_j e^{x_i^\top w_j})$, accounts for the scores of *all* classes and ensures normalization.

#### 5. Matrix Form Implementation

In matrix form, we define:

$$
\mathbf{Z} = \mathbf{XW}
$$

where:
- $\mathbf{X} \in \mathbb{R}^{N \times D}$ is the design matrix (inputs),
- $\mathbf{W} \in \mathbb{R}^{D \times K}$ contains the class weights,
- $\mathbf{Z} \in \mathbb{R}^{N \times K}$ contains all class scores (logits).

To select the correct logit for each observation, we use the **one-hot encoded** label matrix $\mathbf{Y}$,  
where each row corresponds to a sample and has a 1 in the column of the true class:

$$
Y_{ik} =
\begin{cases}
1 & \text{if sample } i \text{ belongs to class } k, \\
0 & \text{otherwise.}
\end{cases}
$$

This allows us to extract all correct logits at once as:

$$
z_{\text{correct}} = \sum_{k=1}^{K} Y_{ik} Z_{ik} = (\mathbf{Y} * \mathbf{Z}) \text{ summed over axis } k.
$$

Thus, the NLL in matrix form becomes:

$$
NLL(W) = - \frac{1}{N} \sum_{i=1}^{N} 
\left( z_{\text{correct}, i} - 
\ln \left( \sum_{k=1}^{K} e^{Z_{ik}} \right) \right)
$$

This is the cost function implemented in code, and minimizing it with respect to $W$ corresponds to **maximizing the log-likelihood** of the categorical model. 

#### Loss function with L2 regularization

Adding L2 regularization remains straight forward. If 

$$
NLL(W) = - \sum_{i=1}^{N} \left( x_i^\top w_{c_i} - \ln \left( \sum_{j=1}^{K} e^{x_i^\top w_j} \right) \right)
$$

Then the negative log likelihood with L2 regularization is simply : 

$$
NLL_{\text{l2}}(W) = NLL(W) + \frac{\lambda}{2}||W||_2^2
$$


### Implementation of the gradient and weight update : TODO

### Fine tuning : TODO