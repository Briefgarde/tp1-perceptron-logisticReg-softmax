# TP1 : Perceptron, Logistic regression, Softmax, without ChatGPT

## Explanation 

In the interest of showing that not all of my text are written by ChatGPT, I've decided to put here my initial writting of the rapport. In lots of ways, this rapport is less good than the main, `rapport.md` that exists alongside it, and I'll personnally be using the `rapport.md` file when studying the models we see here. 

Still, I want to show that I understand what I'm writing, and that ChatGPT is only helping me put on the final layer of pretty writing and correcting minor error that can be tied to inattention. 

Sadly, I did not come up with this idea immediately. This file does not have all of my writing pre-chatGPT enhancement as I hadn't thought of doing something like this. 

Going forward, in the next TPs, I'll aim to have a file similar as this one immediately. 

## Logistic Regression 

### Implementation of the gradient function 

With the cost function defined above, we can now take the derivative in regards to the weights to guide our optimization process. 

$$
E(\mathbf{w}) = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})]
$$

To work out this big derivation, we'll be using the chain, which wants us to break down this big formula into smallers ones that are more easily derivable. I choose to do so in the following way : 

- $h(w) = \mathbf{Xw} = z$
- $g(z) = \sigma(z) = \frac{1}{1+e^{-z}} = \hat{y}$
- $f(\hat{y}) = E(\mathbf{w}) = - \frac{1}{N}[\mathbf{y^T} \ln(\hat{\mathbf{y}}) - (\mathbf{1-y})^T \ln(\mathbf{1}- \hat{\mathbf{y}})]$

With those three functions defined initially, we can start working through the chain rules as follow : 

$$
\nabla_w E(\mathbf{w}) = \frac{df}{d\hat{y}} \cdot \frac{dg}{dz} \cdot \frac{dh}{dw}
$$

Let's now derivate each of the terms of this chain rule, staring with $h(w)$

$$
\begin{array}{rcl}
h(w) & = &\mathbf{Xw} \\
\frac{dh}{dw} & = &\mathbf{X} \\
\end{array}
$$

This one is rather simple, there isn't much to say. 

Moving on to $g(z)$ : 

$$
\begin{array}{rcl}
g(z) & = & \sigma(z) = \frac{1}{1+e^{-z}} = \hat{y} \\
\frac{dg}{dz} & = &  \frac{d}{dz} (1+e^{-z})^{-1}\\
\end{array}
$$

Here, it becomes easier to do a chain rule again. We can define $(1+e^{-z})^{-1}$ as two function like so : $r(z) = 1+e^{-z}$ and $t(u) = t(r(z)) = u^{-1}$. Then, $\frac{d}{dz} (1+e^{-z})^{-1} = t'(r(z)) \cdot r'(z)$ 

So : 
- $t'(u) = -u^{-2} = -(1+e^{-z})^{-2}$
- $r'(z) = 0 - e^{-z} = - e^{-z}$

Combining to make : 

$$
\begin{array}{rcl}
\frac{d}{dz} (1+e^{-z})^{-1} &=&t'(u) \cdot r'(z) \\
&=& (-(1+e^{-z})^{-2}) \cdot (- e^{-z})\\
&=& {e^{-z}/(1 + e^{-z})^2} \\
&=& \frac{e^{-z}}{(1 + e^{-z})^2} \\
\end{array}
$$

We can then do a little rewrite of that last step to arrive to the following, "nicer" result for the derivate of $g(z)$

$$
\begin{array}{rcl}
\frac{e^{-z}}{(1 + e^{-z})^2} & = & \frac{1}{1 + e^{-z}} \cdot (1 - \frac{1}{1 + e^{-z}}) &\\
&=& \sigma(z)(1-\sigma(z)) & \text{with } \sigma(z) = \frac{1}{1 + e^{-z}}
\end{array}
$$

And then, we can take the derivative of $f(\hat{y})$ : 

$$
\frac{d}{d\hat{y}} \cdot f(\hat{y}) = -\left( \frac{\mathbf{y}}{\hat{\mathbf{y}}} - \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} \right)
$$

I'll admit to skipping a few steps here because this derivation is otherwise pretty long, but not particularly difficult. It mostly uses the logarithmic rule $\frac{d}{dx} \left( \ln(x) \right) = \frac{1}{x}$.

We now have all the pieces together to complete the chain rule. Putting it all together, we have the following : 

$$
\begin{array}{rcl}
\nabla_w E(\mathbf{w}) &=& \frac{df}{d\hat{y}} \cdot \frac{dg}{dz} \cdot \frac{dh}{dw}\\
&=& \underbrace{-\left( \frac{\mathbf{y}}{\hat{\mathbf{y}}} - \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} \right)}_{\frac{df}{d\hat{y}}} \underbrace{\frac{1}{1 + e^{-z}} \cdot (1 - \frac{1}{1 + e^{-z}})}_{\frac{dg}{dz}} \underbrace{\mathbf{X}}_{\frac{dh}{dw}}\\
& \text{we can rewrite } \frac{dg}{dz} \text{ in term of } \hat{{\mathbf{y}}}  &&\\
&=&-\left( \frac{\mathbf{y}}{\hat{\mathbf{y}}} - \frac{1 - \mathbf{y}}{1 - \hat{\mathbf{y}}} \right) \hat{\mathbf{y}} \cdot (1 - \hat{\mathbf{y}}) \mathbf{X}\\
& \text{this simplify like so} & \\
&=& -\mathbf{X^T(\hat{y} - y)} \\
& \text{finally adding normalization back : } &\\
&=& -\frac{1}{N} \mathbf{X^T(\hat{y} - y)} \\
\end{array}
$$

### Fine Tuning of the model and analysis : TODO

When testing our multiple different hyperparameter like the learning rate and the strength of the regulation, a few different thing arise that are rather odd. 

Firstly, a lot of combination of parameters end up with surprisingly the exact same accuracy values. For example, a learning rate of 1e-06 and any of 0.0001, 0.01 or 0 regulation end up with the same accuracy (0.651 the last time I ran the code). This hint that, no matter the combination, the model end up in the exact same minima it find, and the hyperparameters do not manage to change the result. 

The second thing that comes to mind is finding of the best regulation strength, which is 10'000. Most of the time, we're used to hyperparameters being rather small, seeing how the learning rate is in the range 0.0000001-0.000000001 for example. Yet, here, the best regulation strength is not just well above 0, but outright 10K ! Such a regulation strength means the model seem to perform the best when weights are *extremely* small. 

At lower learning rate (1e-07, 1e-08), we see that the model barely learn anything comparated to the 1e-06 learning rate. All validation accuracy remain in the zone of ~0.42, with little to no impact from the regulation strength. This seem to hint that a learning rate this small is unable to escape the initial zone of initialization of its weights, and the model is unable to take steps to avoid this trap. 


## Softmax classifier

### Implementation of the loss function 

The process to get the loss function for a softmax classifier is once again very similar to the Logistic Regression (and really most probabilistic models). We are going to use the negative log likelihood to define a cost function for our model. 

Firstly, the Softmax classifier is based on the **Categorical distribution**, which is an extension of the Bernoulli distribution for multiple classes instead of a binary classification. It's defined as : 

$$ 
p(x=i) = p_i 
$$ 

It is, at face value, pretty simple, and that's partially because it is. It tells that the probability of instance $x$ being of class $i$ is equal to $p_i$, which is pretty straightforward. Just like with any PMF, summing over all $i$ will give 1, and each of those $p_i \in {0,1}$. 

We also know that the softmax function we saw earlier can be used to produce the same result. Thus : 

$$ 
p(x=k) = p_k = \frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}}. 
$$ 

With this defined, we can now take the likelihood of the PMF and replace the $p_i$ by the softmax function, like so : 

$$ 
L(W) = \prod_{i=1}^K p_k = \prod_{i=1}^K \frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}} 
$$ 

We can then move on to taking the negative log likelihood of this function to define the classical cost function used often. 

$$ 
\begin{array}{rcl} 
NLL(W) & = &- \ln(\prod_{i=1}^K \frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}}) \\ 
& = & - \sum_{i=1}^K \ln(\frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}}) \\ 
& = & - \sum_{i=1}^K (\ln{e^{x_i w_k}} - \ln{(\sum_{j=1}^{K} e^{x_i w_j})})\\ 
& = & - \sum_{i=1}^K (\mathbf{x_i w_c} - \ln{(\sum_{j=1}^{K} e^{\mathbf{x_i w_j}})}) 
\end{array} 
$$ 

where : 
- $\mathbf{x_i w_c}$ represent the logit score (before being put through the softmax function) for the *correct* class. 

Finally, to put this in matrix form, we do the following : 

$$ 
\begin{array}{rcl} 
NLL(W) & = & - \sum_{i=1}^K (\mathbf{x_i w_c} - \ln{(\sum_{j=1}^{K} e^{\mathbf{x_i w_j}})}) \\ 
& = & - \frac{1}{N} \sum(z_\text{correct} - \ln(\sum_{i=i}^K(e^{z_i}))) 
\end{array} 
$$ 

With : 
- $\mathbf{z = XW}$ 
- $z_\text{correct}$ is the result of multiplying the base $z$ by the One Hot Encoded $\mathbf{y}$ result, thus giving us an easy way to single out the correct logit to use. This One Hot Encoded y can be defined like so : 

$$
Y_{ik} =
\begin{cases}
1 & \text{if sample } i \text{ belongs to class } k, \\
0 & \text{otherwise.}
\end{cases}
$$

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

Defining the gradient remain the same task as always : we use the derivative of the cost function in regards to the weight to define the way we'll move during Gradient descent. 

With the cost function defined as : 

$$
E(W) = NLL(W) = - \sum_{i=1}^{N} \left( x_i^\top w_{c_i} - \ln \left( \sum_{j=1}^{K} e^{x_i^\top w_j} \right) \right)
$$

The gradient can thus be defined like so : 

$$
\begin{array}{rcl}
\nabla_{\mathbf{w}} E(\mathbf{w}) & = & \frac{d(E(w))}{d(w)} \\
& = & \frac{1}{N}\mathbf{X^T (\text{softmax}(XW) - \mathbf{Y_{\text{OHE}}})}
\end{array}
$$

where : 
- $\mathbf{Y_{\text{OHE}}}$ is the $\mathbf{y}$ vector turned into a sparse matrix, as defined earlier. 

### Fine tuning : TODO

