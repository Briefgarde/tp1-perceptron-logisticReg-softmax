# TP1 : Perceptron, Logistic regression, Softmax, without ChatGPT

## Explanation 

In the interest of showing that not all of my text are written by ChatGPT, I've decided to put here my initial writting of the rapport. In lots of ways, this rapport is less good than the main, `rapport.md` that exists alongside it, and I'll personnally be using the `rapport.md` file when studying the models we see here. 

Still, I want to show that I understand what I'm writing, and that ChatGPT is only helping me put on the final layer of pretty writing and correcting minor error that can be tied to inattention. 

Sadly, I did not come up with this idea immediately. This file does not have all of my writing pre-chatGPT enhancement, as I only began doing it during the softmax loss function question. 

Going forward, in the next TPs, I'll aim to have a file similar as this one immediately. 

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

$$ L(W) = \prod_{i=1}^K p_k = \prod_{i=1}^K \frac{e^{x_i w_k}}{\sum_{j=1}^{K} e^{x_i w_j}} 
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
- $z_\text{correct}$ is the result of multiplying the base $z$ by the One Hot Encoded $\mathbf{y}$ result, thus giving us an easy way to single out the correct logit to use. It can be defined like so : 

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

### Fine tuning : TODO

