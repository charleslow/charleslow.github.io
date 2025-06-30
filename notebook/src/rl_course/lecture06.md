# Value Function Approximation

[Lecture 6: Value Function Approximation](https://www.youtube.com/watch?v=UoPei5o4fps&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=6)

This lecture looks into approximating functions with neural networks to overcome the large state-action space problem.

RL often encounters large problems:
- Backgammon: $10^{20}$ states
- Go: $10^{170}$ states
- Helicopter: continuous state space

We want to do policy evaluation and control efficiently in large state spaces. So far, we have represented $V$ or $Q$ with a lookup table:
- Every state $s$ has an entry $V(s)$
- Every state-action pair $s, a$ has an entry $Q(s, a)$

This is a problem for large MDPs:
- Too many states or actions to store in memory
- It is too slow or data inefficient to learn the value of each state individually

Solution for large MDPs:
- Estimate the value function with function approximation using parameters $w$:
    - $\hat{v}(s, w) \approx v_\pi(s)$
    - $\hat{q}(s, a, w) \approx q_\pi(s, a)$
- Generalizes from seen states to unseen states
- Update parameters of our function using MC or TD learning

Types of value function approximation (different architectures):
- Represent a given state $s$ with some parameters $w$. Then neural network spits out <<$\hat{v}(s, w)$>>, which is our value function for being in state $s$
- Have a neural network <<$\hat{q}(s, a, w)$>>, which takes in a state-action pair and spits out the Q value
- Sometimes, it is more efficient to have a neural network <<$\hat{q}(s, w)$>>, such that we feed in a single state and we get Q-values for every possible action in a single forward pass, i.e. we get $\hat{q}(s, a_1, w), \hat{q}(s, a_2, w), ...$

Which function approximator? We focus on <<differentiable>> function approximators that we can easily optimize, i.e. Linear combinations of features, neural networks. Furthermore, we want a training algorithm for a <<non-iid, non-stationary>> set of data, so it is not standard supervised learning.

## Incremental Methods

### Gradient Descent

Starting with gradient descent.
- Let $J(w)$ be a differentiable function of parameter vector $w$
- Define the gradient of $J(w)$ to be a vector $\nabla_w J(w)$ where $\nabla_w J(w)[0]$ is $\frac{\partial J(w)}{\partial w_1}$
- To find the local minimum of $J(w)$, we adjust the parameter $w$ in the -ve gradient direction:
$$
    \triangle w = -\frac{1}{2} \alpha \nabla_w J(w)
$$

Goal: find parameter vector $w$ minimizing mean squared error between approximate value fn $\hat{v}(s, w)$ and true oracle value fn $v_\pi(s)$ (assuming we know the oracle)
$$
    J(w) = \E_\pi[(v_\pi(S) - \hat{v}(S, w)^2)]
$$

Gradient descent finds a local minimum:
$$
\begin{align*}
    \triangle w &= -\frac{1}{2} \alpha \nabla_w J(w) \\
    &= \alpha \E_\pi [(v_\pi(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)]
\end{align*}
$$

Stochastic gradient descent samples the gradient:
$$
    \triangle w 
    = \alpha (v_\pi(S) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)
$$

The nice thing about SGD is that it still converges under non-stationary environment. The expected update is equal to full gradient update.

### Feature Vectors

To represent a state, we use a feature vector.
$$
\mathbf{x}(S) = 
\begin{pmatrix}
  \mathbf{x}_1(S) \\
  \vdots        \\
  \mathbf{x}_n(S)
\end{pmatrix}
$$

For example, the features (numeric) could be:
- Distance of robot to landmarks
- Trends in the stock market
- Configuration of pawn on a chess board

### Linear Value Function Approximation

Let us represent the value function using a linear combination of features (i.e. just a dot product between two vectors):
$$
    \hat{v}(S, w) = \mathbf{x}(S)^T w = \sum_{j=1}^n \mathbf{x}_j (S) w_j
$$

The nice thing is that linear approximator is quadratic in the parameters $w$, so it is a convex optimization problem, i.e. SGD will converge on the global optimum:
$$
    J(w) = \E_\pi \left[
        (v_\pi(S) - \mathbf{x}(S)^T w)^2
    \right]
$$

The gradient update is really simple:
$$
\begin{align*}
    \nabla_w \hat{v}(S, w) &= \mathbf{x}(S)\\
    \triangle w &= \alpha \left( v_\pi(S) - \hat{v}(S, w) \right) \mathbf{x}(S)
\end{align*}
$$

Note that we are just subbing the simple expression for $\nabla_w \hat{v}(S, w)$ into the general $\triangle w$ formula above. The update may be interpreted as `step-size x prediction error x feature value`. This means that features with high correlation with the prediction error will have large gradient updates intuitively.

We can think of table lookup as a special case of linear value function approximation. Suppose we use a table lookup feature (1-hot) as follows:
$$
\mathbf{x}^{table}(S) = 
\begin{pmatrix}
    \mathbf{1}(S = s_1)\\
    \vdots \\
    \mathbf{1}(S = s_n)
\end{pmatrix}
$$

And suppose we have a parameter vector of size $n$, such that we have one parameter for each state. Then we have:
$$
\hat{v}(S, w)
=
\begin{pmatrix}
    \mathbf{1}(S = s_1)\\
    \vdots \\
    \mathbf{1}(S = s_n)
\end{pmatrix}
\cdot
\begin{pmatrix}
    w_1\\
    \vdots\\
    w_n
\end{pmatrix}
$$

And we can see that this reduces to a table lookup where the parameter $w_j$ represents the state value for each state $j$.

### Estimating the Oracle

So far, we have assumed that the true oracle value function $v_\pi(s)$ is available, but in RL there is no true label, only rewards. So in practice, we need to substitute a target for $v_\pi(s)$:
- For MC, the target is the return $G_t$:
$$
    \triangle w = \alpha(G_t - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w)
$$ 
- For TD(0), the target is the TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$:
$$
    \triangle w = \alpha(R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w))\nabla_w \hat{v}(S_t, w)
$$
- For TD($\lambda$), the target is the $\lambda$-return $G_t^{\lambda}$:
$$
    \triangle w = \alpha (G_t^\lambda - \hat{v}(S_t, w))\nabla_w \hat{v}(S_t, w)
$$

### Monte Carlo with Value Function Approximation

We can think of our algorithm as supervised learning.
- Treat the return $G_t$ as an unbiased noisy sample of the true value $v_\pi(S_t)$
- We therefore are applying supervised learning to "training data":
$$
    <S_t, G_1>, <S_2, G_2>, ..., <S_T, G_T>
$$
- For example, using linear MC policy evaluation:
$$
\begin{align*}
    \triangle w &= \alpha(G_t - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w)\\
    &= \alpha(G_t - \hat{v}(S_t, w)) x(S_t)
\end{align*}
$$
- MC evaluation converges to a local optimum even when using non-linear value function approximation

### TD with Value Function Approximation

The same applies to TD learning, but we have some biased estimate:
- The TD target $R_{t+1} + \gamma \hat{v}(S_{t+1}, w)$ is a biased sample of the true value $v_\pi(S_t)$ - it's biased because our own value function is a biased estimate
- We can still apply supervised learning to the "training data":
$$
    <S_1, R_2 + \gamma \hat{v}(S_2, w)>,\\
    <S_2, R_3 + \gamma \hat{v}(S_3, w)>,\\
    ...,\\
    <S_{t-1}, R_T>
$$
- For example using linear TD(0):
$$
\begin{align*}
    \triangle w &= \alpha(R + \gamma \hat{v}(S', w) - \hat{v}(S, w)) \nabla_w \hat{v}(S, w)\\
    &= \alpha \delta x(S)
\end{align*}
$$
- There is a theorem showing that for linear TD(0), despite the bias, we will always converge (close) to the global optimum

### TD($\lambda$) with Value Function Approximation

And again, we can do the same with TD-$\lambda$, since the $\lambda$-return $G_t^\lambda$ is also a biased sample of the true value $v_\pi(s)$:
- The training data is now:
$$
    <S_1, G_1^\lambda>, <S_2, G_2^\lambda>, ..., <S_{T-1}, G_{T-1}^\lambda>
$$
- The forward view linear TD($\lambda$) is:
$$
\begin{align*}
    \triangle w &= \alpha(G_t^\lambda - \hat{v}(S_t, w)) \nabla_w \hat{v}(S_t, w)\\
    &= \alpha(G_t^\lambda - \hat{v}(S_t, w)) x(S_t)
\end{align*}
$$
- The backward view linear TD($\lambda$) is:
$$
\begin{align*}
    \delta_t &= R_{t+1} + \gamma \hat{v}(S_{t+1}, w) - \hat{v}(S_t, w)\\
    E_t &= \gamma \lambda E_{t-1} + x(S_t)\\
    \triangle w &= \alpha \delta_t E_t\\
\end{align*}
$$
- There is a theorem to show that the forward view and backward view linear TD($\lambda$) are equivalent.

For the backward view, notice that the eligibility trace is now updated using the gradient wrt the parameter vector, namely $\nabla_w \hat{v}(S_t, w)$, which is of the same dimensionality as $w$. More precisely, the eligibility trace is the decaying accumulation of past gradients. In the linear case, this is an accumulation of the feature vector $x(S_t)$. 

It is a bit unintuitive to understand why we use the accumulated gradient as the eligibility trace, but I suppose it is proved in the equivalence theorem between the forward and backward view.


