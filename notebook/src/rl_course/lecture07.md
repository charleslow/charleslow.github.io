# Policy Gradient

[Lecture 7: Policy Gradient](https://www.youtube.com/watch?v=KHZVXao4qXs&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=7)

Look at methods that update the policy directly, instead of working with value / action-value functions. 

- In the last lecture we approximated the value of action value function using parameters $\theta$.
$$
\begin{align*}
    V_\theta(s) &\approx V^\pi(s)\\
    Q_\theta(s, a) &\approx Q^\pi(s, a)
\end{align*}
$$
- We generated a policy directly from the value function, e.g. using $\epsilon$-greedy algorithm
- In this lecture, we will directly parametrise the policy:
$$
    \pi_\theta(s, a) = \mathcal{P}[a | s, \theta]
$$
- Again, we will focus on model free reinforcement learning

## Value-based vs Policy-based RL

- Value Based
    - Learn the value function
    - Policy is implicit from the value function (e.g. $\epsilon$-greedy)
- Policy Based
    - No value function
    - Learn policy directly
- Actor-Critic
    - Learn a value function
    - Also learn a policy

Advantages of Policy-based RL
- Better convergence properties than value based
    - Value based sometimes swings or chatters around the optimum and do not converge
- Effective in high dimensional or continuous action spaces
    - We do not need to compute max action over Q-values
    - E.g. if the action space is continuous, the maximization is not at all trivial and may be prohibitive
    - This may be the main impediment to value-based RL
- Can learn stochastic policies

Disadvantages of Policy-base RL
- Typically converges to a local rather than global optimum
- Evaluating a policy is typically inefficient and high variance

Why might we want to have a stochastic policy?
- e.g. Rock paper scissors
- Having a deterministic policy is easily exploited
- A uniform random policy is optimal

Stochastic policy is also necessary in a case of state-aliasing, in which our state representation cannot exhaustively differentiate states from each other. In this case, we do not have a Markov Decision Process, and there may not exist a deterministic policy that is optimal. That is why we need a stochastic policy.

## Policy Objective Functions

Our goal in policy gradient is to find the best $\theta$ for a given policy $\pi_\theta(s, a)$ with parameters $\theta$. But how do we measure the quality of a given policy?

There are 3 ways of measuring:
- In episodic environments we can use the start value:
$$
    J_1(\theta) = V^{\pi_\theta}(s_1) = \E_{\pi_\theta}[v_1]
$$
- In continuing environments we can use the average value:
$$
    J_\text{average value}(\theta) = \sum_s d^{\pi_\theta}(s) V^{\pi_\theta}(s)
$$
- Or the average reward per time step
$$
    J_\text{average reward}(\theta) = \sum_s d^{\pi_\theta}(s) \sum_a \pi_\theta(s, a) \mathcal{R}^a_s
$$

In the above, $d^{\pi_\theta}(s)$ is the stationary distribution of the markov chain for $\pi_\theta$. It tells us the amount of time we spend in each state, and so it provides the weighting required to get the average value or reward.

## Policy Optimization

Policy based reinforcement learning is an optimisation problem:
- Gradient free methods:
    - Hill climbing
    - Simplex / amoeba / Nelder Mead
    - Genetic algorithms
- Gradient methods are almost always more efficient:
    - Gradient descent
    - Conjugate gradient
    - Quasi Newton

## Finite Difference Policy Gradient

- Let $J(\theta)$ be any policy objective function
- Policy gradient algorithms search for a local maximum in $J(\theta)$ by ascending the gradient of the policy wrt parameters $\theta$
$$
    \triangle \theta = \alpha \nabla_\theta J(\theta)
$$
- Where $\nabla_\theta J(\theta)$ is the policy gradient (a vector of partial derivatives along each dimension)
$$
    \nabla_\theta J(\theta) = 
\begin{pmatrix}
    \frac{\partial J(\theta)}{\partial \theta_1}\\
    \vdots\\
    \frac{\partial J(\theta)}{\partial \theta_n}\\
\end{pmatrix}
$$

The simplest way to compute the policy gradient is to use finite differences:
- For each dimension $k \in [1, n]$:
    - Estimate the kth partial derivative of objective function wrt $\theta$
    - By perturbing $\theta$ by a small amount $\epsilon$ in the kth dimension
    $$
        \frac{\partial J(\theta)}{\partial \theta_k} \approx 
        \frac{J(\theta + \epsilon u_k) - J(\theta)}{}
    $$
    - Where $u_k$ is a unit vector with 1 in the kth component and 0 elsewhere
- This is not the most efficient algorithm, as it requires $n$ evaluations (once for each dimension) to compute a single gradient step
- It is simple, noisy and inefficient, but sometimes works
- Works for arbitrary policies, even if the policy is not differentiable

## Likelihood Ratios

We now want to compute the policy gradient analytically, assuming:
- The policy $\pi_\theta$ is differentiable whenever it is non-zero; and
- We know the gradient $\nabla_\theta \pi_\theta (s, a)$
- Likelihood ratio methods exploit the following identity:
$$
\begin{align*}
    \nabla_\theta \pi_\theta(s, a) &= 
    \pi_\theta(s, a) \frac{\nabla_\theta \pi_\theta (s, a)}{\pi_\theta (s, a)} \\
    &= \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a)
\end{align*}
$$
- Note that we use the simple identity $\partial_\theta \log f(\theta) =  \frac{\partial_\theta f(\theta)}{ f(\theta)}$
- The new formulation is nicer to work with because we have $\pi_\theta(s, a)$ on the left, which when integrated over, basically gives us the expectation over our policy $\pi_\theta$
    - This allows us to basically sample trajectories from the data and compute the gradient at each step
- The <<score function>> is the quantity $\nabla_\theta \log \pi_\theta(s, a)$ 

## Linear Softmax Policy

Use the softmax policy as a simple running example:
- Weight actions using linear combination of features $\phi(s, a)^\intercal \theta$
- The probability of action is then proportional to the exponentiated weight:
$$
    \pi_\theta(s, a) \propto e^{\phi(s, a)^\intercal \theta}
$$
- The score function is then:
$$
    \nabla_\theta \log \pi_\theta(s, a) = \phi(s, a) - \E_{\pi_\theta} [\phi(s, \cdot)]
$$
- Note that we are omitting the derivation of the second term of the score function which is a bit more involved, as it involves differentiating the normalization factor (not shown above)

## 