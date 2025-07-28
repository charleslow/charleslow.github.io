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
- Likelihood ratio methods exploit the following identity (call it the <<log trick>>):
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

## Gaussian Policy

In continuous action spaces, a Gaussian policy is natural
- Let the mean of the gaussian be a linear combination of state features
$$
    \mu(s) = \phi(s)^\intercal \theta
$$
- The variance may be fixed $\sigma^2$ or parametrized
- The policy is gaussian (recall that we are in a continuous action space, so $a$ is a vector of floats):
$$
    a \sim \mathcal{N} \left( \mu(s), \sigma^2 \right)
$$
- The score function is then:
$$
    \nabla_\theta \log \pi_\theta(s, a) = \frac{
        (a - \mu(s)) \phi(s)
    }{
        \sigma^2
    }
$$
- We can probably derive this score function by writing down the PDF of the gaussian distribution for $\pi_\theta(s, a)$ and then taking the derivative

## Policy Gradient Theorem: One-Step MDPs

Consider a simple class of one-step MDPs to simplify the math
- Start in a state $s \sim d(s)$
- Terminate after one step with reward $r = \mathcal{R}_{s,a}$
- This is a sort of contextual bandit

Use likelihood ratios to compute the policy gradient
- First we pick our objective function, which is just the expected reward (averaged over our start state and action that we choose)
$$
\begin{align*}
    J(\theta) &= \E_{\pi_\theta} [r] \\
    &= \sum_{s \in \S} d(s) \sum_{a \in \A} \pi_\theta(s, a) \mathcal{R}_{s,a}
\end{align*}
$$
- Then we take the derivative to do gradient ascent:
$$
\begin{align*}
    \nabla_\theta J(\theta) 
    &=
    \sum_{s \in \S} d(s) \sum_{a \in \A} \nabla_\theta \pi_\theta(s, a) \mathcal{R}_{s,a}\\
    &=
    \sum_{s \in \S} d(s) \sum_{a \in \A} \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a) \mathcal{R}_{s,a}\\
    &=
    \E_{\pi_\theta} \left[
        \nabla_\theta \log \pi_\theta(s, a) r
    \right]
\end{align*}
$$
- Note that when taking the gradient of $\pi_\theta$, we use the log trick to rewrite it in line 2, and it becomes a new expectation again because we recover $\pi_\theta(s, a)$ outside of the gradient. This shows the power of the log trick.

## Policy Gradient Theorem

But we don't just want to do one-step MDPs, we want to generalize to multi-step MDPs
- It turns out that we just need to replace the instantaneous reward $r$ with long term value $Q^\pi(s, a)$ (I suppose this means we need to model $Q$ as well)
- Regardless of whether we use the (i) start state objective, (ii) average value objective or (iii) average reward objective, the policy gradient theorem hold

> **Theorem**. Policy Gradient Theorem.
> 
> For any differentiable policy $\pi_\theta(s, a)$, and for any of the policy objective functions $J_1$, $J_\text{average value}$ or $J_\text{average reward}$, the policy gradient is:
$$
    \nabla_\theta J(\theta) = \E_{\pi_\theta} [
        \nabla_\theta \log \pi_\theta(s, a) Q^{\pi_\theta}(s, a)
    ]
$$

## Monte Carlo Policy Gradient (REINFORCE)

The policy gradient theorem basically gives rise to a simple monte carlo policy gradient algorithm to find the optimal policy:

> **REINFORCE Algorithm**.
> 
> - Initialize $\theta$ randomly
> - **For** each episode $\{ s_1, a_1, r_2, ..., s_{T-1}, a_{T_1}, r_T \} \sim \pi_\theta$ do:
>   - **For** $t=1$ to $T-1$ do:
>       - $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi_\theta(s_t, a_t) G_t$
> - Return $\theta$

Note that: 
- We are doing monte carlo, i.e. we wait until the end of the episode before we go back to update the parameters for each time step.
- We are doing SGD, so there is no expectation term
- We use the return $G_t$ as an unbiased sample of $Q^{\pi_\theta}(s_t, a_t)$. Recall that $G_t$ is the total discounted reward from time step $t$ until termination.
- This is the simplest and oldest policy gradient algorithm.

Empirically, policy gradient methods have a nice learning curve without the jittery behaviour of value based methods. But, monte carlo methods take very very long (millions of steps) to converge due to high variance.

## Actor Critic Policy Gradient

The main problem with monte carlo policy gradient is the high variance of the return $G_t$. Sometimes we get no reward, sometimes we get high reward.

The idea is thus to use a <<critic>> to estimate the action-value function $Q$:
$$
    Q_w(s, a) \approx Q^{\pi_\theta}(s, a)
$$

The name <<critic>> refers to the value function, which simply "watches" and evaluates the value of an action, whilst the <<actor>> is the policy itself which decides how we should act.

We maintain two sets of parameters:
- **Critic** updates the action value function parameters $w$
- **Actor** updates the policy parameters $\theta$, in the direction suggested by the critic

Actor-critic algorithms follow an approximate policy gradient:
$$
\begin{align*}
    \nabla_\theta J(\theta) &\approx \E_{\pi_\theta}[
        \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)
    ]\\
    \triangle \theta &= \alpha \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)
\end{align*}
$$

Notice that we just replace $G_t$ with $Q_w(s,a)$, which is the value function of the critic model.

## Estimating the Action-Value Function of the Critic

The critic is solving a familiar problem: policy evaluation. What is the value for policy $\pi_\theta$ for current parameters $\theta$? We have explored this previously, i.e.:
- Monte Carlo policy evaluation
- TD-$\lambda$
- or least squares policy evaluation

This leads us to a simple actor-critic algorithm:
- <<Critic>>: use linear value function approximation $Q_w(s, a) = \phi(s, a)^\intercal w$
    - Update $w$ using linear TD(0)
- <<Actor>>: update $\theta$ using policy gradient

> **Q Actor Critic (QAC) Algorithm**.
> 
> - Initialize $s, \theta$
> - Sample $a \sim \pi_\theta$
> - for each step:
>   - Sample reward $r = \mathcal{R}_s^a$; sample transitions $s' \sim \mathcal{P}^a_s$
>   - Sample action $a' \sim \pi_\theta(s', a')$
>   - $\delta = r + \gamma Q_w(s', a') - Q_w(s, a)$
>   - $\theta = \theta + \alpha \nabla_\theta \log \pi_\theta(s, a) Q_w(s, a)$
>   - $w \leftarrow w + \beta \delta \phi(s, a)$
>   - $a \leftarrow a', s \leftarrow s'$

Note that the $\delta$ line is simply the TD(0) update, and the $\theta$ line is simply the policy gradient step.

## Bias in Actor Critic Algorithms

The problem is that approximating the policy gradient introduces bias
- A biased policy gradient may not find the right solution
- Luckily, if we choose the value function approximation carefully, we can avoid introducing any bias
- i.e. we can still follow the exact policy gradient (see Compatible Function Approximation theorem)

## Reducing Variance using a Baseline

Here we move on to tricks to improve the training algorithm. The baseline method is the best known trick.

The main idea is to subtract a baseline function $B(s)$ from the policy gradient, such that we can *reduce variance without changing expectation*. 

The reason is that since $B(s)$ does not depend on the action $a$, its expectation when plugged in will be $0$, like so:
$$
\begin{align*}
    \E_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) B(s)]
    &=  \sum_{s \in \S} d^{\pi_\theta(s)} \sum_{a \in \A} \nabla_\theta \pi_\theta(s, a) B(s)\\
    &= \sum_{s \in \S} d^{\pi_\theta}B(s) \nabla_\theta \sum_{a \in \A} \pi_\theta(s, a)\\
    &= 0
\end{align*}
$$

Note that in line 2, since $\pi_\theta$ is a probability, the right-most term sums to $1$, which is a constant. Hence the gradient $\nabla_\theta$ will become $0$, and the expectation resolves to $0$.

A good and popular choice of baseline is the *state* value function $B(s) = V^{\pi_\theta}(s)$.
- So we can rewrite the policy gradient using the <<advantage function>> $A^{\pi_\theta}(s, a)$:
$$
\begin{align*}
    A^{\pi_\theta}(s, a) &= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)\\
    \nabla_\theta J(\theta) &= \E_{\pi_\theta}[
        \nabla_\theta \log \pi_\theta(s, a) A^{\pi_\theta}(s, a)
    ]
\end{align*}
$$
- Intuitively, the advantage function tells us the additional benefit of an action over the baseline value of being in that state

## Estimating the Advantage Function

The advantage function can significantly reduce the variance of the policy gradient. The naive way is to estimate $V^{\pi_\theta}(s)$ and $Q^{\pi_\theta}(s, a)$ separately, i.e.:
$$
\begin{align*}
    \hat{V}(s) &\approx V^{\pi_\theta}(s)\\
    Q_w(s, a) &\approx Q^{\pi_\theta}(s, a)\\
    A(s, a) &= Q_w(s,a) - \hat{V}(s)
\end{align*}
$$

Then, we use TD methods to update both value functions. However, this is not efficient in terms of parameters.

The better way is to observe that the TD error $\delta^{\pi_\theta}$ is an unbiased estimate of the advantage function. Therefore, we can plug in the TD error in our policy gradient update instead.
- Recall that the TD error is 
$$\delta^{\pi_\theta} = r + \gamma V^{\pi_\theta}(s') - V^{\pi_\theta}(s)$$
- And this is an unbiased estimate of the advantage function
$$
\begin{align*}
    \E_{\pi_\theta}[\delta^{\pi_\theta} | s, a] &=
    \E_{\pi_\theta}[r + \gamma V^{\pi_\theta}(s') | s, a] - V^{\pi_\theta}(s)\\
    &= Q^{\pi_\theta}(s, a) - V^{\pi_\theta}(s)\\
    &= A^{\pi_\theta}(s, a)
\end{align*}
$$
- Note that we have not taken any approximations above. We just showed that the expectation of the TD(0) error when following policy $\pi_\theta$ corresponds to the advantage function. Interestingly, computing the TD error does not require estimating the $Q$ function, only the $V$ function. This gives us a simpler update
- We can thus use the TD error to compute the policy gradient
$$
    \nabla_\theta J(\theta) =
    \E_{\pi_\theta}[\nabla_\theta \log \pi_\theta (s, a) \delta^{\pi_\theta}]
$$
- In practice, since we do not have the true value function $V^{\pi_\theta}$, we use an estimate $\hat{V}$:
$$
    \delta_w = r + \gamma \hat{V_w}(s') - \hat{V_w}(s)
$$
- This approach only requires one set of critic parameters $w$

## Improving the Actor Updates

Recall that we can estimate the value function $V_\theta(s)$ from targets at different time scales to trade-off bias and variance:
- For MC, the target is the return $G_t$
$$
    \triangle \theta = \alpha (\textcolor{orange}{G_t} - V_\theta(s))\phi(s)
$$
- For TD(0), the target is the TD target
$$
    \triangle \theta = \alpha(\textcolor{orange}{r + \gamma V_\theta(s')} - V_\theta(s))\phi(s)
$$
- For forward view TD($\lambda$), the target is the lambda return
$$
    \triangle \theta = \alpha(\textcolor{orange}{v_t^\lambda} - V_\theta(s))\phi(s)
$$
- For backward view TD($\lambda$), we use eligibility traces
$$
\begin{align*}
    \delta_t &= r_{t+1} + \gamma V(s_{t+1}) - V(s_t)\\
    e_t &= \lambda e_{t-1} + \phi(s_t)\\
    \triangle \theta &= \alpha \delta_t e_t
\end{align*}
$$

Similarly, we can estimate the policy gradient for the actor at many time scales. The main target we want to estimate is
$$
    \nabla_\theta J(\theta) = \E_{\pi_\theta}[
        \nabla_\theta \log \pi_\theta(s, a) A^{\pi_\theta}(s, a)
    ]
$$
- MC policy gradient uses the error from the return
$$
    \triangle \theta = \alpha(\textcolor{orange}{G_t} - \hat{V}(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$
- Actor critic policy gradient uses the one-step TD error
$$
    \triangle \theta = \alpha(\textcolor{orange}{r + \gamma \hat{V}(s_{t+1})} - \hat{V}(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$
- Forward view TD($\lambda$) uses the $\lambda$ target
$$
    \triangle \theta = \alpha(\textcolor{orange}{v_t^\lambda} - \hat{V}(s_t)) \nabla_\theta \log \pi_\theta(s_t, a_t)
$$
- Backward view TD($\lambda$) uses eligibility traces. Note that the eligbility update now uses the score function instead of $\phi$ which was the feature function
$$
\begin{align*}
    \delta_t &= r_{t+1} + \gamma\hat{V}(s_{t+1}) - \hat{V}(s_t)\\
    e_{t+1} &= \lambda e_{t} + \nabla_\theta \log \pi_\theta(s, a)\\
    \triangle \theta &= \alpha \delta_t e_t
\end{align*}
$$
- The update can be performed online, unlike MC policy gradient

## Natural Policy Gradient

The natural policy gradient is a parametrization independent approach. It finds ascent direction that is closest to vanilla gradient, when changing the policy by a small, fixed amount
$$
    \nabla_\theta^{\text{nat}} \pi_\theta(s, a) = G_\theta^{-1} \nabla_\theta \pi_\theta (s, a)
$$
- Where $G_\theta$ is the fisher information matrix
$$
    G_\theta = \E_{\pi_\theta}\left[
        \nabla_\theta \log \pi_\theta(s, a) \nabla_\theta \log \pi_\theta(s, a)^\intercal
    \right]
$$
- Notice that this obviates the need for a critic, as $G_\theta$ is based on the actor itself

## Summary
The policy gradient has many equivalent forms
$$
\begin{align*}
    \nabla_\theta J(\theta) 
    &= 
        \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \textcolor{orange}{G_t}] && \text{REINFORCE} \\
    &= 
        \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \textcolor{orange}{Q_w(s, a)}] && \text{Q Actor-Critic} \\
    &= 
        \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \textcolor{orange}{A_w(s, a)}] && \text{Advantage Actor-Critic} \\
    &= 
        \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \textcolor{orange}{\delta}] && \text{TD Actor-Critic} \\
    &= 
        \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(s, a) \textcolor{orange}{\delta e}] && \text{TD}(\lambda) \text{ Actor-Critic} \\[2ex]
    G_\theta^{-1} \nabla_\theta J(\theta) &= w && \text{Natural Actor-Critic}
\end{align*}
$$

Each formulation leads to a different SGD algorithm. We can learn the critic using policy evaluation (e.g. MC or TD learning) to estimate $Q_\pi, A_\pi$ or $V_\pi$.
