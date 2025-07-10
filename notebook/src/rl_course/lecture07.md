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
- Effective in high dimensional or continuous action spaces
    - We do not need to compute max action over Q-values
    - If the action space is continuous, the maximization is not straightforward at all
- Can learn stochastic policies

Disadvantages of Policy-base RL
- Typically converges to a local rather than global optimum
- Evaluating a policy is typically inefficient and high variance

Why might we want to have a stochastic policy?
- e.g. Rock paper scissors
- Having a deterministic policy is easily exploited
- A uniform random policy is optimal