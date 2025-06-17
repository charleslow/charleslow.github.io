# Model Free Prediction

[Lecture 4: Model Free Prediction](https://www.youtube.com/watch?v=PnHCvfgC_ZA&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=4)

Methods in which no one tells us the environment, as opposed to before.

## Introduction

- Lecture 3: planning by dynamic programming, in which we solve a *known* MDP
- Lecture 4: Model-free prediction, in which we estimate the value function of an unknown MDP
    - Unknown in the sense that we do not have access to the environment, only the interactions that the agent has with the environment
- Lecture 5: Model-free control (or optimization), in which we optimize the value function of an unknown MDP

## Monte Carlo Reinforcement Learning

MC methods learn directly from episodes of experience.
- It is model-free in that there is no knowledge of MDP transitions / rewards.
- It only learns from *complete* episodes
- MC uses the simplest idea: the value function of a state is the average return from that state over many many runs
- Downside: MC can only be applied to episodic MDPs
    - Episodic meaning that all episodes must terminate and we get a return value

The goal of MC Policy Evaluation is to learn $v_\pi$ from episodes of experience under policy $\pi$:
- An episode: $S_1, A_1, R_2, ..., S_k \sim \pi$

Recall that:
- The return is the total discounted reward:
$$
    G_t = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1}R_T
$$
- And the value function is the expected return given that we start at a given state:
$$
    v_\pi(s) = \E_\pi [G_t | S_t = s]
$$

So the whole idea of MC policy evaluation is to replace the expected return function with the <<empirical mean return>> from observing many many episodes.

There are two main methods of performing this:
- First visit MC policy evaluation
- Every visit MC policy evaluation

<<Method 1: First visit MC policy evaluation>>. Algorithm for evaluating a given state $s$ is:
- At the <<first>> time step $t$ where state $s$ is visited in a given episode:
    - Increment counter $N(s) \leftarrow N(s) + 1$. $N(s)$ is the count of episodes where $s$ was visited.
    - Increment total return $S(s) \leftarrow S(s) + G_t$
    - Value is estimated by mean return $V(s) = S(s) / N(s)$
- By the law of large numbers, $V(s) \rightarrow v_\pi(s)$ as $N(s) \rightarrow \infty$

Note that $G_t$ above is the total discounted reward from time step $t$ onwards. 

<<Method 2: Every visit MC policy evaluation>>. The algorithm is identical to first visit, with the only difference in that we perform the increment step <<every time>> we visit state $s$.

### BlackJack example
- There are `200` unique states:
    - Current sum (`12` to `21`). If it's `11` or below, the action is automatically to `twist`.
    - Dealer's showing card `ace to 10`
    - Do I have a "useable" ace?
- Actions:
    - `stick`: stop receiving cards
    - `twist`: take another card.
- Reward for `stick`:
    - `+1` if `our sum > dealer's sum`
    - `0` if `our sum = dealer's sum`
    - `-1` if `our sum < dealer's sum`
- Reward for `twist`:
    - `-1` if `our sum > 21`
    - `0` otherwise

We can use MC policy evaluation algorithm to play `10,000` episodes of blackjack and compute the value function of each state, under a given policy. For e.g. a naive policy is to `stick` if `our sum >= 20`, otherwise `twist`.

### Incremental Mean

The mean of a sequence of values can be computed in an incremental algorithm:
$$
\begin{align*}
    \mu_k 
    &= \frac{1}{k} \sum_{j=1}^k x_j \\
    &= \frac{1}{k} \left(x_k + \sum_{j=1}^{k-1} x_j\right)  \\
    &= \frac{1}{k} \left(x_k + (k-1) \mu_{k-1} \right) \\
    &= \mu_{k-1} + \frac{1}{k} \left(x_k - \mu_{k-1} \right) \\
\end{align*}
$$

The last line shows that at each step, we just need to adjust the running mean $\mu$ by a small quantity, which is the difference between the new observed value $x_k$ and the current mean $\mu_{k-1}$. This is analogous to a gradient update.

So applying this incremental mean algorithm to monte carlo updates. Recall that the value function $V(s)$ is the mean return over episodes. Hence we can change the above MC algorithm to an incremental mean update:
- After observing a given episode $S_1, A_1, R_2, ..., S_T$:
    - For each state $S_t$ with return $G_t$:
    $$
    \begin{align*}
        N(S_t) &\leftarrow N(S_t) + 1 \\
        V(S_t) &\leftarrow V(S_t) + \frac{1}{N(S_t)} \left( 
            G_t - V(S_t)
        \right) \\
    \end{align*}
    $$
    - We may even replace the running count $N(S_t)$ with a fixed step size $\alpha$. This is the usual approach in non-stationary problem. This algorithm allows us to avoid keeping track of old episodes and just keep updating $V(S_t)$.
    $$
        V(S_t) \leftarrow V(S_t) + \alpha \left( 
            G_t - V(S_t)
        \right)
    $$

## Temporal Difference Learning

TD methods are different from MC methods, in that we do not need to wait for full episodes to learn.
- TD methods, like MC methods, learn directly from episodes of experience
- TD methods, like MC methods, are also model-free
- TD learns from incomplete episodes using <<bootstrapping>>
- TD updates a guess towards a guess

Goal remains the same: learn $v_\pi$ online from experience under policy $\pi$.

Simplest temporal difference learning algorithm: $TD(0)$.
- Update value $V(S_t)$ toward estimated return $R_{t+1} + \gamma V(S_{t+1})$
$$
    V(S_t) \leftarrow V(S_t) + \alpha (R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$
- $R_{t+1} + \gamma V(S_{t+1})$ is called the TD target
-$R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ is called the TD error 

Contrast this with incremental every-visit Monte Carlo which we saw earlier:
- Update value $V(S_t)$ toward actual return $G_t$
$$
    V(S_t) \leftarrow V(S_t) + \alpha(G_t - V(S_t))
$$

### Car Driving Example

An analogy for understanding the difference between MC and TD methods. Imagine we are driving the car home. At the start, we expect to take `30 mins` for the journey. And then it rains, so we update our prediction to `40 mins`. And so on. Eventually, the final journey takes `43 mins`.
- For MC method, we need to wait until the journey is complete, and then we update the value of our policy to `43 mins` each step of the way
- For TD method, we can immediately update the value function to the next prediction each step of the way

### Bias Variance Trade-off

There is a bias variance trade-off between choosing MC or TD method for policy evaluation.
- The return $G_t = R_{t+1} + \gamma R_{t+2} + .. + \gamma^{T-1}R_T$ is an unbiased estimate of $v_\pi(S_t)$
- The oracle TD target $R_{t+1} + \gamma v_\pi(S_{t+1})$ is also an unbiased estimate of $v_\pi(S_t)$
    - We know this from the bellman expectation equation
    - But it requires access to the oracle $v_\pi(S_{t+1})$ which we do not have
- The TD target $R_{t+1} + \gamma V(S_{t+1})$ is a biased estimate of $v_\pi(S_t)$
    - This is because $V$ is our current estimate of the value function which can be wildly wrong
- Observe that the TD target has much lower variance than the return:
    - The return depends on many random actions, transitions, rewards through the entire run of the episode
    - The TD target only depends on one random action, transition and reward
        - The value function $V$ is a deterministic function

So to summarize:
- MC has high variance and zero bias
    - So it has good convergence properties, even with function approximation later
    - It is not very sensitive to the initial value
    - Very simple to understand and use
- TD has low variance but some bias
    - Usually it is much more efficient than MC
    - TD(0) can be proven to converge to $v_\pi(s)$ using a table lookup
    - But with function approximation convergence is not always guaranteed
    - More sensitive to the initial value

What is function approximation? This will be covered later on. But in general, we have been looking at $v(s)$ as a table lookup for each state. This is not feasible for problems with large state spaces, hence we need to learn a function to approximate $v(s)$ for all states.

MC vs TD empirical example
- TD generally converges faster than MC 
- But if the step size $\alpha$ is too larger, TD may not fully converge as it will oscillate

So far we have seen that both MC and TD converge as the number of episodes goes to infinity.
- That is, $V(s) \rightarrow v_\pi(s)$ as $episodes \rightarrow \infty$
- But what if we only have a limited number of $K$ episodes to learn from?
- For example, what if we are repeatedly sampling episode $k \in [1, K]$?

### AB Example
A simple example to illustrate difference between MC and TD in the finite data case. Suppose we have 6 episodes:
- `A, 0, B, 0`
- `B, 1`
- `B, 1`
- `B, 1`
- `B, 1`
- `B, 0`

What would we think $V(A), V(B)$ are?
- If we use MC, then $V(A) = 0, V(B) = 4/6$. $V(A)$ is $0$ because we only encountered one episode involving state $A$ and the reward was $0$.
- If we use TD, then $V(A) = 4/6, V(B) = 4/6$. $V(A)$ is $4/6$ because we observed a `100%` probability of transiting from $A \rightarrow B$, so the value of $A$ (without discounting) is the same as value of $B$ due to bootstrapping.

In more precise terms:
- MC converges to the solution which minimizes the mean squared error
    - i.e. minimizes the divergence from observed returns
    $$
        \min \sum_{k=1}^K \sum_{t=1}^{T_k} \left[ 
            g_t^k - V(s_t^k)
        \right]^2
    $$
    - In the AB example above, this sets $V(A) = 0$
- TD converges to the solution of the maximum likelihood Markov model
    - i.e. it converges to the MDP that best fits the data
    - In the AB example, $V(A) = 4/6$

In summary:
- TD exploits the markov property, so it is usually more efficient in markov environments, where we can rely on states to encode information
- MC does not exploit the markov property, so it is usually more efficient in non-markov environments, e.g. partial observability etc.

So far we have looked at 3 types of backup:
- <<Monte Carlo Backup>>: we sample one entire trajectory / episode from the agent's interactions with the environment till termination
$$
    V(S_t) \leftarrow V(S_t) + \alpha (G_T - V(S_t))
$$
- <<Temporal Difference Backup>>: we sample one step lookahead and then update parameters
$$
    V(S_t) \leftarrow V(S_t) + \alpha ( R_{t+1} + \gamma V(S_{t+1}) - V(S_t))
$$
- <<Dynamic Programming Backup>>: we look ahead one step, but because we have access to the environment, we can compute the expectation over all possible next steps.
$$
    V(S_t) \leftarrow \E_{\pi} [ R_{t+1} + \gamma V(S_{t+1})]
$$

This gives us two dimensions to categorize our algorithms:
- <<Bootstrapping>>: the update involves an estimate (e.g. our value function)
    - MC does not bootstrap
    - DP bootstraps
    - TD bootstraps
- <<Sampling>>: we use sampling instead of a full-width expectation / search
    - MC samples
    - DP does not sample
    - TD samples

## TD Lambda

TD Lambda is a generalization of the above trade-off. We let TD target look $n$ steps into the future before updating. If we look forward $\infty$ number of steps, it becomes monte carlo learning.

Specifically, for $n=1,2,\infty$, our returns are:
- $n=1$:  $G_t^{(1)} = R_{t+1} + \gamma V(S_{t+1})$
- $n=2$:  $G_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 V(S_{t+2})$
- $n=\infty$:  $G_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{T-1} R_T$. We can see this corresponds to MC update, without use of value function $V$ at all

So the n-step return is:
$$
    G_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^n V(S_{t+n})
$$

And the n-step TD learning update is:
$$
    V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^{(n)} - V(S_t) \right)
$$

What is the best $n$? It is a highly sensitive parameters that depends on the problem, $\alpha$ etc. Hence a proposal is made to average the returns from each time step, up to step $n$. For example, we could average $\frac{1}{2} G_t^{(2)} + \frac{1}{2} G_t^{(4)}$. This averaging would make the algorithm much more robust to step size $n$.

The common way to perform a weighted average of returns is to use exponential $\lambda$ decay, such that returns with a longer look-ahead window are weighted less. This algorithm is called TD-$\lambda$. Specifically:
$$
    G_t^\lambda = (1-\lambda) \sum_{n=1}^\infty \lambda^{n-1} G_t^{(n)}
$$

Note that the weight given to the final return $G_t^{(n)}$ is the sum to $\infty$ of weights from step $n$ onwards, i.e. it is a geometric series. It makes sense to put more weight on the final, actual return.

This leads directly to <<forward-view TD($\lambda$)>>, where we sample trajectories of $n$ steps and update the value function according to:
$$
    V(S_t) \leftarrow V(S_t) + \alpha \left( 
        G_t^\lambda - V(S_t)
    \right)
$$

Now, the forward view has a shortcoming, which is that we need to wait until we have sampled $n$ steps into the future, before we can update the value function. Thus it suffers similar downside to MC update, where we cannot update the value function immediately after each step.

## Backward View TD Lambda

One key idea is <<eligibility traces>>. In deciding to assign credit to past events for a current reward, there are generally two intuitive heuristics to use:
- <<Frequency heuristic>>: assign credit to most frequent recent states
- <<Recency heuristic>>: assign credit to most recent states

The eligiblity trace combines both heuristics in a simple formula:
- $E_0(s) = 0$
- $E_t(s) = \gamma \lambda E_{t-1}(s) + \mathbf{1}(S_t=s)$

The eligibility trace gives us a weight at a given time step for each state $s$. This weight tells us how much credit we should assign to $s$ for a reward at the current time step.

The <<Backward View TD Lambda>> uses this idea:
- Keep an eligibility trace for every state $s$
- Update value V(s) for *every* state $s$ in proportion to the TD-error $\delta_t$ and eligibility trace $E_t(s)$:
$$
\begin{align*}
    \delta_t &= R_{t+1} + \gamma V(S_{t+1}) - V(S_t)\\
    V(s) &\leftarrow V(s) + \alpha \delta_t E_t(s)
\end{align*}
$$

Observe that $\delta_t$ is just our update for <<TD(0)>> with a single step look ahead, i.e. $G_t^{(1)} - V(S_t)$. Thus we can see that when $\lambda=0$, only the current state is updated, since $E_t(s) = \mathbf{1}(S_t = s)$. This results in the TD(0) update: $V(S_t) \leftarrow V(S_t) + \alpha \delta_t$.

On the other extreme, when $\lambda=1$, all credit is deferred until the end of the episode (not sure I see this from the formula). Thus it is equivalent to MC update.

In fact, there is a theorem that the sum of offline updates is identical for both forward view and backward view TD-lambda. This is nice because the backward view with eligibility traces makes it easy to implement, as we never need to look forward into the future. We just need to keep track of eligibility traces at each time step, and then apply the update to all states at each step.




