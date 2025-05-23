# Lecture 2: Markov Decision Processes

[Lecture 2: Markov Decision Processes](https://www.youtube.com/watch?v=lfHX2hHRMVQ&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=2)

Markov Decision Processes formally describe an environment for Reinforcement Learning. 
- The environment is fully observable, i.e. the current state fully characterizes the process
- Almost all RL problems can be characterized as an MDP
- Even continuous things like Optimal Control
- Partially observable cases can be formulated as MDPs
- Bandits are MDPs with one state

The Markov Property is central to MDPs. "The future is independent of the past given the present."

<<State Transition Matrix>>. For a Markov state $s$ and successor state $s'$, the state transition probability is defined as:
$$
    P_{ss'} = P(S_{t+1} = s' | S_t = s)
$$

The state transition matrix $\mathcal{P}$ defines transition probabilities from all states $s$ to all successor states $s'$. 

$$
\mathcal{P} = 
\begin{bmatrix}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{11} & \cdots & \mathcal{P}_{nn}
\end{bmatrix}
$$

Each row of the transition matrix sums to $1$.

<<Markov Process>>. A Markov Process is a memoryless random process, i.e. a sequence of of random states $S_1, S_2, ...$ with the Markov Property. 

> **Definition**. 
A Markov Process (or Markov Chain) is a tuple $<\mathcal{S}, \mathcal{P}>$, where:  
> (i) $\mathcal{S}$ is a (finite) set of states  
> (ii) $\mathcal{P}$ is a state transition probability matrix  
> (iii) $P_{ss'} = P(S_{t+1} = s' | S_t = s)$ 

Example of a Markov Process. A student can transit from Class 1 to Class 2 to Class 3, Pass or Sleep or Pub based on transition probabilities. We can sample <<episodes>> for the markov chain. E.g. one episode may be `C1 C2 C3 Pass Sleep`.

The transition probability matrix may look something like the below. Note that `Sleep` is the terminal state, so its self-probability is `1.0`.

$$
\mathcal{P} = 
\begin{bmatrix}
\text{} & \text{C1} & \text{C2} & \text{C3} & \text{Pass} & \text{Pub} & \text{FB} & \text{Sleep} \\
\hline % Horizontal line separating labels from matrix data
\text{C1} & 0 & 0.5 & 0 & 0 & 0 & 0.5 & 0 \\
\text{C2} & 0 & 0 & 0.8 & 0 & 0 & 0 & 0.2 \\
\text{C3} & 0 & 0 & 0 & 0.6 & 0.4 & 0 & 0 \\
\text{Pass} & 0 & 0 & 0 & 0 & 0 & 0 & 1.0 \\
\text{Pub} & 0.2 & 0.4 & 0.4 & 0 & 0 & 0 & 0 \\
\text{FB} & 0.1 & 0 & 0 & 0 & 0 & 0.9 & 0 \\
\text{Sleep} & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\
\end{bmatrix}
$$

A markov reward process is a markov chain with values.

> **Definition.** A <<Markov reward process>> is a tuple $<\mathcal{S}, \mathcal{P}, \mathcal{R}, \gamma>$:  
> - $\S$ is a finite set of states  
> - $\mathcal{P}$ is a transition probability matrix, $\mathcal{P}_{ss'} = P[S_{t+1} = s' | S_t = s]$
> - $\mathcal{R}$ is a reward function, $\mathcal{R}_s = \E[R_{t+1} | S_t=s]$  
> - $\gamma$ is a discount factor, $\gamma \in [0,1]$

> **Definition.** The <<return>> $G_t$ is the total discounted reward from time step $t$.  
> $G_t = R_{t+1} + \gamma R_{t+2} + ... = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

There is no expectation because $G_t$ is one sample run of the Markov reward process. We'll take expectation later to get the expected return over infinite runs. 
- Note that the discount factor is the present value of future rewards. $0$ implies maximally short-sighted and $1$ implies maximally far-sighted. 
- The value of receiving reward $R$ after $k+1$ time steps is $\gamma^k R$
- This setup values immediate reward above future reward

Most Markov reward and decision processes are discounted. Why?
- We do not have a perfect model, so the expected future rewards are more uncertain. Hence we put higher weights on immediate rewards.
- Avoids inifite returns in cyclic Markov processes
- If the reward is financial immediate rewards earn more interest than delayed rewards
- Animal / human behaviour shows preference for immediate rewards
- It is sometimes possible to use undiscounted Markov reward processes, e.g. if all sequences terminate

The value function $v(s)$ gives the long-term value of state $s$.
> **Definition.** The <<state value function>> $v(s)$ of an MRP is the expected return stating from state $s$:  
>   $v(s) = \E[G_t | S_t = s]$

How do we compute the state value function? One way is to sample returns from the MRP. e.g. stating from $S1 = C1$ and $\gamma = 1/2$:
- `C1 C2 C3 Pass Sleep`: `-2.25`
- `C1 FB FB C1 C2 Sleep`: `-3.125`

Consider if we set $\gamma = 0$. Then the value function $v(s) = R_s$, i.e. the value is just the immediate reward.

Now the important <<Bellman equation for MRPs>>:
$$
    v(s) = \E[R_{t+1} + \gamma v(S_{t+1})| S_t = s]
$$

It essentially tells us that the value function can be decomposed into two parts:
- Immediate reward $R_{t+1}$
- Discounted value of successor state $\gamma v(S_{t+1})$

$$
\begin{align*}
    v(s) 
    &= \E[G_t | S_t = s]\\
    &= \E[R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... | S_t = s]\\
    &= \E[R_{t+1} + \gamma (R_{t+2} + \gamma R_{t+3} + ...) | S_t = s]\\
    &= \E[R_{t+1} + \gamma G_{t+1} | S_t = s]\\
    &= \E[R_{t+1} + \gamma v(S_{t+1})| S_t = s]\\
    &= \mathcal{R}_s + \gamma \sum_{s'} P_{ss'} v(s')
\end{align*}
$$

- Note that in the second-to-last line, the argument inside $v(S_{t+1})$ is a random variable, to express the fact that the state at time $t+1$ is random. 
- Note that both $G_{t+1}$ and $v(S_{t+1})$ are random variables, which express the value function at each possible state at time step $t+1$. 
- $G_t$ becomes $v(S_{t+1})$ due to the law of iterated expectations. Recall that $\E[X] = \E[\E[X | Y]]$. (Not very sure exactly how this works out.)

To dig into bellman equation a bit more. Use a 1-step look ahead search. We start at state $s$, we look ahead one step and integrate over the probabilities of the next time step. Hence we get $v(s) = \mathcal{R}_s + \gamma \sum_{s'} P_{ss'} v(s')$.

We can use the bellman equation to verify if our value function is correct. Taking the value at a particular state, we can check if it is indeed the sum of the immediate reward and the weighted sum of values in all possible next steps.

The Bellman equation can be expressed concisely using matrices,
$$
    v = \mathcal{R} + \gamma \mathcal{P} v
$$

$$
\begin{equation*}
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
=
\begin{bmatrix}
\mathcal{R}_1 \\
\vdots \\
\mathcal{R}_n
\end{bmatrix}
+ \gamma
\begin{bmatrix}
\mathcal{P}_{11} & \cdots & \mathcal{P}_{1n} \\
\vdots & \ddots & \vdots \\
\mathcal{P}_{n1} & \cdots & \mathcal{P}_{nn}
\end{bmatrix}
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
\end{equation*}
$$

The bellman equation is a linear equation and can be solved directly using matrix inversion.
$$
\begin{align*}
v &= \mathcal{R} + \gamma \mathcal{P} v\\
(1-\gamma \mathcal{P}) v &= \mathcal{R} \\ 
v &= (I - \gamma \mathcal{P})^{-1} \mathcal{R}
\end{align*}
$$

The complexity due to the matrix inversion if $O(n^3)$ for $n$ states, which is not feasible for a large number of states. There are many iterative methods which are more efficient:
- Dynamic programming
- Monte Carlo evaluation
- Temporal Difference learning

## Markov Decision Process

So far it has been a building block. The MDP is what we really use. A Markov Decision Process (MDP) is a markov reward process with decisions (actions). It is an environment in which all states are Markov.

> **Definition.** A Markov Decision Process is a tuple $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >:$
> - $\mathcal{S}$ is a finite set of states
> - $\mathcal{A}$ is a finite set of actions
> - $\mathcal{P}$ is a transition probability matrix with $P^a_{ss'} = P[S_{t+1} = s' | S_t =s, A_t = a]$
> - $\mathcal{R}$ is a reward function, $\mathcal{R}_s^a = \E[R_{t+1} | S_t =s, A_t = a]$
> - $\gamma$ is a discount factor

Note that the transition probabilities and reward functions now also depend on an action, in which we can have some agency now. We can choose actions to influence the reward and values.

> **Definition.** A <<policy $\pi$>> is a distribution over actions given states,  
> $$\pi(a | s) = P[ A_t =a | S_t = s]$$

A policy fully defines the behaviour of an agent. Some properties of a policy:
- It only depends on the current state (not the history)
- The policy does not depend on the time step $t$ (i.e. stationary)

We can still obtain the optimal policy because of the markov property - the current state captures all relevant information to make the optimal decision.

> **Definition.** The <<state-value function $v_{\pi}(s)$>> of an MDP is the expected return starting from state $s$, and following policy $\pi$:
> $$v_{\pi}(s) = \E_{\pi} [G_t | S_t = s]$$

> **Definition.** The <<action-value function $q_{\pi}(s, a)$>> of an MDP is the expected return starting from state $s$, taking action $a$, and following policy $\pi$:
> $$q_{\pi}(s, a) = \E_{\pi} [G_t | S_t = s, A_t=a]$$

<<Bellman Expectation Equation>>. The state-value function can again be decomposed into immediate reward plus discounted value of the successor state.
$$
    v_{\pi}(s) = \E_{\pi} [R_{t+1} + \gamma v_{\pi}(S_{t+1}) | S_t=s]
$$

Similarly we can do so for the action-value function, by inserting the chosen action:
$$
    q_{\pi}(s, a) = \E_{\pi} [R_{t+1} + \gamma q_{\pi}(S_{t+1}, A_{t+1}) | S_t=s, A_t=a]
$$

From a given state, we have a value function attached to that state, i.e. $v_{\pi}(s)$. From this state, we have some possible actions to take. The policy determines the probability distribution over which action to take. With each action comes an action-value function $q_{\pi}(s, a)$. Hence we have:
$$
    v_{\pi}(s) = \sum_{a \in A} \pi(a|s) q_{\pi}(s,a)
$$

Another way to look at it. We start with having chosen a particular action. Having chose a particular action, the environment will determine the particular state I end up in (based on the transition probability matrix $\mathcal{P}$). Hence we have:
$$
    q_{\pi}(s, a) = \mathcal{R}_s + \gamma \sum_{s' \in S} P_{ss'}^a v_{\pi}(s') 
$$

Now we can stitch these two perspectives together. Starting from a particular state, we can write $v_{\pi}(s)$ in terms of $q_{\pi}$, then write $q_{\pi}$ in terms of $v_{\pi}$ again. This will allow us to get a recursive relationship of $v_{\pi(s)}$ in terms of $v_{\pi}(s')$ and allow us to solve the equation.

The <<bellman expectation equation>> for $v_{\pi}(s)$ is thus:

$$
    v_{\pi}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left( 
        \mathcal{R}_s^a + \gamma \sum_{s' \in \S} \mathcal{P}_{ss'}^a v_{\pi}(s')
    \right)
$$

The math is expressing a simple idea: that the value at a particular state $s$ is the weighted sum of values from all possible actions we take under the current policy $\pi$. The value of each action is in turn affected by the reward function and the transition probability that determines the state we end up in after taking a particular action.

Similarly, we can do the same by starting at an action instead of a state. The bellman expectation equation for $q_{\pi}$ is thus:

$$
    q_{\pi}(s, a) = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' | s')q_{\pi}(s', a')
$$

## Optimal Value Function

So far we have been defining the dynamic process of the MDP, but have not tried solving the optimization problem. We will turn to this now.

> **Definition.** The <<optimal state-value function>> $v_*(s)$ is the maximum value function over all policies:
> $$v_*(s) = \max_{\pi} v_{\pi}(s)$$
> The <<optimal action-value function>> $q_*(s, a)$ is the maximum action-value function over all policies:
> $$q_*(s, a) = \max_{\pi} q_{\pi}(s, a)$$

The MDP problem is solved once we find $q_*$. We thus need some algorithms to systematically find $q_*$.

Define a partial ordering over policies:
$$
    \pi \geq \pi' \text{ if } v_{\pi}(s) \geq v_{\pi'}(s) \forall s
$$

> **Theorem.** For any MArkov Decision Process:
> - There exists an optimal policy $\pi_*$ that is better than or equal to all other policies, i.e. $\pi_* \geq \pi, \forall \pi$
> - All optimal policies achieve the optimal value function, i.e. $v_{\pi_*}(s) = v_*(s), \forall s$
> - All optimal policies achieve the optimal action-value function, i.e. $q_{\pi_*}(s, a) = q_*(s, a), \forall s, a$

How do we find the optimal policy? An optimal policy can be found trivially by maximizing over $q_*(s, a)$, if we knew it. That is, we always pick the action $a$ with the highest $q(s, a)$ value. Hence if we have $q_*$, we have $\pi_*$.
$$
\pi_*(a \mid s) =
\begin{cases} 
1 & \text{if } a = \arg\max\limits_{a' \in A} \, q_*(s, a') \\
0 & \text{otherwise}
\end{cases}
$$

Intuitively, we find the optimal policy by starting at the end (resting), and iteratively look backward. This is the same kind of intuition for the <<Bellman optimality equations>>.

The optimal value of being in a state $s$ is the highest value action we can take in that state. Note that we use $q_*$ instead of a generic $q$ because we are choosing from the optimal action-value function.
$$
    v_*(s) = \max_{a} q_*(s, a)
$$

The optimal value of an action $a$ is the weighted sum of values of states that we can end up in after taking the action. Note that in this step, we do not get to choose an action - the transition probabilities will determine what state we end up in after taking actions $a$:
$$
    q_*(s, a) = \mathcal{R}^a_s + \sum_{s' \in \S} \mathcal{P}^a_{ss'} v_*(s')
$$

Finally, stitching these two equations together, we get the bellman optimality equation for $v_*$:
$$
    v_*(s) = \max_a \left[
        \mathcal{R}^a_s + \sum_{s' \in \S} \mathcal{P}^a_{ss'} v_*(s')
    \right]
$$

How do we solve the bellman optimality equations? It is now non-linear due to the `max` function, so we cannot solve it with matrix inversion as before. There is no closed form solution in general, but there are many iterative solution methods:
- Value iteration
- Policy iteration
- Q-learning
- Sarsa

> **Intuition.** The core idea behind the bellman equations is to break down a complex sequential decision problem into a series of simpler, recursive steps. Imagine we are at a particular point in time and in a particular state. The bellman equations tell us that if we can assume that we will act optimally for all future steps after this action, then the problem of finding the best current action becomes trivial - we simply choose the action that yields the highest expected value (based on assuming future optimality).
> 
> To actually start unravelling the equations and solving them, we start from the termination point of a process (where the assumption of future optimality trivially holds) and work backwards.
