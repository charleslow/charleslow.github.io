# RL Course

Lecture notes based on [David Silver's Reinforcement Learning course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ).

Textbooks:
- An Introduction to Reinforcement Learning - Sutton and Barto
- Algorithms for Reinforcement Learning, Szepesvari

## Lecture 1: Introduction

[Lecture 1](https://youtu.be/2pWv7GOvuf0?feature=shared)

Reinforcement Learning is at the centre of many disciplines such as engineering, psychology, economics, neuroscience because it deals with the science of decision making. E.g. in neuroscience, one major part is the dopamine neurotransmitter which resembles RL. Mathematics - operations research. Economics - bounded rationality.

What makes RL different from other ML paradigms? 
- There is no supervisor, only a reward signal. There is no clear correct action to take, only a reward
- The feedback is delayed, not instantaneous
- Time really matters in RL (non iid data)
- Agent's actions affect the subsequent data it receives

Examples:
- Fly stunt manoeuvres in a helicopter
- Play backgammon
- Manage an investment portfolio
- Make a humanoid robot walk

Rewards. A reward $R_t$ is a scalar feedback signal. Indicates how well agent is doing at step $t$. The agent's job is to maximize cumulative reward. RL is based on the reward hypothesis.

> <<Reward Hypothesis>>. All goals can be described by the maximization of expected cumulative reward.

Is a scalar feedback sufficient? David's argument is that in order to pick one action, we must always be able to compare and rank two actions. Hence we must have a way of ordering actions and it boils down to a scalar score.

Examples of rewards:
- Fly stunt manoeuvres in a helicopter:
    - +ve reward for following desired trajectory
    - -ve reward for crashing
- Backgammon:
    - +ve / -ve reward for winning / losing a game
- Manage investment portfolio
    - +ve reward for each \$ in bank

Sequential decision making. Goal: select actions to maximize total future reward. We have to plan ahead, because actions may have long term consequences. It may be better to sacrifice immediate reward to gain more long term reward. Greedy approach does not work in RL.

Formalism. At each step $t$, the agent:
- Executes action $A_t$
- Receives observation $O_t$ from environment
- Receives reward $R_t$ from environment

The environment:
- Receives action $A_t$
- Emits observation $O_t$
- Emits reward $R_t$

The <<history>> is the sequence of observations, actions, rewards.
$$
    H_t = A_1, O_1, R_1, ..., A_t, O_t, R_t
$$

What happens next depends on the history:
- The agent selects the next action based on the history
- The environment selects observation and reward to emit

This is a very generic framework or formalism that can handle all types of real world scenarios. 

<<State>> is the information used to determine what happens next. We do not want to have to load the entire history (e.g. stream of video frames) to make a decision. Formally, state is a function of the history:
$$
    S_t = f(H_t)
$$

The simplest example of state is to e.g. just take the observation at the last timestamp (this worked for Atari games).

The <<environment state>> $S_t^e$ is the environment's private representation. What state the environment is in. That is, the data that the environment uses to pick the next observation / reward. The environment state is not usually visible to the agent or the algorithm. 

The <<agent state>> $S_t^a$ is the agent's internal representation, i.e. it is the information used by reinforcement learning algorithms. It can be any function we choose of the history: $S_t^a = f(H_t)$.

A more mathematical definition of state. An information state (or <<Markov state>>) contains all useful information from the history.

> **Definition**. A state S_t is Markov if and only if
$$
    P(S_{t+1} | S_t) = P(S_{t+1} | S_1, ..., S_t)
$$

The idea may be stated as "the future is independent of the past given the present". Once the state is known, the history may be thrown away. i.e. The state is a sufficient statistic of the future. In the helicopter example, the markov state might be something like position, velocity, angle, angular velocity etc. Having known all this information, it does not matter where the helicopter was 10 minutes ago, as we already have all the information we need to make an optimal decision next.

Two trivial statements to show that there always exists a markov state:
- The environment state $S_t^e$ is Markov by definition
- The history $H_t$ is Markov, albeit not a useful one

<<Fully observable environments.>> Full observability means that agent directly observes the environment state. The nice case.
$$
    O_t = S_t^a = S_t^e
$$

Formally, this is a <<markov decision process (MDP)>>. 

<<Partially observable environments>>. Partial observability means that agent indirectly observes the environment. e.g.
- Robot with camera doesn't know its absolute location
- A trading agent only observes current prices
- A poker playing agent only observes public cards, not hidden cards

Now, agent state is not the same as the environment state. Formally this is a partially observable markov decision process. So now we need the agent to construct its own state representation $S_t^a$, e.g.:
- Remember the complete history $S_t^a = H_t$
- Build beliefs of the environment state, i.e. $S_t^a = (P(s_t^e) = s_1, .. P(S_t^e) = s_n)$. So we have a probability distribution over possible states that we believe the environment is in.
- Recurrent neural network, i.e. $S_t^a = \sigma(S_{t-1}^a W_s + O_tW_o)$. Use a linear transformation to combine the current observation with previous time step to get current time step.

### Inside an RL agent

An RL agent may include these components:
- Policy: the agent's behaviour function
- Value function: how good is each state or action
- Model: the agent's representation of the environment

<<Policy>>. A policy is the agent's behaviour. It is a map from state to action, e.g.
- Deterministic policy, $a = \pi(s)$
- Stochastic policy, $\pi(a|s) = P(A=a | S=s)$

<<Value function>> is a prediction of future reward, or the expected future reward. It is used to evaluate the goodness or badness of states. 
$$
    v_{\pi}(s) = E_{\pi} \left[ R_T + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... | S_t=s \right]
$$

Note that the value function is subscripted by $\pi$. This indicates that the agent's expected future reward depends on its current policy. If the policy is causing a robot to fall a lot, the expected future reward is probably low.

<<Model>> A model predicts what the environment will do next. There are usually 2 parts to the model:
- <<Transitions>> $\mathcal{P}$ predicts the next state (dynamics of the environment)
- <<Rewards>> $\mathcal{R}$ predicts the next immediate reward

Formally,
$$
    \mathcal{P}^a_{ss'} = P(S' = s' | S=s, A=a)\\
    \mathcal{R}^a_s = \E \left[ R | S=s, A=a \right]
$$

Note that having a model is optional, there are many model-free methods in RL.

<<Maze example>>. We have a grid maze. The rewards are $-1$ per time step. The actions are N, E, S, W. The states are the agent's current location.

- An example of a deterministic policy would be to have a fixed arrow direction for the action to take in any given grid that the agent is in.
- An example of value function is to have a number in each position showing the number of steps it would take to get to the end (but negated as we lose value each time step) 

A taxonomy of RL agents:
- Value based - only using a value function, the policy is implicit
- Policy based - just have a policy function, no value function
- Actor critic - have both a value function and policy function, and try to get best of both worlds

Another categorization is model free vs model approach:
- Model free means we just go straight to policy and/or value function
    - No model
- Model based approach. Try to model the environment and world first, then build the policy accordingly

### Learning and Planning

Two fundamental problems in sequential decision making:
- Reinforcement learning
    - The environment is initially unknown
    - The agent interacts with the environment
    - The agent improves its policy
- Planning
    - A model of the environment is known and provided to the agent
    - The agent performs internal computations with the model without any external interaction
    - e.g. if when playing an atari game, we gave the model access to an atari emulator. The model then knows if it takes action $a$, the emulator will be in this state etc. The model can then plan ahead, build a search tree etc.

### Exploration and Exploitation

Reinforcement learning is like trial and error learning. But it is possible to miss out on exploring steps that can lead to more reward.

Exploration means choosing to gives up some known reward in order to find out more information about the environment. Exploitation exploits known information to maximize reward. 

Examples:
- Restaurant selection:
    - Exploitation: go to your favourite restaurant
    - Exploration: try a new restaurant
- Online banner advertisements
    - Exploitation: show the most successful advert
    - Exploration: show a different and new advert

## Lecture 2: Markov Decision Processes

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

### Markov Decision Process

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

### Optimal Value Function

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

# Lecture 3: Planning by Dynamic Programming

[Lecture 3: Planning by Dynamic Programming](https://www.youtube.com/watch?v=Nd1-UUMVfz4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=3)

What is dynamic programming?
- Dynamic: sequential or temporal component to the problem
- Programming: optimising a mathematical "program", i.e. a policy

It is a method for solving complex problems, by breaking them into subproblems that are simpler to solve. 

Dynamic Programming is a very general solution method for problems which have two properties:
- <<Optimal substructure>>: the pricniple of optimality applies. The optimal solution can be decomposed into subproblems.
    - e.g. to find the shortest path from A to B, we can find the shortest path from A to midpoint, and then find the shortest path from midpoint to B, and then combine the paths together.
- <<Overlapping subproblems>>. The subproblems need to recur many times and solutions can be re-used.
    - e.g. if we have the shortest path from midpoint to B, we can reuse that to find the shortest path from C to B if it traverses the midpoint as well.

Note that Markov Decision Processes satisfy both properties:
- The bellman equations decompose the large problem into recursive steps
- The value functions for a particular state cache the sub-solutions and are re-used

Planning by dynamic programming. Planning is a different problem from RL. Someone tells us the dynamics of the MDP, and we try to solve it.
- Assume full knowledge of the MDP
- Used for planning in an MDP
- We can use this for <<prediction>>:
    - e.g. input: an MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$ and policy $\pi$
    - The output of this planning step is to output the value function $v_{\pi}$
- We can also use this for <<control>>:
    - Input: MDP $<\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma >$
    - Output: optimal value function $v_*$, i.e. we want to find the best policy $\pi_*$

## Policy Evaluation

Problem: we want to evaluate a given policy $\pi$ to see how good it is. The solution is to iteratively apply the bellman expectation.
- Let $v_1$ be a vector representing the value at all states
- We use the bellman equation to update $v_1 \rightarrow v_2 \rightarrow ... \rightarrow v_{\pi}$, i.e. it will converge to the true value function for this policy
- Do this using synchronous backups:
    - At each iteration step $k+1$
    - For all states $s \in \S$
    - Update $v_{k+1}(s)$ using $v_k(s')$, i.e. we use the estimate of the value function in the previous iteration to form the new estimate of the value function
    - $s'$ is a successor state of $s$, i.e. the next states we can reach using an action from $s$
- It can be proven that this algorithm will converge to $v_{\pi}$

How exactly do we update $v_{k+1}(s)$? We use the <<bellman expectation equation>> from before. Intuitively, it is a one-step look ahead from the current state $s$ to compute the value for $s$.
$$
    v_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a | s) \left(
        \mathcal{R}_s^a + \gamma \sum_{s' \in \S} \mathcal{P}_{ss'}^a v_k (s')
    \right)
$$

Or in vector form:
$$
    \mathbf{v}^{k+1} = \mathcal{R^{\pi}} + \gamma \mathcal{P}^{\pi}\mathbf{v}^k
$$

**Small Gridworld example**. Suppose we have a 4x4 grid, in which the top-left and bottom-right grids are terminal states. The reward for any state is $-1$, and we can walk NSEW from any spot. If we walk off the grid, the action just does nothing. 

Now suppose we take a random walk ($0.25$ probability of walking any direction) and set $\gamma = 1$. How would the value update algorithm look like?
- We initialize all grids with value $0$
- Recall that the update algorithm essentially adds the immediate reward and a discounted sum of the value function from the previous iteration
- At $k=1$, every spot will be updated to $-1$ because the reward is $-1$, and $v_0(s) = 0 \ \forall s$. Except the two terminal states which remain at value $0$ by definition

If we continue to update like that, it will converge to the true value function for $\pi$. Also note that with this value function $v_{\pi}$, if we take a greedy approach to devise a new policy, we can obtain the optimal policy. 


> **Idea:** A lousy policy can be used to devise a better policy after computing the value function.

## Policy Iteration

We did policy evaluation in the previous section, i.e. finding the true value function $v_\pi$ for a given policy. Now in this section we want to optimize and find the best policy.

Two step process:
- <<Policy Evaluation>>. Given a policy $\pi$, we first evaluate the policy $\pi$, finding $v_\pi(s) = \E [R_{t+1} + \gamma R_{t+2} + ... | S_t = s]$
- <<Policy improvement>>. We take a greedy approach and choose the best action at each state: $\pi' = \text{greedy}(v_\pi)$

Typically, we need many rounds of iteration of this process to converge. But the process of policy iteration always converges to $\pi_*$. Specifically:
- $v$ converges to $v_*$
- $\pi$ converges to $\pi_*$

Somewhat realistic toy example from Sutton and Barto:
- Suppose we have two locations, maximum of 20 cars at each location
- Each day, we get to move up to `5` cars between locations overnight
- Reward: `$10` for each car rented (can only rent if we have enough cars)
- Transitions: every day, a random number of cars are requested and returned at each location (governed by a poisson distribution)
    - Location A: $\text{request} \sim Poisson(3)$, $\text{return} \sim Poisson(3)$
    - Location B: $\text{request} \sim Poisson(4)$, $\text{return} \sim Poisson(2)$

Naturally we expect the optimal policy to involve moving cars from location A to location B. Using the policy iteration process, we get convergence to the optimal policy in 4 steps. Note that since this is a planning problem, we do know the underlying probability mechanisms, which allows us to compute the value function.

Now we can show that this <<policy iteration process converges to the optimal policy>>. 

> **Theorem.** Policy iteration process converges to the optimal policy.  
> 
> **Proof.**
> First, consider a deterministic policy $a = \pi(s)$. Now, we consider what happens if we change the policy by acting greedily wrt the value function of this policy, i.e.:
$$
    \pi '(s) = \argmax_{a \in \mathcal{A}} q_{\pi}(s, a)
$$
> We see that taking the greedy action can only improve the policy, as expressed:
$$
\begin{align*}
    q_\pi(s, \pi'(s)) &= \max_{a \in \mathcal{A}} q_\pi(s, a) \\
    &\geq q_\pi(s, \pi(s)) = v_\pi(s)
\end{align*}
$$
> Notes on the above statement:
> - On the 2nd line equality: Recall that $q_\pi(s, a)$ is the value at state $s$ if we took action $a$ at $s$ and then followed policy $\pi$ thereafter. So it follows that $q_\pi(s, \pi(s)) = v_\pi(s)$, since we are just following policy $\pi$ in the current step + future steps
> - The statement is quite simply saying that $\pi'(s) \geq \pi(s)$, which leads to an improvement in $q_\pi$. Hence choosing the highest value action will improve the action value function (quite trivial).
> 
> Now we want to go from this somewhat trivial statement to show that the value function itself must improve with every step of policy iteration (not trivial at all!).
> 
> The idea is to use a telescoping argument to show that this improves the value function, or $v_{\pi'}(s) \geq v_{\pi}(s)$:
$$
\begin{align*}
    v_\pi(s) 
    &\leq q_\pi(s, \pi'(s)) \\
    &= \E_{\pi'}[R_{t+1} + \gamma v_\pi(S_{t+1}) | S_t = s] \\
    &\leq \E_{\pi'}[R_{t+1} + \gamma q_{\pi}(S_{t+1}, \pi'(S_{t+1}))| S_t = s] \\
    &\leq \E_{\pi'}[R_{t+1} + \gamma R_{t+2} + \gamma^2 q_{\pi}(S_{t+2}, \pi'(S_{t+2}))| S_t = s] \\
    &\leq \E_{\pi'}[R_{t+1} + \gamma R_{t+2} + ...| S_t = s] \\
    &= v_{\pi'}(s)
\end{align*}
$$
> 
> Some notes on the above:
> - We start with the trivial inequality expressed above
> - The expression $\E_{\pi'}$ means taking expectation over possible trajectories under the policy where we take $\pi'(s)$ in the current step, then follow policy $\pi$ for the rest of the trajectory
> - In line 2, we unpack $q_\pi$ according to the Bellman equation, which simply splits up the $q$ value into (i) the immediate reward and (ii) the expected value of our new state (expressed as a random variable $S$):
>    - Note that $R_{t+1}, R_{t+2}, ...$ are rewards from taking the greedy action $\pi'(s)$ at each step
>    - Note that $v_{\pi}(S_t+n)$ is the random variable expressing the value we have at the next time step, but evaluated under the previous policy $\pi$. It is the previous $\pi$ intead of $\pi'$ because we have access to the cached $v_\pi$ up to this point. 
> - In line 3, we apply the trivial inequality again to show that taking the greedy step at the next state will again improve the value
> - In line 4, we again use the Bellman equation to unpck $q_\pi$
> - We keep repeating the two steps until termination. What we have at the end is simply the value function of our new policy $\pi'$
> 
> We have shown that policy iteration must improve the value function with each iteration. Now what happens when improvements stop? We now have:
$$
    q_\pi(s, \pi'(s)) = v_\pi(s)
$$
> 
> Since $q_\pi(s, \pi'(s)) = \max_{a \in \mathcal{A}} q_\pi(s, a)$, we have:
$$
    v_\pi(s) = \max_{a \in \mathcal{A}} q_\pi(s, a)
$$
> 
> This is simply the Bellman optimality equation. Satisfying the Bellman optimality equation means that we are in the optimal state (will show later). Hence we have shown that we get $v_\pi(s) = v_*(s)$.

Now, an observation is that policy iteration is quite wasteful. This is because we need to get the value function to converge fully to $v_\pi$ before we take the greedy step to improve the policy. In most cases, this is unnecessary because the greedy policy would already improve even with an imperfect value function. 

Some ways to early stop policy evaluation to speed up this process:
- Introduce a stopping condition once the value function does not change by much ($\epsilon$-convergence of value function)
- Stop policy evaluation after $k$ iterations
    - In the extreme case, if we stop policy evaluation after $k=1$ iterations, it is called <<value iteration>>

## Value Iteration

Moving into value iteration, but recall the fundamentals of dynamic programming. Observe that any optimal policy can be subdivided into two components:
- An optimal first action $A_*$
- Followed by an optimal policy from successor state $S'$

> **Theorem**. Principle of optimality.
> 
> A policy $\pi(a|s)$ achieves the optimal value from state s, i.e. $v_\pi(s) = v_*(s)$, if and only if for any state $s'$ reachable from $s$, $\pi$ achives the optimal value from state $s'$, i.e. $v_\pi(s') = v_*(s')$

This theorem seems a bit of a truism, but it will be used to build the idea of value iteration.

Let us think of the value function as "caching" the solutions to subproblems. Now suppose we start "at the end" and assume we know the solution to all the subproblems $v_*(s')$ where $s'$ is all the states reachable from our current state $s$. 

Then we can solve immediately for $v_*(s)$ by doing a one-step lookahead to all these states $s'$:
$$
    v_*(s) \leftarrow \max_{a \in \mathcal{A}} \mathcal{R}^a_s + \gamma \sum_{s' \in \S} \mathcal{P}^a_{ss'} v_*(s')
$$

The statement above shows us how we can propagate the optimal value function from some states $s'$ to a new state $s$. So we can propagate the optimal value function across to all states as we continue to iterate.

The way to think about it (using the small gridworld as example) is that the termination point (trivially) starts off with the optimal value function. After one step of update, the states next to the termination point will now have the optimal value function, and then the states next to these, and so on until we propagate through all states.

Note that in contrast to policy iteration, where in the policy evaluation step we update the value function across all states based on the <<bellman *expectation* equation>>, in value iteration, we are updating the value in each state by choosing the optimal action. This is a key difference in how the two algorithms differ. The value iteration algorithm may be thought of as combining the (i) policy evaluation step and the (ii) greedy policy step from value itaration into one single step.

So we have seen that value iteration iteratively applies the bellman optimality equation to update $v_1 \rightarrow v_2 \rightarrow ... \rightarrow v_*$. Convergence to $v_*$ will be proved later. The update equation is:
$$
\begin{align*}
    v_{k+1}(s) = \max_{a \in \mathcal{A}} \left(
        \mathcal{R}_s^a + \gamma \sum_{s' \in S} \mathcal{P}_{ss'}^a v_k(s')
    \right)
\end{align*}
$$

Note that another difference between value iteration and policy iteration is that there is no explicit policy in value iteration. Since we are only doing one step of policy evaluation and then immediately taking the greedy step, the value function we have may not correspond to any real policy. But this does not stop the algorithm from converging.






