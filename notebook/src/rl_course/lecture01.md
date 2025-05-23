# Lecture 1: Introduction

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

## Inside an RL agent

An RL agent may include these components:
- Policy: the agent's behaviour function
- Value function: how good is each state or action
- Model: the agent's representation of the environment

<<Policy>>. A policy is the agent's behaviour. It is a map from state to action, e.g.
- Deterministic policy, $a = \pi(s)$
- Stochastic policy, $\pi(a|s) = P(A=a | S=s)$

<<Value function>> is a prediction of future reward, or the expected future reward. It is used to evaluate the goodness or badness of states. 
$$
    v_{\pi}(s) = E_{\pi} \left[ 
        R_T + \gamma R_{t+1} + \gamma^2 R_{t+2} + ... | S_t=s 
    \right]
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

## Learning and Planning

Two fundamental problems in sequential decision making:
- Reinforcement learning
    - The environment is initially unknown
    - The agent interacts with the environment
    - The agent improves its policy
- Planning
    - A model of the environment is known and provided to the agent
    - The agent performs internal computations with the model without any external interaction
    - e.g. if when playing an atari game, we gave the model access to an atari emulator. The model then knows if it takes action $a$, the emulator will be in this state etc. The model can then plan ahead, build a search tree etc.

## Exploration and Exploitation

Reinforcement learning is like trial and error learning. But it is possible to miss out on exploring steps that can lead to more reward.

Exploration means choosing to gives up some known reward in order to find out more information about the environment. Exploitation exploits known information to maximize reward. 

Examples:
- Restaurant selection:
    - Exploitation: go to your favourite restaurant
    - Exploration: try a new restaurant
- Online banner advertisements
    - Exploitation: show the most successful advert
    - Exploration: show a different and new advert