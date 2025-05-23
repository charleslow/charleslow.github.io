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

So we have seen that value iteration iteratively applies the bellman optimality equation to update $v_1 \rightarrow v_2 \rightarrow ... \rightarrow v_*$. (<<Note:>> Useful to compare this update statement with the bellman expectation equation used for policy evaluation above). Convergence to $v_*$ will be proved later. The update equation is:
$$
\begin{align*}
    v_{k+1}(s) = \max_{a \in \mathcal{A}} \left(
        \mathcal{R}_s^a + \gamma \sum_{s' \in S} \mathcal{P}_{ss'}^a v_k(s')
    \right)
\end{align*}
$$

Note that another difference between value iteration and policy iteration is that there is no explicit policy in value iteration. Since we are only doing one step of policy evaluation and then immediately taking the greedy step, the value function we have may not correspond to any real policy. But this does not stop the algorithm from converging.

The following table sums up the relationship between what we've learnt.

| Problem | Bellman Equation | Algorithm |
|----------------|------------|---------|
| Prediction | Bellman Expectation Equation | Iterative Policy Evaluation |
| Control | Bellman Expectation Equation + Greedy Policy Improvement | Policy Iteration |
| Control | Bellman Optimality Euqation | Value Iteration |

Some notes:
- These algorithms are based on the state value function $v_\pi(s)$
- The complexity for $m$ actions and $n$ states is $O(mn^2)$ per iteration
- We could also apply the same algorithm to the action-value function $q_\pi(s, a)$
- But the complexity worsens to $O(m^2n^2)$ per iteration

## Extensions to Dynamic Programming

- The DP methods described so far used synchronous backups, i.e. we backup all states in parallel
- Asynchronous backs up state individually in any order, without updating all states in one step
- This can significantly reduce computation
- There are nice properties to show that it is guaranteed to converge if we still select all states in the way we update

Now, three simple ideas:
- In place Dynamic Programming
- Prioritized Sweeping
- Real time dynamic programming

<<In place dynamic programming>>. A simple idea where we update the value function in-place rather than store it in a separate array.

Original (store updates in a separate array): 
- For all $s \in \S$:
    - $v_{\text{new}}(s) \leftarrow \max_{a \in \mathcal{A}} ( \mathcal{R}_s^a + \gamma \sum_{s' \in S} \mathcal{P}_{ss'}^a v_{\text{old}}(s') )$
- $v_{\text{old}} \leftarrow v_{\text{new}}$

New (in place updates right away):
- For all $s \in \S$:
    - $v(s) \leftarrow \max_{a \in \mathcal{A}} \left( \mathcal{R}_s^a + \gamma \sum_{s' \in S} \mathcal{P}_{ss'}^a v(s') \right)$

The method with in place updates has more recent updates to $v$, and thus often are a lot more efficient in convergence.

<<Prioritized Sweeping>>. Since we are doing immediate updates to $v$, it begs the question: in what order should we update states?
- One method is to use the magnitude of the bellman error to guide state selection:
$$
    \text{error} = \left| 
        \max_{a \in \mathcal{A}} \left(
            \mathcal{R}_s^a + \gamma \sum_{s' \in S} \mathcal{P}_{ss'}^a v(s')
        \right)
        - v(s)
    \right| 
$$
- The idea is that states with the largest bellman error are those states whose value functions will change the most, which will significantly change the dynamics of the system, so we should update them first
- This can be implemented efficiently by maintaining a priority queue

<<Real time dynamic programming.>> Idea is to select states that the real world agent is visiting to update.
- Basically update the states that the agent is visiting right now
- After each time step we have $S_t, A_t, R_{t+1}$
- So we update the state $S_t$

Important note: DP uses full-width backups, whether we are doing sync or async updates. This means that we consider the max over every successor state and action. Also, we need full knowledge of the MDP dynamics to compute. For large problem spaces, one single backup may be too expensive to compute. Hence in subsequent lectures we will consider sample backups.

Can use the contraction mapping theorem to prove convergence etc.

