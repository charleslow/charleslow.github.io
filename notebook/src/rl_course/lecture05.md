# Model Free Control

[Lecture 5: Model Free Control](https://www.youtube.com/watch?v=0g4j2k_Ggc4&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&index=5)

All lectures are building up to this point, to optimize a problem where we do not have access to the underlying MDP. For such problems, we either do not know the underlying MDP, or it is too big to use (e.g. game of Go).

On policy vs Off policy
- On policy learning is to learn on the job. Learn about policy $\pi$ based on experience sampled from $\pi$
- Off policy learning is to learn by observing others. Learn about policy $\pi$ by sampling another robot (or human) experience $\mu$

Start with the simpler case, which is on policy learning. The basic framework is generalized policy iteration (recap), which alternates between:
- Policy evaluation: estimate $v_\pi$
- Policy improvement: generate a new $\pi' \geq \pi$

<<Naive case: policy iteration with Monte-Carlo evaluation>>. Basically, we use MC policy evaluation to update our value function, and then do greedy policy improvement. Would this work?
- No. The main problem is that previously when we had access to the underlying MDP, we could do greedy policy improvement because we had access to the transition dynamics. Specifically, when we do policy improvement, we want to compute:
$$
    \pi'(s) = \argmax_a \mathcal{R}_s^a + P_{ss'}^a V(s')
$$ 
- However in model-free control, we do not have access to $P_{ss'}^a$, meaning that we do not know what probabilities determine the state we will end up in given action $a$. So there is no clear way to do greedy policy improvement if we only have an estimate of $V(s) \forall s$.
- To deal with this issue, we can do greedy policy improvement over $Q(s, a)$ instead. Then we can simply take:
$$
    \pi'(s) = \argmax_{a \in \A} Q(s, a)
$$

So now we do generalized policy iteration with action-value function.
- Start with $Q, \pi$
- Update action-value function $Q = q_\pi$
- Greedily update policy to $\pi = greedy(Q)$

However, we still have another problem, which is the <<exploration issue>>. If we act greedily all the time, there is no guarantee that we will explore all states and thus find the optimal policy.

## Toy Example: Greedy Action Selection

Choose between two doors:
- Open left door: reward `0`. $V(left) = 0$
- Open right door: reward `+1`. $V(right) = +1$
- Open right door: reward `+3`. $V(right) = +2$
- Open right door: reward `+2`. $V(right) = +2$

The greedy policy will lock us onto right door forever. But we will never know if the left door actually has higher mean return.

## $\epsilon$-Greedy Exploration

The simplest idea for ensuring continual exploration.
- Try all $m$ actions with non-zero probability
- With probability $1-\epsilon$ choose the greedy action
- With probability $\epsilon$ choose an action at random

$$
\pi(a|s) = 
\begin{cases}
    \epsilon/m + 1 - \epsilon \ &\text{ if } a^* = \argmax_{a \in \A} Q(s, a)\\
    \epsilon/m \ \ &\text{otherwise}
\end{cases}
$$

Note that $\epsilon/m$ is added for the first case as well since the action chosen at random can include the greedy policy $a^*$ as well.

$\epsilon$-greedy policy is important because there is a theorem to assure us that we will indeed get a policy improvement on every step.

> **Theorem**. For any $\epsilon$-greedy policy $\pi$, the $\epsilon$-greedy policy $\pi'$ with respect to $q_\pi$ is an improvement, i.e. $v_{\pi'}(s) \geq v_{\pi}(s)$
> 
> **Proof**.
$$
\begin{align*}
    q_\pi(s, \pi'(s)) &= \sum_{a \in \A} \pi'(a|s) q_\pi(s, a) \\
    &= \epsilon / m \sum_{a \in \A} q_\pi(s, a) + (1 - \epsilon) \max_{a \in \A} q_\pi(s, a) \\
    &\geq \epsilon / m \sum_{a \in \A} q_\pi(s, a) + (1 - \epsilon) \sum_{a \in \A} \frac{\pi(a|s) - \epsilon / m}{1 - \epsilon} q_\pi(s, a) \\
    &= \sum_{a \in \A} \pi(a | s) q_\pi(s, a) = v_\pi(s)
\end{align*}
$$
> Therefore from the policy improvement theorem, $v_{\pi'}(s) \geq v_\pi(s)$.

The key step in the proof is the transition from line 2 to line 3. The idea is that the maximum q-value (by choosing the greedy action) will be greater than or equal to any weighted average of $q_\pi(s, a)$. Hence we choose a clever weighted average such that we can end up with $\sum_{a \in \A} \pi(a | s) q_\pi(s, a)$ in line 4.

Note that it is indeed a weighted average because of the following. Note that $\pi(a|s)$ must sum to 1 over all actions as it is a valid policy. And since there are $m$ unique actions, we multiply the constant $\epsilon/m$ by $m$. 
$$
\begin{align*}
    \sum_{a \in \A} \frac{\pi(a|s) - \epsilon / m}{1 - \epsilon}
    &= \frac{1}{1-\epsilon} \sum_{a \in \A} \left[
         \pi(a|s) - \epsilon / m
    \right]\\
    &= \frac{1}{1-\epsilon} \left[ 1 - m \cdot \epsilon/m \right]\\
    &= 1
\end{align*}
$$

An idea that we encountered earlier. We do not need to fully evaluate the policy before we do a greedy improvement. In the context of Monte Carlo policy evaluation, in the extreme case, we can update the policy after every episode instead of gathering many episodes.

How can we guarantee that we find the optimal policy $\pi^*$? We need to ensure that our algorithm balances two things: (i) suitably explore all options and (ii) ensure that at the end, we converge on a greedy policy.

This leads us to GLIE, which is a property that we want our algo to have.

> **Definition** Greedy in the Limit with Infinite Exploration (GLIE).
> - All state-action pairs are explored infinitely many times, i.e.
$$
    \lim_{k \rightarrow \infty} N_k(s, a) = \infty
$$
> - The policy converges on a greedy policy, i.e.
$$
    \lim_{k \rightarrow \infty} \pi_k(a | s) = \mathbf{1} \left(
        a = \argmax_{a' \in \A} Q_k(s, a')
    \right)
$$

One simple way to get GLIE is to use $\epsilon$-greedy with a decaying schedule for $\epsilon_k = \frac{1}{k}$.

## GLIE Monte Carlo Control

This brings us to GLIE Monte Carlo control.

> **Algorithm** GLIE Monte-Carlo Control.
> - Sample kth episode using policy $\pi: \{ S_1, A_1, R_2, ..., S_T \} \sim \pi$
> - For each state $S_t$ and action $A_t$ in the episode, update
$$
\begin{align*}
    N(S_t, A_t) &\leftarrow N(S_t, A_t) + 1 \\
    Q(S_t, A_t) &\leftarrow Q(S_t, A_t) + \frac{1}{N(S_t, A_t)}
        (G_t - Q(S_t, A_t)) \\
\end{align*}
$$
> - Improve policy based on the new action-value function:
$$
\begin{align*}
    \epsilon &\leftarrow 1/k\\
    \pi &\leftarrow \epsilon\text{-greedy(Q)}
\end{align*}
$$

## MC vs TD Control

- TD learning has several advantages over MC:
    - Lower variance
    - Online
    - Can deal with incomplete sequences
- Natural idea: use TD instead of MC in our control loop
    - Apply TD to Q(S, A)
    - Use $\epsilon$-greedy policy improvement
    - Update every time step
    - This is probably the most well known RL algorithm (Sarsa)

Sarsa policy evaluation update step:
$$
    Q(S, A) \leftarrow Q(S, A) + \alpha (R + \gamma Q(S', A') - Q(S, A))
$$

Note that we are updating the Q value for one single state-action pair. We take action $A$ on state $S$ and observe reward $R$, and use that to update the Q-value. In addition, we also sample a next action $A'$ and corresponding resultant state $S'$, and we bootstrap the Q-value to use $Q(S', A')$ to also update the Q-value. So it corresponds to a one-step lookahead in TD.

So the off-policy control with Sarsa algo. For every time step:
- Policy evaluation with Sarsa: $Q = q_\pi$
- Policy improvement using $\epsilon$-greedy

> **Algorithm**. Sarsa algorithm for on-policy control.
> - Initialize $Q(s, a), \forall s \in \S, a \in \A(s)$ arbitrarily
> - Repeat (for each episode):
>   - Initialize $S$
>   - Choose $A$ from $S$ using policy derived from $Q$ (e.g. $\epsilon$-greedy choice)
>   - Repeat (for each step of episode):
>       - Take action $A$, observe $R, S'$
>       - Choose $A'$ from $S'$ using policy derived from $Q$ (e.g. $\epsilon$-greedy)
>       - Update $Q(S,A) \leftarrow Q(S, A) + \alpha [R + \gamma Q(S', A') - Q(S, A)]$
>       - $S \leftarrow S'$, $A \leftarrow A'$
>   - Until $S$ is terminal

Note that this is a fundamentally on-policy algorithm, because the $A', S'$ that we sample and use to bootstrap is also the next action and state we end up in.

> **Algorithm.** Sarsa converges to the optimal action value function, $Q(s,a) \rightarrow q*(s, a)$ under the following conditions:
> - GLIE sequence of policies $\pi_t(a|s)$
> - Robbins Monro sequence of step sizes $\alpha_t$:
$$
\begin{align*}
    \sum_{t=1}^\infty \alpha_t &= \infty\\
    \sum_{t=1}^\infty \alpha_t^2 &< \infty\\
\end{align*}
$$

## $n$-step Sarsa

As before, we saw that $n$-step algorithm gets the best of both worlds in betwen MC and TD. So we do the same here.

Consider the following $n$-step returns for $n=1, 2, \infty$:
- $n=1$, $q_t^{(1)} = R_{t+1} + \gamma Q(S_t+1)$
- $n=2$, $q_t^{(2)} = R_{t+1} + \gamma R_{t+2} + \gamma^2 Q(S_t+2)$
- $n=\infty$, $q_t^{(\infty)} = R_{t+1} + \gamma R_{t+2} + ... +\gamma^{T-1} R_t$

Define the $n$-step Q-return:
$$
    q_t^{(n)} = R_{t+1} + \gamma R_{t+2} + ... + \gamma^{n-1} R_{t+n} + \gamma^nQ(S_{t+n})
$$

Sarsa update $Q(s, a)$ towards the n-step Q-return:
$$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left(
        q_t^{(n)} - Q(S_t, A_t)
    \right)
$$

## Forward View Sarsa($\lambda$)

As before, we saw that the n-step return itself is noisy and sensitive to hyperparameter choice of n and $\alpha$. So the better way is to average the value over all $n$ steps.

- The $q^\lambda$ return combines all $n$-step Q-returns $q_t^{(n)}$
- Using weight $(1 - \lambda)\lambda^{n-1}$, we have:
$$
    q_t^{\lambda} = (1 - \lambda) \sum_{n=1}^\infty \lambda^{n-1} q_t^{(n)}
$$
- And the forward view Sarsa($\lambda$) is:
$$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left(
        q_t^\lambda - Q(S_t, A_t)
    \right)
$$

## Backward View Sarsa($\lambda$)

Recall that we used eligibility traces to construct the backward view TD($\lambda$). As the forward view algo is not an online policy - we need to wait until the end of the episode to do the update.
- Just like TD($\lambda$), we use eligibility traces in an online algorithm
- But Sarsa($\lambda$) has one eligibility trace for each state-action pair instead of just for every state
$$
\begin{align*}
    E_0(s, a) &= 0\\
    E_t(s, a) &= \gamma \lambda E_{t-1}(s, a) + \mathbf{1}(S_t=s, A_t=a)
\end{align*}
$$
- $Q(s, a)$ is updated for every state $s$ and action $a$
- In proportion to TD-error $\delta_t$ and eligibility trace $E_t(s, a)$:
$$
\begin{align*}
    \delta_t &= R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)\\
    Q(s, a) &\leftarrow Q(s, a) + \alpha \delta_t E_t(s, a)
\end{align*}
$$

> **Algorithm.** Sarsa($\lambda$) On Policy Algorithm.
> - Initialize $Q(s, a)$ arbitrarily, for all $s \in \S, a \in \A$
> - Repeat (for each episode):
>   - $E(s, a) = 0$, for all $s \in \S, a \in \A$
>   - Initialize $S, A$
>   - Repeat (for each step of episode):
>       - Take action $A$, observe $R, S'$
>       - Choose $A'$ from $S'$, using policy derived from $Q$ (E.g. $\epsilon$-greedy)
>       - $\delta \leftarrow R + \gamma Q(S', A') - Q(S, A)$
>       - $E(S, A) \leftarrow E(S, A) + 1$
>       - For all $s \in \S, a \in \A$:
>           - $Q(s, a) \leftarrow Q(s, a) + \alpha \delta E(s, a)$
>           - $E(s, a) \leftarrow \gamma \lambda E(s, a)$
>       - $S \leftarrow S', A \leftarrow A'$
>   - Until $S$ is terminal

Note that for a given step we have a single value of $\delta$ which is our TD error, but we propagate that to all $s,a$ pairs based on the eligibility trace, as potentially every $s,a$ pair could have contributed to it.

## Off Policy Learning

So far we have been looking at on-policy learning. However it is often useful to do off policy learning, i.e. evaluate a target policy $\pi(a|s)$ to compute $v_\pi(s)$ or $q_\pi(s,a)$, while we follow the behaviour policy $\mu(a|s): \{ S_1, A_1, R_2, ..., S_T \sim \mu \}$. Of course in this case, $\mu \neq \pi$.

Why is off policy learning useful?
- We can learn from observing humans or other agents
- We can re-use experience that was previously generated from old policies $\pi_1, \pi_2, ... \pi_{t-1}$, possibly in a batched manner
- We can learn about the optimal policy while following the exploratory policy
- We can learn about multiple policies while following one policy

First mechanism is <<importance sampling>>. The main idea is to estimate the expectation of a different distribution by re-weighting the distributions:
$$
\begin{align*}
    \E_{X \sim P} [f(X)] &= \sum P(X) f(X)\\
    &= \sum Q(X) \frac{P(X)}{Q(X)} f(X) \\
    &= \E_{X \sim Q} \left[ \frac{P(X)}{Q(X)} f(X) \right]
\end{align*}
$$

We can apply importance sampling to Monte Carlo for <<Off policy monte carlo learning>>:
- We use returns generated from behaviour policy $\mu$ to evaluate $\pi$
- Then we weight the return $G_t$ according to the ratio of probabilities between the two policies
- We need to apply the correction at every time step along the whole episode, because the change in policy affects every time step
$$
    G_t^{\pi / \mu} = \frac{\pi(A_t|S_t)}{\mu(A_t|S_t)}
    \frac{\pi(A_{t+1}|S_{t+1})}{\mu(A_{t+1}|S_{t+1})}
    ...
    \frac{\pi(A_T | S_T)}{\mu(A_T | S_T)} G_t
$$
- And then update the value towards the corrected return
$$
    V(S_t) \leftarrow V(S_t) + \alpha \left( G_t^{\pi / \mu} - V(S_t) \right)
$$

While off policy MC learning is theoretically sound, there are some major problems which make it practically useless in practice:
- Importance sampling dramatically increases variance, as we are adjusting over every time step, and the cumulative effect over the whole episode makes our estimate of $G_t^{\pi / \mu}$ vary wildly
- We also cannot use this adjustment if $\mu$ is zero when $\pi$ is non-zero

So we have to use bootstrapping for importance sampling. This allows us to only adjust the probability for one time step. So we have <<importance sampling for off policy TD>>:
- We use TD targets generated from $\mu$ to evaluate $\pi$
- For TD(0), We weight the TD target $R + \gamma V(S')$ by importance sampling
- This means we only need a single importance sampling correction:
$$
    V(S_t) \leftarrow V(S_t) + \alpha \left(
        \frac{\pi(A_t | S_t)}{\mu(A_t | S_t)}
        (R_{t+1} + \gamma V(S_{t+1}) ) - V(S_t)
    \right)
$$
- This has much lower variance that MC importance sampling, and could work if $\mu$ and $\pi$ do not differ by too much over a single step

As we have seen, importance sampling leads to large variances. The best solution is known as <<Q-learning>>, which is specific to TD(0) or Sarsa(0).
- Does not require any importance sampling
- Allows off policy learning of action values $Q(s, a)$

Recall that $\mu$ is the behaviour policy that our agent is actually following, and $\pi$ is a target policy that we want to learn from. The main idea is that in our Sarsa(0) update step, we update the Q-value towards the *target policy $\pi$*, but allow our agent to continue following the *behaviour policy $\mu$*.

This allows the agent to explore the environment using $\mu$, but learn from the action-value function of $\pi$. Specifically:
- We choose each next action for the agent using behaviour policy $A_{t+1} \sim \mu(. | S_t)$
- But use alternative successor action $A' \sim \pi(. | S_t)$ in our Q-value update
- So we update $Q(S_t, A_t)$ using $A'$:
$$
    Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha(R_{t+1} + \gamma Q(S_{t+1}, A') - Q(S_t, A_t))
$$

Note importantly that we are using $A' \sim \pi$ in the Q value above, instead of $A_{t+1}$. This allows us to learn off policy.

### Q-Learning (or SARSA-MAX)

A special case of Q-learning is the case where the target policy is greedy wrt $Q(s, a)$. This is usually what people refer to as <<Q-learning>>.

We allow both behaviour and target policies to improve:
- The target policy $\pi$ is greedy wrt $Q(s,a)$, i.e.
$$
    \pi(S_{t+1}) = \argmax_{a'} Q(S_{t+1}, a')
$$
- The behaviour policy $\mu$ is $\epsilon$-greedy wrt $Q(s,a)$ again
- The learning target inside the Q-update then simplifies as follows:
$$
\begin{align*}
    &R_{t+1} + \gamma Q(S_{t+1}, A')\\
    &= R_{t+1} + \gamma Q(S_{t+1}, \argmax_{a'} Q(S_{t+1}, a')) \\
    &= R_{t+1} + \max_{a'} \gamma Q(S_{t+1}, a')
\end{align*}
$$

Note that since we are following a greedy target policy, the action chosen will be the Q-maximizing one (line 2). Since we are choosing the Q-maximizing action, we get the maximum Q-value over all possible actions (line 3). This simplifies the equation quite abit, and now it resembles the Bellman optimality equation.

This leads us to the well known Q-learning algorithm, which David calls Sarsa-max. The Q-update is:
$$
    Q(S, A) \leftarrow Q(S, A) + \alpha \left(
        R + \gamma \max_{a'} Q(S', a') - Q(S, A)
    \right)
$$

There is a theorem that tells us that the Q-learning control algorithm converges to the optimal action-value function, i.e. $Q(s, a) \rightarrow q_*(s, a)$

To wrap up, here is a classification of some algorithms we have so far:
|        | Full Backup (Dynamic Programming) | Sample Backup (Temporal Difference) |
|:--------:|:---------:|:----------:|
|Bellman Expectation Equation for $v_\pi(s)$ | Iterative Policy Evaluation | TD Learning |
| Bellman Expectation Equation for $q_\pi(s, a)$ | Q-policy iteration | Sarsa |
| Bellman Optimality Equation for $q_*(s, a)$ | Q-value iteration | Q-learning |