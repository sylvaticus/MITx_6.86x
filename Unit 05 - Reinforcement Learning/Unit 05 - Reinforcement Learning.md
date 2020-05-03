$\newcommand\array[1]{\begin{bmatrix}#1\end{bmatrix}}$ [[MITx 6.86x Notes Index]](https://github.com/sylvaticus/MITx_6.86x)

<!-- <img src="https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit 05 - Reinforcement Learning/assets/.png" width="500"/> -->

# Unit 05 - Reinforcement Learning

## Lecture 17. Reinforcement Learning 1

### 17.1. Unit 5 Overview

This unit covers reinforcement learning, a situation somehow similar to
supervised learning, but where instead of receiving feedback for each action/decision/output, the algorithm get a overall feedback only at the very end.
For example, in "playing" a game like Chess or [Go](https://www.youtube.com/watch?v=WXuK6gekU1Y), we don't give to the machine a reward for every action, but what counts whether the whole game has been successful or not.

So we will start approaching the problem of reinforcement learning by looking at a more simplified scenario, which is called **Markov Decision Processes (MDPs)**, where the machine needs to identify the best action plan when it knows all possible rewards and transition between states (more on these terms soon).
Then, we are going to lift some of the assumptions and look at the case of full reinforcement learning, when we experience different transitions without knowing their implicit reward, and collect them only at the end, which is a typical case in life.

In the project at the end of this unit we will implement an agent that
can play text-based games.

## 17.2. Learning to Control: Introduction to Reinforcement Learning

This lesson introduces reinforcement learning. The kind of tasks we can solve are more extended than supervised learning, as the algorithm doesn't need te be given an info of the payoff at each single step. And in many real-world problems we cannot get supervision to the algorithm at every point of the progression.

In real world, we go, we try different things, and eventually some of them succeed and some of them don't.
And as intelligent human being, we actually know to trace back an attribute, what made our whole exploration successful.

For example, a mouse learning to escape a maize and get some food, learns how to exit the maize even if he doesn't get a reward at each crossing, but only at the end of exiting the maize.

In many occasions we are faced with the same scenarios: computer or board games, robots learning to arrive to a certain point (Andrew Ng PhD dissertation was about teaching a robot helicopter to fly), but also the business decisions, e.g. call, email, sending gift to lead a client to sign a contract (note that each of this option may have, taken individually, a negative reward, as they imply costs).

Note that in some cases you may also get some rewards in the middle, but still what must count, is the final situation, if the goal has been reached or not.

So these are the type of problems that are related to reinforcement learning, where our agent can take a lot of different actions, and what counts is the reward this agent gets or doesn't get at the end.
And the whole question of reinforcement learning is how can we get this ultimate reward that we will get and propagate it back and learned how to behave.

Defined the objectives and the type of problems that can be solved with reinforcement learning, the next point of this lecture will discuss how to formalize these problems, setting the problem in clear mathematical terms.

We will first introduce Markov Decision Processes, introducing concepts as state, reward, actions, and so on.

We will then discuss the so called **Bellman equations** that allow us to propagate the goodness of the state from the reward state to all the other state, including the initial one.
Based on this discussion  we would be able to formulate value iteration algorithms that can solve the Markov Decision Processes.

Across the remaining of the exposition in this lecture we will consider the following, very simple, example.

In the following table of 3 X 3 cells, a "robot" has to learn to reach the top-right corner, getting then a $+1$ reward, while if it ends in the lower cell it gets a negative reward $-1$. Also there will be some contraints, like being forbidden to go to the central cell.

Our goal is to find a strategy to send the robot on that particular top-right corner

<img src="https://github.com/sylvaticus/MITx_6.86x/raw/master/Unit 05 - Reinforcement Learning/assets/mdp_example.png" width="500"/>

### 17.3. RL Terminology


The terminology for today:

- **states** $s \in S$: in this lecture they are assumed to be all observed, i.e. the robot know in which state it is (the "observed case")
- **actions** $a \in A$: What we can actually do in this space, in this example going one cell up/down/right or left (and if we hit the border we remain in the same state)
- **transition** $T(s,a,s^\prime) = p(s^\prime |s,a)$: The probability to end up to state $s^\prime$ conditional to being in state $s$ and performing action $a$. For example, starting in the bottom-left cell and performing action "going up" you may have 80% probability of actually going up, 10% of going on the adiacent right cell, and 10% of remaining in the same cell. Note that $\sum_{s^\prime \in S} T(s,a,s^\prime) = 1$
- **reward** $R(s,a,s^\prime)$: The reward for ending up in state $s^\prime$ conditional to being in state $s$ and performing action $a$. In the example here the reward can be defined as only a function of $s$ (i.e. the _leaving_ cell), but more generically (as in the example of marketing where each action involves a cost) it is a function also of $a$ and $s^\prime$.

This problem in a deterministic set up would just require planning, nothing particular.

But here we frame the problem in a stochastic way, that is, there is a probability that even if the chosen action was to move to the top cell, the robot ends up instead to an other cell. This is why we introduce transitions.
Note: this is very similar to the Markov Chains studied in the Probability course (6.431x Probability–The Science of Uncertainty and Data, Unit 10). There, there is a simple transition matrix. Here the transition matrix depends from the action employed by the agent.

[Indeed](https://en.wikipedia.org/wiki/Markov_decision_process): _"A Markov decision process (MDP) is a discrete time stochastic control process. It provides a mathematical framework for modeling decision making in situations where outcomes are partly random and partly under the control of a decision maker.
Markov decision processes are an extension of Markov chains; the difference is the addition of actions (allowing choice) and rewards (giving motivation). Conversely, if only one action exists for each state (e.g. "wait") and all rewards are the same (e.g. "zero"), a Markov decision process reduces to a Markov chain."_


In MDP we assume that the sets of states, actions, transitions, and rewards are given to us: $MDP = ~ <S,A,T,R>$. MDPs satisfy the Markov property in that the transition probabilities, the rewards and the optimal policies depend only on the current state and action, and remain unchanged regardless of the history (i.e. past states and actions) that leads to the current state.

We can consider several variants of this problem, like assuming a particular initial state or a state whose transitions are all zero except those leading to the state itself, e.g. once arrived on the rewarding cell on the top-right corner you can't move any more, you are done.

When we will start thinking about more realistic problem, we will want to have a more complicated definition of transition, function, and the rewards.
But for now, for what we're discussing today, we can just imagine them as tables, i.e. constant across the steps.

#### Reward vs cost difference
What is the difference between _reward_ and _cost_ ?

The reward generally refers to the value an agent might receive for being in a particular state. This value is used to positively "reinforce" the set of actions taken by the agent to get itself into that state.

The cost generally refers to the value an agent might have to "pay", "expend", or "lose" in order to take an action.

Think about there are things you want to do (e.g. pass this course): doing this would result in some reward (e.g. getting a degree or qualification). But doing this isn't free: you have to spend time studying, etc. which can be seen as "costly". In other words, you're going to have to spend something (in this case, your time) in order to reach that rewarding state (in this case, passing this course).

In the same way, we can model RL settings where an agent is told that a particular state has a reward (completing a maze gives the agent +100) but each action it takes has a cost (walking forward, backward, left, or right gives the agent -1). In this way, the agent will be biased towards finding the most efficient/cheapest way to complete the maze.

### 17.4. Utility Function

The main problem for MDPs is to optimize the agent's behavior. To do so, we first need to specify the criterion that we are trying to maximize in terms of accumulated rewards.

We will define a utility function and maximize its expectation. Note that this should be a finite number, in order to compare different possible strategies, to find which is the best one.

We consider two different types of "bounded" utility functions:

- **Finite horizon based utility** : The utility function is the sum of rewards after acting for a fixed number $n$ steps. For example, in the case when the rewards depend only on the states, the utility function is $U[s_0,s_1,\ldots , s_n]= \sum_{i=0}^{n} R(s_i)$ for some fixed number of steps $n$.
In particular $U[s_0,s_1,\ldots , s_{n+m}]=U[s_0,s_1,\ldots , s_ n]$ for any positive integer $m$

- **(Infinite horizon) discounted reward based utility** : In this setting, the reward one step into the future is discounted by a factor $\gamma$, the reward two steps ahead by $\gamma^2$, and so on. The goal is to continue acting (without an end) while maximizing the expected discounted reward. The discounting allows us to focus on near term rewards, and control this focus by changing $\gamma$. For example, if the rewards depend only on the states, the utility function is $U[s_0,s_1,\ldots ]= \sum_{k=0}^{\infty } \gamma ^ k R(s_k)$.

While tempting for its simplicity, the finite horizon utility is not applicable to our context, as it would drop time-invariance property of the Markov Decision Model in terms of the optimal behaviour. Indeed, as there is a finite step (time), the behaviour of the agent would become non-stationary: it would not only depends from which state it is located, but also on which step we are, on how far we would be from such ending horizon.
So if you arrive to a certain state, and you just have one step to go, you may decide to take a very different step versus if you have still many steps to go and you can do a lot of different things.
So for instance, if you just go one step to go, you may go to extremely risky behaviour because you have no other chances.

We will then employ the discounted reward utility, where $\gamma$ is the **discount factor**, a number between 0 and 1 that measures our "impatience for the future", how greedy we are to get the reward now instead of tomorrow.
In economy it is often given as $\frac{1}{1+r}$ in discrete time applications (as here) or $\frac{1}{e^r}$ in continuous time ones, where $r$ is the "discount rate" and we would then say that the utility is the net present value of this infinite serie of future rewards (a bit like the value of some land is equal to the discounted annuity you can get from that land, i.e. renting it or using it directly.)

But why the discounted reward is bounded?

The discounted utility is $U = \sum_{k=0}^{\infty } \gamma ^ k R(s_k)$ but we can write the inequality $U = \sum_{k=0}^{\infty } \gamma ^ k R(s_k) \leq \sum_{k=0}^{\infty } \gamma ^ k R_{max}$ where $R_{max}$ is the maximal reward obtenible in any state, and then taking it outside the sum and using the geometric series poperties that $\sum_{k=0}^\infty \gamma^k = \frac{1}{1-\gamma}$ for $|\gamma| \leq 1$, we can write that $U \leq \frac{R_{max}}{1-\gamma}$

Going back to the land example, even if you can use or rent some land forever, its value is a finite one, indeed because of the discounting of future incomes.


## 17.5. Policy and Value Functions

We now define a last term, "policy".

Given an MDP, and a utility function $U[s_0,s_1,\ldots , s_ n]$, a **policy** is a function $\pi : S\to A$ that assigns an action $\pi$ to any state $s$. We denote the optimal policy $\pi_s^* ~$ as the optimal action you can take in a given state, in term of maximising the expected utility $\pi_s^* : argmax_{a_s}  E[U(a_s)]$.

Note that the goal of the optimal policy function is to maximize the expected discounted reward, even if this means taking actions that would lead to lower immediate next-step rewards from few states.

The policy depends from the structure of the rewards in the problem. For example, in our original robot example, let's assume that for each action we have to pay a small price, like 0.01. Then the "policy" for our robot would likely be (it then depends from the transitions probabilities) to go round the table to arrive to the $+1$ reward without passing for the $-1$ one. But if instead the reward structure is such that we "pay" each action 10, the policy would likely be to go instead direct to the top-right cell.

Again, we want this policy to be independent on any previous actions or the time step we are, just a function of the state where we are.

When we are talking about solving MDPs, we're talking about finding
these optimal policies.

This problem of solving the MDP is very similar to the reinforcement learning problem, but with the sole difference that when you have an reinforcement learning problem, you're not provided with a transition function and the reward function until you actually go in the world, experience, and collect it.
But for today, we assume these are given.

We will now introduce the Bellman equation. We use it so somehow propagate our rewards.

In our robot example, considering only the $+1$ and $-1$ rewards as given, we need a tool to formalise the intuition that the cell adjacent to the $-1$, while not having any reward by itself, it is a "great place to be", as it is only one step away from the $-1$ reward, the top-left cell a bit less great place, and the bottom-left cell an even less great place.

In other words, we need to provide our agent with some quantification is, how good is the state, which is its value, even if the reward is coming many steps ahead.
So we need to introduce some notion of this value of each state, and what the Bellman equations do, they actually connect this notion of this value of the state and the notion of policy.

The **value function V(s)** of a given state $s$ is defined as the expected reward (i.e. the expectation of the utility function) if the agent acts optimally starting at state *s*.


### Setting the timing in discrete time modelling

One very important aspect of discrete time modelling with discounting is to make clear "when the time starts" and if rewards/costs are beared at the beginning or at the end of the period.

In our case the time start clocking when we leave the departing state. At that time we pay any cost incurred in the action and cash the reward linked to state 1. After one period (so discounted with $\gamma$) we move to the second state paying at that moment the action cost and the reward linked to that state and so on.

Mathematically: $U = R(a_{0 \to 1},s_1) + \gamma * R(a_{1 \to 2},s_2) + \gamma^2 * R(a_{2 \to 3},s_3) + ...$

### 17.6. Bellman Equations


Let's introduce two new definitions:

- the **value function** $V^* (s)$ is the expected reward from starting at state $s$ and acting optimally, i.e. following the optimal policy
- the **Q-function** $Q^* (s,a)$ is the expected reward from starting at state $s$, then acting with action $a$ (not necessarily the optimal action), and acting optimally afterwards.

The first of the two Bellman equations relate the value function to the Q function in terms that the value function is the Q-function when the optimal $a$ is followed:

$\displaystyle  V^* (s) = \max_a Q^* (s,a) = Q^* (s,a = \pi^* (s))$ where $\pi^* ~$ is the optimal policy.

The second Bellman equation relates recursively the Q-function to the reward associated to the action cost and the destination reward (girst term) plus the value function of the destination state (second term):

$\displaystyle Q^* (s, a) =  R(s, a, s') + V^* (s')$

When we consider the discounting and the transition probabilities it becomes $\displaystyle Q^* (s, a) =  \sum_{s'} T(s, a, s') ( R(s, a, s') + \gamma V^* (s'))$

Expressed in these terms the value function becomes:

$\displaystyle  V^* (s) = \max_a  \sum_{s'} T(s, a, s') ( R(s, a, s') + \gamma V^* (s'))$


## 17.7. Value Iteration

Given the Bellman equations would be tempting to try to compute the value of a state starting from the values of the reachable states from the optimal action (that, in turn, can be just picked up looking at all the possible actions).

The problem however is that the V(A), the value of state A, then may depends from V(B), but then V(B) may depends as well from V(A) and many more complicated interrelations.
Aside very simple cases, this make impossible to find an analytical solution or even a direct numerical one.

The trick is then to initialise the values of the states to some values (we use 0 in this lecture) and then starting iteratively to loop to define the value of a state at iteration (k) from the value of the reachable states _that we did memorised in the previous iteration_ from the optimal action :

$\displaystyle  V^* (s_k) = \max_a  \sum_{s\prime} T(s, a, s') ( R(s, a, s') + \gamma V^* (s_{k-1}^{\prime}))$

The algorithm than stop when the differences from the values at $k$ iteration and those at $k+1$ become small enough on each state.

Note that $V^* (s_k)$ can be interpreted also as the expected reward from state $s$ acting optimally for $k$ steps.

Once we know the final values of each state we can loop over each state and each possible action on each state one last time to select the "optimal" action for each state and hence define the optimal policy:

$\displaystyle \pi^* (s) = argmax_a Q^* (s,a) = argmax_a \sum_{s'} T(s, a, s') ( R(s, a, s') + \gamma V^* (s'))$


#### Convergence

While the iterative algorithm described above converges, we are now proving it only for the simplest case of a single state and single action.

In such case the algorithm rule at each iteration would reduce to:

$\displaystyle  V^* (s_k) = ( R(s) + \gamma V^* (s_{k-1}))$

In a fully converged situation the Bellman equation gives us that:

$\displaystyle  V^* (s) = ( R(s) + \gamma V^* (s))$

Putting them together we can write:

$\displaystyle  (V^* (s) - V^* (s_k) ) = \gamma * (V^* (s) - V^* (s_{k-1}) )$.

That is, at each new iteration of the algorithm, the difference between the converged value and the value of the state at that iteration get multiplied by a factor $\gamma$, that being a number between 0 and 1, it means this difference reduces at each iteration, implying convergence.


### 17.8. Q-value Iteration

Instead of computing first the _values_ of the states and then the Q function to retrieve the policy, we can directly, using exactly the same approach, compute the Q function. the algorithm then is named **Q-value iteration**  and is the one we will use in the context of reinforcement learning in the next lecture.

Using the Bellman equations we can write the Q function as:

$\displaystyle Q^* (s, a) = \sum_{s'} T(s, a, s') \left(R(s, a, s') + \gamma \max_{a'} Q^* (s', a')\right)$

The iteration update rule than becomes:

$\displaystyle Q_{k+1}^* (s, a) = \sum_{s'} T(s, a, s')\left(R(s, a, s') + \gamma ~ \text {max}_{a'} Q_k^* (s', a')\right)$








## Lecture 18. Reinforcement Learning 2

## Lecture 19: Applications: Natural Language Processing

## Homework 6

## Project 5: Text-Based Game

[[MITx 6.86x Notes Index]](https://github.com/sylvaticus/MITx_6.86x)
