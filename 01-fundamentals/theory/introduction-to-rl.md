# Introduction to Reinforcement Learning

## What is RL?

**Core Definition:** A system that needs to make multiple decisions based on a stream of information.

**The RL Loop:**
```
Observe State → Take Action → Receive Reward → Observe New State → Repeat
```

RL is fundamentally about learning a policy—a mapping from states/observations to actions—that maximizes long-term cumulative reward.

---

## Supervised Learning vs Reinforcement Learning

### **Supervised Learning (SL)**
- **Input:** Features `X` from a fixed distribution
- **Output:** Target label `Y`
- **Learning:** Direct feedback (ground truth labels)
- **Example:** Image classification, speech recognition
- **Sampling:** Data sampled from a fixed, pre-collected distribution

### **Reinforcement Learning (RL)**
- **Input:** State `S_t` (and observations)
- **Output:** Action `A_t` 
- **Learning:** Indirect feedback (reward signal)
- **Example:** Motor control, chatbots, web agents, game playing
- **Sampling:** Data distribution changes based on the policy we've learned

**Key Difference:**
In SL, the input distribution is fixed. In RL, the distribution of states we see depends on our current policy—this creates a **feedback loop** where our decisions influence the data we collect.

---

## Why Reinforcement Learning?

### **1. Beyond Supervised Learning**
Many real-world problems don't have labeled data. RL can learn from interaction and experience without explicit supervision.

### **2. No Direct Supervision Available**
- **LLM Chatbots (e.g., ChatGPT, Copilot):** Can't directly label "optimal response" for every prompt
- **Autonomous Vehicles:** Impossible to label "correct steering angle" for every scenario
- **Robot Control:** Can't pre-label millions of motor commands

### **3. Learning from Experience is More Fundamental**
RL mimics how humans and animals learn—through trial, error, and reward. This approach:
- Enables discovery of **novel solutions** humans might not have programmed
- Scales to complex domains (AlphaGo, AlphaFold)
- Transfers knowledge across related tasks

### **4. Powering Modern AI**
Most large language models (LLMs) use RL in their training pipeline:
- **SFT (Supervised Fine-Tuning):** Initial training on high-quality data
- **RLHF (RL from Human Feedback):** Fine-tune using human preferences as rewards

---

## How to Represent Data in RL?

### **1. State (`S_t`)**
The complete description of the world at time `t`.
- **Example:** Position and velocity of a robotic arm
- **Property:** Fully informative for decision-making
- **Note:** Often unobserved; we only see observations

### **2. Observation (`O_t`)**
What the agent **actually observes** at time `t` (may be partial/noisy).
- **Example (Chatbot):** Most recent user message sent to the model
- **Example (Vision):** Raw pixel values from a camera
- **Property:** May not contain all information about the true state

**Note:** Observations are like history which can help you define the state. The full observation history can be used to infer the underlying state when the state is not fully observable.

### **3. Action (`A_t`)**
The decision taken by the agent at time `t`.
- **Example:** Motor command to a robotic arm
- **Example (Chatbot):** Generated text/tokens

### **4. Reward (`R_t`)**
Scalar feedback signal indicating how good the (state, action) pair was.
- **Mathematical form:** `r(s, a)` or `r(s, a, s')`
- **Example:** +1 for winning a game, -1 for losing
- **Example (Chatbot):** User satisfaction score, thumbs up/down

**Note on Sparse Rewards:**
In many RL problems, rewards are very sparse (e.g., a robot without a camera inserting a block into an array of shapes). In such cases, it is better to show the robot once or twice what the reward looks like through demonstrations or shaped rewards to guide learning.

### **5. Trajectory**
A sequence of states, observations, and actions over time.
```
τ = (S_0 or O_0, A_0, S_1 or O_1, A_1, S_2 or O_2, ...)
```
The full history of the agent's interaction with the environment.

---

## Chatbot Example: Complete RL Setup

Let's walk through a concrete chatbot example to see how all the RL components come together:

- **Observation (`O_t`):** The user's most recent message
- **Action (`A_t`):** Chatbot's next message (generated response)
- **Trajectory (`τ`):** The history of the current conversation trace: `(O_1, A_1, O_2, A_2, O_3, A_3, ...)`
- **Reward (`R_t`):**
  - `+1` if user gives an upvote
  - `-1` if user gives a downvote
  - `0` if no vote

This example illustrates how a chatbot learns from user feedback over multiple conversation turns.

---

## Policy

**Definition:** A policy, denoted by **π(a|s)**, is a probability distribution over actions given a state.

- **π(a|s):** Probability of taking action `a` when in state `s`
- **Deterministic Policy:** Always takes the same action in a given state (e.g., `a = π(s)`)
- **Stochastic Policy:** Samples actions from a probability distribution (e.g., `a ~ π(·|s)`)

**Note:** In RL, both states (`s_t`) and policies are often **not deterministic**. A stochastic policy can help with:
- **Exploration:** Trying different actions to discover better strategies
- **Modeling Stochastic Behavior:** Capturing uncertainty in the environment
- **Finding Optimal Mixed Strategies:** e.g., in game theory and adversarial settings

---

## Episode (Policy Rollout)

**Definition:** An **episode** (or **policy rollout**) is a complete trajectory from a starting state to a terminal state.

When we follow a policy through all stages, we observe:
```
Episode = (S_0, A_0, R_0, S_1, A_1, R_1, ..., S_T)
```

**Examples:**
- **Game:** One complete game from start to finish
- **Chatbot:** One full conversation until the user closes the chat
- **Robot Task:** One attempt at inserting a block (success or failure)

Each episode provides training data for improving the policy.

---

## Goal of Reinforcement Learning

The fundamental objective of RL is to find a policy that **maximizes the sum of rewards**:

```
max_π  Σ_{t=0}^{T} r(s_t, a_t)
```

Where:
- **π** is the policy we're optimizing
- **r(s_t, a_t)** is the reward received at time `t`
- **T** is the episode length (may be finite or infinite)

**Key Points:**
- We maximize **cumulative reward**, not just immediate reward
- Both `(s_t, a_t)` and the policy π are typically **not deterministic**
- This may require balancing short-term vs. long-term gains
- In practice, we often use discounted rewards to prioritize near-term rewards

---

## Discount Factor (γ)

The **discount factor (γ)**, a number between 0 and 1, is crucial in RL for balancing immediate vs. long-term gains:

**Mathematical Representation:**
```
Return = Σ_{t=0}^{T} γ^t * r_t
```

**Why We Need It:**
- Reduces the value of future rewards exponentially
- Ensures the total return stays mathematically finite (prevents infinite sums)
- Reflects the intuition that immediate rewards are more certain and valuable

**Interpretation:**
- **γ closer to 0 (e.g., 0.1):** Agent is more **greedy**—cares mostly about recent/immediate rewards
- **γ closer to 1 (e.g., 0.99):** Agent is more **farsighted**—values long-term rewards equally with immediate ones
- **Typical choice:** γ = 0.99 (prioritizes long-term gains but not infinitely far in the future)

---

## Value Functions

### **How Do We Know a Policy is Best?**

We need a way to **evaluate** and **compare** policies. This is where **value functions** come in.

### **Value Function: V(s)**

The **value function** `V(s)` estimates the expected cumulative discounted reward starting from state `s` under a given policy π:

```
V^π(s) = E[Σ_{t=0}^{T} γ^t * r_t | S_0 = s, policy = π]
```

**Interpretation:** How "good" is state `s` if we follow policy π?

### **Q-Value Function: Q(s, a)**

The **Q-value function** `Q(s, a)` estimates the expected cumulative discounted reward from taking action `a` in state `s`, then following policy π:

```
Q^π(s, a) = E[Σ_{t=0}^{T} γ^t * r_t | S_0 = s, A_0 = a, policy = π]
```

**Interpretation:** How "good" is the action `a` in state `s`?

**Relationship:** `V(s) = Σ_a π(a|s) * Q(s, a)` — the value of a state is the expected Q-value over all possible actions.

**Using Q-Values for Policy Improvement:**
- **Greedy Policy:** Always pick the action with the highest Q-value: `a* = argmax_a Q(s, a)`
- **Epsilon-Greedy:** Pick the best action most of the time, but explore randomly occasionally

---

## RL Algorithms: Key Approaches

### **🌱 Why So Many RL Algorithms?**

Different RL algorithms are like different tools in a toolbox. You wouldn't use a hammer for every job — same with RL:

- Some algorithms learn fast but are unstable.
- Some are super stable but slow.
- Some work well when actions are continuous (like steering a car).
- Others work only when actions are discrete (like moving up/down/left/right).
- Some need lots of data; some can work with very little.

That's why we have many algorithms — each shines under different conditions.

---

### **1. Imitation Learning – "Monkey See, Monkey Do"**

#### **Simple idea:**
Teach the AI to copy an expert.

#### **Imagine:**
You show a robot how to stack blocks, and it learns by copying your moves.

#### **How it works:**
- Gather recordings of an expert (human or trained model).
- The AI learns to imitate those moves.

#### **Pros:**
- Learns quickly because it doesn't explore on its own.

#### **Cons:**
- It can't get better than the teacher.
- If the expert makes mistakes, the AI learns them too.

---

### **2. Policy Gradient Methods – "Learn the Best Way to Act Directly"**

#### **Simple idea:**
Instead of learning "how good each action is," the AI directly learns **how to act**.

#### **Imagine:**
Trying different ways of swinging your arm until you find a motion that throws a ball farthest. You tweak your movement a little each time.

#### **How it works:**
- Try a behavior → see reward → tweak the behavior.
- Repeat this millions of times.

#### **Pros:**
- Great for continuous controls (robots, self-driving cars).
- Can produce "smooth" or "randomized" behaviors naturally.

#### **Cons:**
- Needs lots of trials (sample inefficient).
- Noisy learning — sometimes makes big mistakes.
- Can settle for "good enough" instead of truly optimal.

**Examples:** REINFORCE, PPO (Proximal Policy Optimization)

---

### **3. Value-Based Methods – "Score Each Action and Pick the Best"**

#### **Simple idea:**
The AI learns a *score* for every possible action in every situation.

#### **Imagine:**
A game where you learn that "jump now gives +10 points" and "don't jump gives +5." You choose the higher score.

#### **How it works:**
- Learn **Q(s, a)** = "How good is doing action **a** in state **s**?"
- Then choose the best-scoring action.

#### **Pros:**
- Usually learns faster than policy gradients.
- Can reuse old experience logs ("off-policy"), which speeds things up.

#### **Cons:**
- Doesn't work naturally with continuous actions (infinite possibilities).
- Can give unstable training with neural networks (DQN).
- Tends to overestimate values sometimes.

**Examples:** Q-Learning, DQN (Deep Q-Networks), SARSA

---

### **4. Actor-Critic – "Two Brains Working Together"**

#### **Simple idea:**
You have:
- **Actor** → decides what to do
- **Critic** → judges how good that action was

Just like a student guided by a coach.

#### **How it works:**
- Actor proposes an action
- Critic evaluates it
- Actor improves based on feedback
- Critic also learns to evaluate better over time

#### **Pros:**
- More stable than pure policy gradients.
- Good for continuous actions.
- More sample efficient.

#### **Cons:**
- Two networks → more complex.
- More hyperparameters to tune.

**Examples:** A3C, TD3, SAC (Soft Actor-Critic)

---

### **5. Model-Based Methods – "Learn the Rules of the World, Then Plan"**

#### **Simple idea:**
Instead of learning only by trial-and-error, the AI tries to **understand how the world works**.

#### **Imagine:**
Before moving a chess piece, you imagine how the board will look afterward. That imagination is the "model."

#### **How it works:**
- Learn how the environment changes after each action.
- Use this learned model to "think ahead" (planning).

#### **Pros:**
- Very sample efficient — less trial-and-error needed.
- Can simulate many futures without interacting with the real world.
- Useful for robotics, offline RL, and transfer learning.

#### **Cons:**
- Hard to learn an accurate world-model.
- Small prediction errors can snowball.
- More computation required.

**Examples:** Dyna, MCTS (Monte Carlo Tree Search), MuZero

---

### **Quick Comparison: When to Use What?**

| Situation | Best Choice |
|-----------|-------------|
| **Lots of data, continuous actions** | Policy Gradient or Actor-Critic |
| **Limited data, continuous actions** | Actor-Critic or Model-Based |
| **Discrete actions, fast learning needed** | Value-Based (Q-Learning, DQN) |
| **Want to reuse past experience** | Value-Based (off-policy) or Actor-Critic |
| **Need sample efficiency** | Model-Based or Actor-Critic |
| **Want simplicity** | Value-Based or Policy Gradient |
| **Starting from expert demo** | Imitation Learning → then Actor-Critic |

---

## Data Collection in RL

How we collect data significantly impacts learning. Key considerations:

### **Action Space Dimensionality**
- **Discrete Action Space:** Finite number of actions (e.g., Left/Right/Up/Down). Easier for value-based methods.
- **Continuous Action Space:** Infinite actions in a range (e.g., motor torques). Better suited for policy gradient and actor-critic methods.

### **Data Collection Strategies**

1. **On-Policy:** Collect data using the current policy being trained
   - Only use data from the policy being improved
   - Less sample efficient but easier to implement
   - Example: REINFORCE, PPO

2. **Off-Policy:** Collect data with one policy, learn another
   - Reuse past data from old policies
   - More sample efficient
   - Example: Q-Learning, DQN, off-policy actor-critic

### **Stability and Ease of Use**
- **Stability:** Some algorithms (e.g., Q-Learning) can diverge or oscillate
- **Ease of Use:** Actor-Critic methods are often easier to tune than pure policy gradients
- **Trade-offs:** More stable often means more complex implementation



### **Markov Property**
The future state depends **only on the current state and action**, not the entire history.

```
P(S_{t+1} | S_t, A_t) = P(S_{t+1} | S_t, A_t, S_{t-1}, A_{t-1}, ...)
```

**In other words:**
```
S_3 depends on (S_2, A_2)  [NOT on S_1, A_1, ...]
```

This is the **Markov property**—the environment is memoryless beyond the current state.


