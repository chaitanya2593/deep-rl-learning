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
- Exploration (trying different actions to learn)
- Handling uncertainty in the environment
- Finding optimal mixed strategies (e.g., in game theory)

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

## Environment Dynamics

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



To Be Continued