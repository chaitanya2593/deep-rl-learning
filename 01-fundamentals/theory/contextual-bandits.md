# Contextual Bandits

## Overview
Contextual bandits bridge **simple N-armed bandits** and **full RL**. Unlike standard bandits where you choose an arm blindly, contextual bandits observe **context** (state) before acting. This makes them more realistic for real-world problems like recommendation systems, ad placement, and personalization.

**Key Difference:**
- **N-armed bandits:** Choose arm → Get reward (no context)
- **Contextual bandits:** Observe context → Choose arm → Get reward
- **Full RL:** Observe state → Choose action → Get reward + next state (sequential!)

## Core Concepts

### 1. **Context & Policy**
- **Context** `x_t`: Observed state before decision (e.g., user features, item properties)
- **Policy** `π(a|x)`: Maps context to action probabilities
- **Reward** `r_t`: Depends on both action AND context

### 2. **Regret & Exploration-Exploitation**
- **Regret:** Cumulative loss from suboptimal actions
- **Goal:** Minimize regret by learning which actions work best for each context
- **Trade-off:** Exploit known good actions vs. explore uncertain ones

### 3. **Key Algorithms**

| Algorithm | Approach | Use Case |
|-----------|----------|----------|
| **ε-Greedy** | Explore random, exploit best | Simple baseline |
| **UCB (Upper Confidence Bound)** | Optimism under uncertainty | Theoretical guarantees |
| **Thompson Sampling** | Bayesian approach, sample from posterior | Exploration via uncertainty |
| **LinUCB** | Linear model + UCB | Structured, interpretable |


## Real-World Applications

1. **Recommendation Systems** (Netflix, YouTube)
   - Context: User profile, watch history
   - Action: Which content to recommend
   - Reward: Click, engagement, watch time

2. **Ad Placement** (Google Ads, Facebook)
   - Context: User info, page content
   - Action: Which ad to show
   - Reward: Click-through rate (CTR)

3. **Healthcare** (Treatment selection)
   - Context: Patient features
   - Action: Which treatment to prescribe
   - Reward: Patient outcome

4. **E-commerce** (Pricing, layout)
   - Context: Customer demographics
   - Action: Price point or UI variant
   - Reward: Purchase, conversion


## Connection to RL

- **Bandits:** Single decision, immediate reward
- **Contextual Bandits:** Decision depends on state, immediate reward
- **Full RL:** Sequential decisions, state transitions, delayed rewards

Contextual bandits are the **sweet spot**: more realistic than bandits, simpler than full RL.

## References & Resources
- Contextual Bandits - https://www.youtube.com/watch?v=JmbEheH7gVw
- https://www.findingtheta.com/blog/ultimate-guide-to-contextual-bandits-from-theory-to-python-implementation



