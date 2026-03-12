# deep-rl-learning

A structured exploration of Reinforcement Learning (RL) concepts—with hands‑on implementations, small experiments, and real‑world examples. The goal is to understand RL fundamentals, apply them in practical environments and document all learnings, code, insights, and results in one organised place.

## 📚 Learning Path

### **01 - Fundamentals**
RL basics, N-armed bandits, MCTS, Bellman equation
- See [`01-fundamentals/`](01-fundamentals/README.md)

### **02 - Value-Based Methods**
Classic + deep value-based RL (Q-learning, DQN on Mountain Car & Atari)
- See [`02-value-methods/`](02-value-methods/README.md)

### **03 - Policy-Based Methods**
Policy gradients, REINFORCE, actor-critic, exploration strategies
- See [`03-policy-methods/`](03-policy-methods/README.md)

## 🛠 Setup

**Requirements:** Python 3.10+, `uv`

```bash
uv sync
```

## 📁 Structure

```
├── 01-fundamentals/    # RL intro, bandits, MCTS
├── 02-value-methods/   # Q-learning, DQN
├── 03-policy-methods/  # Policy gradient, actor-critic
├── utils/              # Shared helpers & utilities
├── notebooks/          # Jupyter walkthroughs
└── pyproject.toml      # uv dependencies
```

## 📝 How to Use

1. Start with `01-fundamentals/README.md` for learning objectives
2. Implement practicals in each module's `practicals/` folder
3. Document findings in Jupyter notebooks in `notebooks/`
4. Results/outputs saved to `practicals/results/`

Main References:
- RL Theory I learnt from my masters as part AI Frontier module at Frankfurt school of finance and management.
- Stanford CS224R Deep Reinforcement Learning | Spring 2025 | Lecture 1: Class Intro : https://www.youtube.com/watch?v=EvHRQhMX7_w

