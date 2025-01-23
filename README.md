📌 Project Description – Reinforcement Learning for a Racecar Agent 🚗💨

🏁 Project Overview

This project focuses on implementing and analyzing Temporal Difference (TD) learning and Monte-Carlo (MC) methods in a simulated Racetrack environment. The goal is to train an AI agent to drive a car around a racetrack as efficiently as possible while avoiding gravel and staying within track boundaries.

The project involves:

- On-policy and off-policy learning using Monte-Carlo methods.
- TD-learning approaches such as SARSA and Q-learning.
- Exploration-exploitation trade-offs using ε-greedy policies.
- Value function estimation and policy optimization.

🎯 Main Objectives

- Implement TD and Monte-Carlo algorithms for reinforcement learning.
- Train an agent to drive optimally using the Racetrack environment.
- Compare the effectiveness of on-policy vs. off-policy learning.
- Visualize training performance through reward convergence plots.

🏎️ Racetrack Environment

The Racetrack is a grid-based track where the agent must reach the finish line while controlling its acceleration:
- State Space: The car’s position (x, y) and velocity (vx, vy).
- Action Space: Acceleration choices (ax, ay), including speed up, slow down, or maintain velocity.
- Rewards:
  - -1 per step (encourages faster completion).
  - +50 for reaching the finish line.
  - -1000 for hitting gravel (episode ends).
  - -500 for running out of time.
    
🔥 Implemented Algorithms

The project implements various RL algorithms for policy evaluation and control:

- Monte-Carlo Methods
  - First-visit MC (FV-MC): Estimates value functions using the first visit of each state-action pair.
  - Every-visit MC (EV-MC): Updates value functions for every occurrence of a state-action pair.
  - Off-policy MC: Uses importance sampling to improve a target policy while learning from a different behavior policy.

- Temporal Difference (TD) Learning
  - SARSA (On-policy TD control): Updates Q-values using the next chosen action.
  - Q-learning (Off-policy TD control): Updates Q-values using the max estimated return from the next state.

📊 Key Findings

- Monte-Carlo methods work well for large state spaces but require many episodes for convergence.
- SARSA is more conservative and performs better in stochastic environments.
- Q-learning converges faster due to its off-policy nature.
- Off-policy learning is useful but needs careful handling of importance sampling ratios.


This project explores fundamental reinforcement learning concepts through a Racetrack agent trained with Monte-Carlo and TD learning methods. It demonstrates policy evaluation, value function estimation, and control techniques in an interactive environment.
