# Target-Conditioned-Shape-Based-Object-Search-Using-Tactile-Sensing

This project explores tactile-only object search for robotic manipulation. Instead of vision, a robot uses pressure-map tactile images to identify an object among four shape categories (sphere, cube, cylinder, cone).

# Summary

- Robot touches objects one by one

- Extracts 2D tactile pressure maps

- Classifier (ConvNeXt-Tiny) predicts shape

- PPO reinforcement learning decides:

  - keep probing

  - switch objects

  - stop if confident

The system requires no visual input, enabling search in occluded or visually degraded environments.

# Features

- Target-conditioned tactile search

- ConvNeXt-Tiny tactile classifier

- PPO exploration strategy

- Simulated multi-object environment
