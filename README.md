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

# Dataset

Dataset/
   sphere/
   cube/
   cylinder/
   cone/

# Results

The tactile classifier achieves good overall accuracy, especially on spheres, cubes, and cones, while cylinders remain more challenging due to less distinctive contact patterns. In multi-object search, the manual strategy performs very well on easier shapes, but struggles on ambiguous ones. PPO improves performance on harder cases but often continues probing even when confidence is already high, leading to unnecessary exploration. Overall, tactile perception quality strongly affects search performance, and improving stopping behavior could offer significant gains.
