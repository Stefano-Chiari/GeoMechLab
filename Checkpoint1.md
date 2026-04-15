# Problem

This checkpoint focuses on a two-dimensional linear elasticity problem defined on a terrain cross-section composed of multiple layers with different material properties.

The objective is to solve an **inverse problem** aimed at identifying the location of a cavity within the domain. In particular, the task consists in estimating the coordinates of the **center of a cavity** from displacement measurements taken at a single observation point, namely ( (0, 1) ).

The implementation must satisfy the following requirements:

1. **Forward Model**
   The displacement field must be computed using the provided finite element solvers for linear elasticity.

2. **Inverse Procedure**
   Develop an algorithm to estimate the position of the cave center based on the observed displacement at the measurement point.

3. **Multi-Grid Approach**
   The solution must be implemented using two computational grids:

   * a **coarse grid**, for initial approximation of the inverse problem
   * a **refined grid**, for improved accuracy

---

## Evaluation

The evaluation will be based on the following criteria:

* **Accuracy (3 points):** quality of the estimated cave location
* **Efficiency (3 points):** computational performance of the algorithm

+ bonus point

## Dataset

No dataset will be provided. All necessary simulations and data must be generated within your implementation using the provided tools.



