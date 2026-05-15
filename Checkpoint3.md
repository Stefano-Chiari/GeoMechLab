# Problem

This third checkpoint focuses on solving an inverse problem in a two-dimensional terrain cross-section composed of multiple subsurface layers.

The objective is to build an inversion algorithm based on the emulator developed in checkpoint 2 that, given displacement measurements at known surface locations, reconstructs the subsurface anomaly responsible for the observed deformation.

Specifically, the model must infer:
- the layer containing the pressure anomaly
- the pressure value of the anomaly
- the center of the cavity producing the deformation

# Input Parameters

The model must accept the following inputs:
- surface displacement measurements
- the spatial locations on the surface where the displacement measurements are taken


# Inverse Problem

The unknowns to be reconstructed are:
- the anomalous layer
- the anomaly pressure value, assumed to be bounded in the range \([-0.5, 0.5]\)
- the cavity center location

# Evaluation

The evaluation will be based on the following criteria:

- **Layer Identification Accuracy (1 point):** correctness of the predicted layer containing the anomaly
- **Pressure Reconstruction Accuracy (1 point):** accuracy of the reconstructed anomaly pressure value
- **Cavity Localization Accuracy (2 points):** accuracy of the estimated cavity center
- **Efficiency (2 point):** 

+ bonus point for best efficiency/accuracy tradeoff.

# Dataset

No dataset will be provided. All necessary simulations and data must be generated within your implementation using the provided tools.