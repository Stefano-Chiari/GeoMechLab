# Problem

This checkpoint focuses on predicting the surface displacement as a function of \(x\) in a two-dimensional terrain cross-section composed of multiple layers.

The objective is to build an emulator of the forward problem, in which a selected subsurface layer is subject to an anomalous pressure distribution that induces soil movement. The goal is to predict the resulting displacement at the surface.

# Input Parameters

The model must accept the following inputs:
- the selected layer where the anomaly is applied
- the anomaly pressure function driving the deformation, assumed to be bounded in the range \([-0.5,0.5]\)
- the cavity position uncertainty, modeled as a probabilistic variable defined by the mean and standard deviation of its center

# Uncertainty Propagation

The effect of uncertainty in the cavity position must be incorporated into the model and propagated to the predicted surface displacement. As a second output, the model must provide a 90% confidence interval for both the horizontal (\(x\)) and vertical (\(y\)) components of the displacement at the upper-left corner of the domain.

# Output

The outputs should be:
- the predicted surface displacement profile as a function of \(x\)
- the 90% confidence intervals for both the \(x\)- and \(y\)-components of the displacement at the upper-left corner of the domain

The surface displacement prediction may also include uncertainty information, such as its mean and variance.

# Evaluation

The evaluation will be based on the following criteria:

- **Accuracy (3 points):** quality and correctness of the predicted mean surface displacement
- **Handling of Uncertainty (3 points):** accuracy of the predicted 90% confidence intervals for both displacement components at the upper-left corner of the domain

+ bonus point for best efficiency/accuracy tradeoff.

# Dataset

No dataset will be provided. All necessary simulations and data must be generated within your implementation using the provided tools.