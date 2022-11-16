# Compare multiple Image Classification models in JumpStart

## Overview

At times, when you are solving a business problem using machine learning (ML), you might
want to use multiple ML algorithms and compare them against each other to see which
model gives you the best results on dimensions that you care about - model accuracy,
inference time, and training time.

In this notebook, we demonstrate how you can compare multiple image classification
models and algorithms offered by SageMaker JumpStart on dimensions such as model
accuracy, inference time, and training time. Models in JumpStart are brought from hubs
such as TensorFlow Hub and PyTorch Hub, and training scripts (algorithms) were written
separately for each of these frameworks. In this notebook, you can also alter some of the
hyper-parameters and examine their effect on the results.

Image Classification refers to classifying an image to one of the class labels in the training
dataset.

Amazon SageMaker JumpStart offers a large suite of ML algorithms. You can use JumpStart
to solve many Machine Learning tasks through one-click in SageMaker Studio, or through
SageMaker JumpStart API.

Note: This notebook was tested on ml.t3.medium instance in Amazon SageMaker Studio
with Python 3 (Data Science) kernel and in Amazon SageMaker Notebook instance with
conda_python3 kernel.
