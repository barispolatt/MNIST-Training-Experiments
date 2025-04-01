# MNIST Training Experiments: Baseline & Regularization Techniques

## Overview

This repository contains two complementary experiments on training deep neural networks using the MNIST dataset. The project demonstrates two approaches:

- *Baseline Experiment:* A straightforward training routine of a deep neural network without explicit regularization. It logs the training cross-entropy loss per batch and the validation error per epoch.
- *Regularization Experiment:* An investigation into the effects of common regularization techniques, namely dropout and early stopping (as well as their combination), on training dynamics and overall model performance.

By housing both experiments in a single project, you can easily compare the benefits of incorporating regularization methods against a standard training approach.

## Repository Structure

.
├── baseline_experiment.py         # Baseline training code with loss logging and error analysis
├── regularization_experiment.py   # Code implementing dropout, early stopping, and their combination
├── README.me                      # This file
└── requirements.txt               # List of Python dependencies (TensorFlow, NumPy, Matplotlib)

## Requirements

Ensure you have the following installed:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Matplotlib

You can install the dependencies using pip. For example:

pip install tensorflow numpy matplotlib

Alternatively, if you use a requirements.txt file, run:

pip install -r requirements.txt

## How to Run the Code

### Baseline Experiment

Run the baseline experiment to train a neural network without explicit regularization:

python baseline_experiment.py

This script will:
- Load and preprocess the MNIST dataset (normalizing the images and splitting the data into training, validation, and test sets).
- Build a feed-forward neural network with two hidden layers (500 neurons each) and an output layer.
- Train the model for 250 epochs while logging the cross-entropy loss for each batch.
- Plot:
    - Cross Entropy vs. Batch: Visualizes the training loss over batches.
    - Validation Error Rate vs. Epoch: Displays how the validation error rate evolves over training epochs.
- Evaluate and print the model’s performance on the test set.

### Regularization Experiment

Run the regularization experiment to compare the effects of dropout, early stopping, and their combination:

python regularization_experiment.py

This script will:
- Load and preprocess the MNIST dataset similarly to the baseline experiment.
- Build three model variants:
    - Dropout Model: Includes dropout layers after each hidden layer.
    - Early Stopping Model: Uses an early stopping callback to prevent overfitting.
    - Combined Model: Integrates both dropout and early stopping.
- Train each model variant and log the batch losses, as well as the validation error for every epoch.
- Plot:
    - Batch Loss (First 2000 Batches): Compares training cross-entropy loss for all three variants.
    - Validation Error vs. Epoch: Shows the evolution of the error rate for each method.
    - Train vs. Validation Loss (Combined Model): Provides a detailed look at the training and validation loss curves for the combined regularization approach.
- Evaluate each model variant on the test set and print their performance metrics.

## Experiments and Results

### Baseline Experiment

- *Objective:* Train a deep neural network on MNIST without regularization.
- *Observations:*
    - The training cross-entropy loss steadily decreases over batches.
    - The validation error rate decreases gradually over epochs, typically reaching low error values.
    - The model achieves high accuracy on the test set but may risk overfitting if training continues unchecked.

### Regularization Experiment

- *Objective:* Investigate the impact of dropout and early stopping on training dynamics and model generalization.
- *Observations:*
    - *Dropout:* Introduces noise during training by randomly dropping neurons, which can slow down convergence but helps prevent overfitting.
    - *Early Stopping:* Monitors the validation loss to halt training when improvements stall, thereby reducing overfitting risk and saving computation time.
    - *Combined Approach:* Leverages both techniques to achieve stable training, low validation error, and robust test performance.

## Discussion and Future Work

Both experiments successfully train models on the MNIST dataset with high accuracy. The baseline model demonstrates that deep networks can easily achieve low training loss on MNIST, but the risk of overfitting is higher without regularization. In contrast, the regularization techniques not only mitigate overfitting but also help in reaching a reliable validation performance without excessive training.

Future improvements may include:
- Fine-tuning hyperparameters such as dropout rate, early stopping patience, and learning rate.
- Exploring additional regularization methods (e.g., L1/L2 regularization) or more complex network architectures.
- Extending the experiments to more challenging datasets to further validate the effectiveness of the regularization techniques.

Barış Polat
