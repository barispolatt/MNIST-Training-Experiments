import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# --- Step 1: Load and prepare the MNIST dataset ---
# Download MNIST data, convert pixel values to [0, 1], and flatten the 28x28 images to 784-length vectors.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = x_train.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# Split the training set into train and validation parts (about 90% train, 10% valid)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, 
                                                  test_size=0.1, random_state=42)

# --- Step 2: Build the neural network ---
# Here, we create a simple feedforward network:
# - Two hidden layers with 500 neurons each using ReLU
# - An output layer with 10 neurons (for the 10 classes) using softmax
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(784,)),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# --- Step 3: Set up training ---
# Use Adam optimizer and sparse categorical cross-entropy as loss
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# --- Step 4: Create callbacks to track loss and error ---
# This callback stores the loss after each batch.
class BatchLossLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.batch_losses = []
    
    def on_batch_end(self, batch, logs=None):
        # Save the loss from this batch
        self.batch_losses.append(logs.get('loss'))

# This callback stores the validation error (1 - accuracy) after each epoch.
class ValErrorLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.val_errors = []
    
    def on_epoch_end(self, epoch, logs=None):
        # Calculate and store the classification error for this epoch
        val_accuracy = logs.get('val_accuracy')
        self.val_errors.append(1 - val_accuracy)

batch_logger = BatchLossLogger()
val_error_logger = ValErrorLogger()

# --- Step 5: Train the model ---
# Run training for 250 epochs without any fancy regularization stuff.
history = model.fit(x_train, y_train,
                    epochs=250,
                    batch_size=128,
                    validation_data=(x_val, y_val),
                    callbacks=[batch_logger, val_error_logger],
                    verbose=2)

# --- Step 6: Evaluate on the test set ---
# Get final loss and accuracy, then compute classification error.
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
test_classification_error = 1 - test_accuracy
print("Test classification error: {:.4f}".format(test_classification_error))

# --- Step 7: Plot the results ---
# Plot training loss per batch and validation classification error per epoch.
plt.figure(figsize=(12, 5))

# Plot for training loss
plt.subplot(1, 2, 1)
plt.plot(batch_logger.batch_losses)
plt.title("Training Loss per Batch")
plt.xlabel("Batch iteration")
plt.ylabel("Cross-Entropy Loss")

# Plot for validation error
plt.subplot(1, 2, 2)
plt.plot(val_error_logger.val_errors)
plt.title("Validation Classification Error per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Classification Error")
plt.show()
