import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 1. Load and preprocess data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Use the last 5000 samples for validation
x_val = x_train[-5000:]
y_val = y_train[-5000:]
x_train = x_train[:-5000]
y_train = y_train[:-5000]

# Flatten images from 28x28 to 784
x_train = x_train.reshape(-1, 784)
x_val   = x_val.reshape(-1, 784)
x_test  = x_test.reshape(-1, 784)

# 2. Define a model function with an option for dropout
def create_model(use_dropout=False):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(500, activation='relu', input_shape=(784,)))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(500, activation='relu'))
    if use_dropout:
        model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10, activation='softmax'))
    return model

# 3. Callback to log batch loss
class BatchLossLogger(tf.keras.callbacks.Callback):
    def _init_(self):
        super()._init_()
        self.batch_losses = []
    def on_train_batch_end(self, batch, logs=None):
        self.batch_losses.append(logs.get('loss'))

# 4. Early stopping callback
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# 5. Function to train a model and return losses
def train_model(use_dropout=False, use_early_stopping=False, max_epochs=250):
    model = create_model(use_dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    batch_logger = BatchLossLogger()
    callbacks = [batch_logger]
    if use_early_stopping:
        callbacks.append(early_stopping_cb)
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=max_epochs,
        batch_size=128,
        callbacks=callbacks,
        verbose=0
    )
    batch_losses = batch_logger.batch_losses
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_accuracy']
    val_error = [1.0 - acc for acc in val_acc]
    return batch_losses, val_error, train_loss, val_loss

print("Training models...")

# 6. Train all models
# Baseline: no dropout, no early stopping
bl_batch_losses, bl_val_error, bl_train_loss, bl_val_loss = train_model(use_dropout=False, use_early_stopping=False)
# Dropout: with dropout only
dp_batch_losses, dp_val_error, dp_train_loss, dp_val_loss = train_model(use_dropout=True, use_early_stopping=False)
# Early Stopping: with early stopping only
es_batch_losses, es_val_error, es_train_loss, es_val_loss = train_model(use_dropout=False, use_early_stopping=True)
# Combined: with dropout and early stopping
co_batch_losses, co_val_error, co_train_loss, co_val_loss = train_model(use_dropout=True, use_early_stopping=True)

print("Models trained.")

# 7. Plot graphs (exclude baseline from graphs)
fig = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])
ax1 = fig.add_subplot(gs[0, 0])  # Batch Loss graph
ax2 = fig.add_subplot(gs[0, 1])  # Validation Error graph
ax3 = fig.add_subplot(gs[1, :])  # Combined Model Loss graph

# Graph 1: Training Batch Loss for first 2000 batches (Dropout, Early Stopping, Combined)
ax1.plot(dp_batch_losses[:2000], label='Dropout', color='green')
ax1.plot(es_batch_losses[:2000], label='Early Stopping', color='orange')
ax1.plot(co_batch_losses[:2000], label='Combined', color='red')
ax1.set_title('Batch Loss (First 2000 Batches)')
ax1.set_xlabel('Batch')
ax1.set_ylabel('Loss')
ax1.legend()

# Graph 2: Validation Error per epoch (Dropout, Early Stopping, Combined)
ax2.plot(dp_val_error, label='Dropout', color='green')
ax2.plot(es_val_error, label='Early Stopping', color='orange')
ax2.plot(co_val_error, label='Combined', color='red')
ax2.set_title('Validation Error')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Error Rate')
ax2.legend()

# Graph 3: Combined Model Loss Curves (Train vs Validation)
ax3.plot(co_train_loss, label='Train', color='red')
ax3.plot(co_val_loss, label='Validation', color='purple')
ax3.set_title('Combined Model Loss')
ax3.set_xlabel('Epoch')
ax3.set_ylabel('Loss')
ax3.legend()

plt.tight_layout()
plt.show()

# 8. Evaluate models on test set (including baseline)
def evaluate_model(use_dropout, use_early_stopping, model_name):
    model = create_model(use_dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    callbacks = []
    if use_early_stopping:
        callbacks.append(early_stopping_cb)
    model.fit(
        x_train, y_train,
        epochs=10,
        batch_size=128,
        validation_data=(x_val, y_val),
        verbose=0,
        callbacks=callbacks
    )
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"{model_name}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc:.4f}")

print("\nTest Set Performance:")
evaluate_model(False, False, "Baseline")
evaluate_model(True, False, "Dropout")
evaluate_model(False, True, "Early Stopping")
evaluate_model(True, True, "Combined")
