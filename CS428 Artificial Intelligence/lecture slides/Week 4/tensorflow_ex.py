# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Load the MNIST dataset
# The dataset is split into training and test data
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape the data to add a channel dimension (needed for CNN)
# MNIST images are 28x28 pixels, and we need to add a channel dimension (1 for grayscale)
train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# Normalize pixel values to be between 0 and 1 (originally between 0 and 255)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Build the CNN model
model = models.Sequential()

# Add a convolutional layer with 32 filters, kernel size of 3x3, and ReLU activation
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# Add a max-pooling layer with pool size of 2x2
model.add(layers.MaxPooling2D((2, 2)))

# Add another convolutional layer with 64 filters and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# Add another max-pooling layer
model.add(layers.MaxPooling2D((2, 2)))

# Add a final convolutional layer with 64 filters and ReLU activation
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Flatten the output before feeding it into a dense layer
model.add(layers.Flatten())

# Add a fully connected layer with 64 units and ReLU activation
model.add(layers.Dense(64, activation='relu'))

# Add the output layer with 10 units (one for each digit) and softmax activation
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5,
                    validation_data=(test_images, test_labels))

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f'Test accuracy: {test_acc}')
