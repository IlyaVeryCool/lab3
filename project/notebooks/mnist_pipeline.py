import numpy as np
import struct
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class MNISTPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.model = None

    def load_mnist_images(self, filename):
        """Load MNIST images from file."""
        with open(filename, 'rb') as f:
            _, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.fromfile(f, dtype=np.uint8).reshape(num, rows, cols)
        return images

    def load_mnist_labels(self, filename):
        """Load MNIST labels from file."""
        with open(filename, 'rb') as f:
            _, num = struct.unpack(">II", f.read(8))
            labels = np.fromfile(f, dtype=np.uint8)
        return labels

    def load_data(self):
        """Load train and test data."""
        # self.X_train = self.load_mnist_images(f"{self.data_path}/train-images.idx3-ubyte")
        # self.y_train = self.load_mnist_labels(f"{self.data_path}/train-labels.idx1-ubyte")
        # self.X_test = self.load_mnist_images(f"{self.data_path}/t10k-images.idx3-ubyte")
        # self.y_test = self.load_mnist_labels(f"{self.data_path}/t10k-labels.idx1-ubyte")
        (self.X_train, self.y_train), (self.X_test, self.y_test) = \
            tf.keras.datasets.mnist.load_data()

    def visualize_data(self, num_images=10):
        """Visualize first few images from the dataset."""
        fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
        for i, ax in enumerate(axes):
            ax.imshow(self.X_train[i], cmap='gray')
            ax.set_title(f"Label: {self.y_train[i]}")
            ax.axis('off')
        plt.show()

    def preprocess_data(self):
        """Flatten and normalize data."""
        self.X_train = self.X_train.astype('float32') / 255.0
        self.X_test = self.X_test.astype('float32') / 255.0
        self.X_train = self.X_train.reshape(self.X_train.shape[0], -1)
        self.X_test = self.X_test.reshape(self.X_test.shape[0], -1)

    def validate_inputs(self, X, max_samples=100000, max_features=784):
        """Validate input data."""
        if not isinstance(X, np.ndarray):
            raise ValueError("Input data must be a numpy array.")
        if len(X.shape) != 2:
            raise ValueError(f"Expected 2D array, got shape: {X.shape}.")
        if X.shape[0] > max_samples:
            raise ValueError(f"Too many samples: {X.shape[0]} > {max_samples}.")
        if X.shape[1] > max_features:
            raise ValueError(f"Too many features: {X.shape[1]} > {max_features}.")
        if not (0.0 <= X.min() <= X.max() <= 1.0):
            raise ValueError("Data must be normalized in range [0, 1].")

    def process_in_batches(self, X, batch_size=1000):
        """Process data in batches."""
        for i in range(0, len(X), batch_size):
            yield X[i:i+batch_size]

    def train_model(self, hidden_layer_sizes=(128,), activation='relu', max_iter=20):
        """Train the MLP model."""
        self.model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, 
                                   max_iter=max_iter, random_state=42)
        self.model.fit(self.X_train, self.y_train)

    def evaluate_model(self, flag):
        """Evaluate the trained model and display results."""
        y_pred = self.model.predict(self.X_test)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        accuracy = accuracy_score(self.y_test, y_pred)
        if(flag):
            print("Classification Report:")
            print(classification_report(self.y_test, y_pred))
            print(f"Accuracy: {accuracy:.4f}")
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
		            xticklabels=range(10), yticklabels=range(10))
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.show()
        else:
            return accuracy, conf_matrix

# Example usage
if __name__ == "__main__":
    data_path = "/home/kali/yupiter/project/data"
    mnist_pipeline = MNISTPipeline(data_path)

    mnist_pipeline.load_data()
    mnist_pipeline.visualize_data()
    mnist_pipeline.preprocess_data()

    mnist_pipeline.validate_inputs(mnist_pipeline.X_train)
    mnist_pipeline.validate_inputs(mnist_pipeline.X_test)

    mnist_pipeline.train_model()
    mnist_pipeline.evaluate_model()
