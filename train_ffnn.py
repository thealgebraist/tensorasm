#!/usr/bin/env python3
"""
Train a simple feedforward neural network with 1 hidden layer (16 neurons)
to approximate t -> [sin(t), sin(2t), sin(3t), ..., sin(16t)]
"""
import numpy as np
import json
import os

# Hyperparameters
INPUT_DIM = 1
HIDDEN_DIM = 16
OUTPUT_DIM = 16
LEARNING_RATE = 0.01
EPOCHS = 5000
BATCH_SIZE = 32

def generate_data(n_samples=1000):
    """Generate training data: t -> [sin(t), sin(2t), ..., sin(16t)]"""
    t = np.random.uniform(0, 2*np.pi, (n_samples, 1)).astype(np.float32)
    
    # Create output: [sin(t), sin(2t), sin(3t), ..., sin(16t)]
    y = np.zeros((n_samples, OUTPUT_DIM), dtype=np.float32)
    for i in range(OUTPUT_DIM):
        y[:, i] = np.sin((i + 1) * t[:, 0])
    
    return t, y

def relu(x):
    """ReLU activation"""
    return np.maximum(0, x)

def relu_grad(x):
    """ReLU gradient"""
    return (x > 0).astype(np.float32)

class SimpleFFNN:
    def __init__(self):
        # Initialize weights with Xavier initialization
        self.W1 = np.random.randn(INPUT_DIM, HIDDEN_DIM).astype(np.float32) * np.sqrt(2.0 / INPUT_DIM)
        self.b1 = np.zeros((1, HIDDEN_DIM), dtype=np.float32)
        self.W2 = np.random.randn(HIDDEN_DIM, OUTPUT_DIM).astype(np.float32) * np.sqrt(2.0 / HIDDEN_DIM)
        self.b2 = np.zeros((1, OUTPUT_DIM), dtype=np.float32)
    
    def forward(self, x):
        """Forward pass"""
        self.x = x
        self.z1 = x @ self.W1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2
    
    def backward(self, y_true, y_pred):
        """Backward pass and update weights"""
        batch_size = y_true.shape[0]
        
        # Output layer gradients
        dz2 = (y_pred - y_true) / batch_size
        dW2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(self.z1)
        dW1 = self.x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Update weights
        self.W1 -= LEARNING_RATE * dW1
        self.b1 -= LEARNING_RATE * db1
        self.W2 -= LEARNING_RATE * dW2
        self.b2 -= LEARNING_RATE * db2
    
    def train(self, X, y, epochs, batch_size):
        """Train the network"""
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            total_loss = 0
            for i in range(0, n_samples, batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward and backward pass
                y_pred = self.forward(batch_X)
                self.backward(batch_y, y_pred)
                
                # Compute loss (MSE)
                loss = np.mean((y_pred - batch_y) ** 2)
                total_loss += loss
            
            if (epoch + 1) % 500 == 0:
                avg_loss = total_loss / (n_samples // batch_size)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.6f}")
    
    def save_weights(self, weights_dir):
        """Save weights in binary and text format"""
        os.makedirs(weights_dir, exist_ok=True)
        
        # Save in binary format
        self.W1.tofile(f"{weights_dir}/W1.bin")
        self.b1.tofile(f"{weights_dir}/b1.bin")
        self.W2.tofile(f"{weights_dir}/W2.bin")
        self.b2.tofile(f"{weights_dir}/b2.bin")
        
        # Save in text format
        np.savetxt(f"{weights_dir}/W1.txt", self.W1, fmt='%.6f')
        np.savetxt(f"{weights_dir}/b1.txt", self.b1, fmt='%.6f')
        np.savetxt(f"{weights_dir}/W2.txt", self.W2, fmt='%.6f')
        np.savetxt(f"{weights_dir}/b2.txt", self.b2, fmt='%.6f')
        
        print(f"Weights saved to {weights_dir}/")
    
    def save_config(self, config_path):
        """Save network configuration as JSON"""
        config = {
            "input_dim": INPUT_DIM,
            "hidden_dim": HIDDEN_DIM,
            "output_dim": OUTPUT_DIM,
            "activation": "relu",
            "shapes": {
                "W1": list(self.W1.shape),
                "b1": list(self.b1.shape),
                "W2": list(self.W2.shape),
                "b2": list(self.b2.shape)
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config saved to {config_path}")

def load_and_verify(weights_dir, config_path):
    """Load weights and config, verify they make sense"""
    print("\n" + "="*60)
    print("VERIFICATION: Loading weights and config...")
    print("="*60)
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print("\nLoaded config:")
    print(json.dumps(config, indent=2))
    
    # Load binary weights
    W1_bin = np.fromfile(f"{weights_dir}/W1.bin", dtype=np.float32).reshape(tuple(config["shapes"]["W1"]))
    b1_bin = np.fromfile(f"{weights_dir}/b1.bin", dtype=np.float32).reshape(tuple(config["shapes"]["b1"]))
    W2_bin = np.fromfile(f"{weights_dir}/W2.bin", dtype=np.float32).reshape(tuple(config["shapes"]["W2"]))
    b2_bin = np.fromfile(f"{weights_dir}/b2.bin", dtype=np.float32).reshape(tuple(config["shapes"]["b2"]))
    
    # Load text weights
    W1_txt = np.loadtxt(f"{weights_dir}/W1.txt", dtype=np.float32).reshape(tuple(config["shapes"]["W1"]))
    b1_txt = np.loadtxt(f"{weights_dir}/b1.txt", dtype=np.float32).reshape(tuple(config["shapes"]["b1"]))
    W2_txt = np.loadtxt(f"{weights_dir}/W2.txt", dtype=np.float32).reshape(tuple(config["shapes"]["W2"]))
    b2_txt = np.loadtxt(f"{weights_dir}/b2.txt", dtype=np.float32).reshape(tuple(config["shapes"]["b2"]))
    
    # Verify binary and text match
    print("\nVerifying binary and text formats match:")
    print(f"W1 match: {np.allclose(W1_bin, W1_txt)}")
    print(f"b1 match: {np.allclose(b1_bin, b1_txt)}")
    print(f"W2 match: {np.allclose(W2_bin, W2_txt)}")
    print(f"b2 match: {np.allclose(b2_bin, b2_txt)}")
    
    # Test inference with loaded weights
    print("\nTesting inference with loaded weights:")
    test_inputs = np.array([[0.0], [np.pi/4], [np.pi/2], [np.pi]], dtype=np.float32)
    
    for t_val in test_inputs:
        # Forward pass
        z1 = t_val @ W1_bin + b1_bin
        a1 = relu(z1)
        z2 = a1 @ W2_bin + b2_bin
        
        # Expected output
        expected = np.array([np.sin((i+1) * t_val[0]) for i in range(OUTPUT_DIM)])
        
        print(f"\nInput t = {t_val[0]:.4f}")
        print(f"Predicted: {z2[0][:4]} ... (showing first 4 values)")
        print(f"Expected:  {expected[:4]} ... (showing first 4 values)")
        print(f"MSE: {np.mean((z2[0] - expected)**2):.6f}")
    
    return W1_bin, b1_bin, W2_bin, b2_bin

def main():
    print("Generating training data...")
    X_train, y_train = generate_data(n_samples=1000)
    
    print(f"Training data shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Sample input: {X_train[0]}")
    print(f"Sample output: {y_train[0][:4]} ... (showing first 4 values)")
    
    print("\nInitializing network...")
    model = SimpleFFNN()
    
    print(f"\nTraining network for {EPOCHS} epochs...")
    model.train(X_train, y_train, EPOCHS, BATCH_SIZE)
    
    # Save everything
    weights_dir = "weights/ffnn"
    config_path = "weights/ffnn_config.json"
    
    model.save_weights(weights_dir)
    model.save_config(config_path)
    
    # Load and verify
    load_and_verify(weights_dir, config_path)
    
    print("\n" + "="*60)
    print("Training complete! Weights and config saved.")
    print("="*60)

if __name__ == "__main__":
    main()
