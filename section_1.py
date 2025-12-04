import numpy as np

class Layer:
    def forward(self, input):
        """
        Forward pass
        Args:
            input: input data
        Returns:
            output: output data
        """
        raise NotImplementedError
    
    def backward(self, grad_output):
        """
        Backward pass
        Args:
            grad_output: gradient of loss w.r.t. output of this layer
        Returns:
            grad_input: gradient of loss w.r.t. input of this layer
        """
        raise NotImplementedError


class Linear(Layer):
    def __init__(self, in_features, out_features):
        """
        Args:
            in_features: number of input features
            out_features: number of output features
        """
        self.in_features = in_features
        self.out_features = out_features

        self.b = np.zeros(out_features)

        std = np.sqrt(2.0/in_features)*in_features
        self.W = np.random.normal(0, std, (out_features, in_features))
        return
    
    def forward(self, input):
        """
        Forward pass: y = xW^T + b
        Args:
            input: (batch_size, in_features)
        Returns:
            output: (batch_size, out_features)
        """
        self.input = input
        output = np.matmul(input, self.W.T) + self.b
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        Args:
            grad_output: (batch_size, out_features)
        Returns:
            grad_input: (batch_size, in_features)
        """
        self.grad_W = np.matmul(grad_output.T, self.input)
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = np.matmul(grad_output, self.W)
        return grad_input

class ReLU(Layer):
    def __init__(self):
        self.mask = None
        return
    
    def forward(self, input):
        """
        Forward pass:
        Args:
            input: any shape
        Returns:
            output: same shape as input
        """
        self.mask = (input > 0)
        output = input * self.mask
        return output
    
    def backward(self, grad_output):
        """
        Backward pass
        Args:
            grad_output: same shape as input
        Returns:
            grad_input: same shape as input
        """
        grad_input = grad_output * self.mask
        return grad_input
    
def softmax(x):
    x_stable = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(x_stable)
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs

class SimpleNetwork:
    """
    Simple 3-layer neural network
    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear
    """
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        self.linear1 = Linear(input_size, hidden_size1)
        self.relu1 = ReLU()
        self.linear2 = Linear(hidden_size1, hidden_size2)
        self.relu2 = ReLU()
        self.linear3 = Linear(hidden_size2, output_size)
        
        self.layers = [
            self.linear1,
            self.relu1,
            self.linear2,
            self.relu2,
            self.linear3
        ]
        pass
    
    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, grad_output):
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

        return grad_output
    
    
def train_with_gd(network, X_train, y_train, learning_rate, num_epochs):
    def cross_entropy_loss(logits, y):
        probs = softmax(logits)
        batch_size = logits.shape[0]

        log_probs = -np.log(probs[np.arange(batch_size), y])
        loss = np.sum(log_probs) / batch_size

        grad_logits = probs.copy()
        grad_logits[np.arange(batch_size), y] -= 1
        grad_logits /= batch_size

        return loss, grad_logits

    losses = []

    for epoch in range(num_epochs):
        logits = network.forward(X_train)

        loss, grad_logits = cross_entropy_loss(logits, y_train)
        losses.append(loss)

        network.backward(grad_logits)

        for layer in network.layers:
            if isinstance(layer, Linear):
                layer.W -= learning_rate * layer.grad_W
                layer.b -= learning_rate * layer.grad_b
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss: .4f}")

    return losses


if __name__ == "__main__":
    np.random.seed(42)
    
    train_all = np.loadtxt('data/smallTrain.csv', dtype=int, delimiter=',')
    X_train = train_all[:, 1:]  
    y_train = train_all[:, 0]   

    valid_all = np.loadtxt('data/smallValidation.csv', dtype=int, delimiter=',')
    X_val = valid_all[:, 1:]   
    y_val = valid_all[:, 0]   

    input_size = 128    
    hidden_size1 = 128  
    hidden_size2 = 64   
    output_size = 10   

    network = SimpleNetwork(input_size, hidden_size1, hidden_size2, output_size)
    
    # Training parameters
    learning_rate = 0.1
    num_epochs = 100
    losses = train_with_gd(network, X_train, y_train, learning_rate, num_epochs)

    print(f"\nFinal Results:")

    # Training set evaluation
    logits_train = network.forward(X_train)
    probs_train = softmax(logits_train)
    predictions_train = np.argmax(probs_train, axis=1)
    accuracy_train = np.mean(predictions_train == y_train)

    print(f"  Training Loss: {losses[-1]:.4f}")
    print(f"  Training Accuracy: {accuracy_train:.2%}")

    # Validation set evaluation
    logits_val = network.forward(X_val)
    probs_val = softmax(logits_val)
    predictions_val = np.argmax(probs_val, axis=1)
    accuracy_val = np.mean(predictions_val == y_val)

    print(f"\n  Validation Accuracy: {accuracy_val:.2%}")