import jax
import jax.numpy as jnp
import optax
from flax import nnx


# Define the model
class NeuralNet(nnx.Module):

    def __init__(self, d_in: int, d_mid: int, d_out: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=d_in,
                                  out_features=d_mid,
                                  rngs=rngs)
        self.linear2 = nnx.Linear(in_features=d_mid,
                                  out_features=d_out,
                                  rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.gelu(x)
        x = self.linear2(x)
        return x


# Training function
def train_model(x_train, y_train):
    model = NeuralNet(d_in=x_train.shape[1],
                      d_mid=10,
                      d_out=1,
                      rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001))

    def train_step(model, optimizer, x, y):

        def loss_fn(model):
            y_pred = model(x)
            loss = jnp.mean((y_pred - y)**2)
            return loss

        loss, grads = nnx.value_and_grad(loss_fn)(model)
        optimizer.update(grads)
        return loss

    for i in range(100):
        loss = train_step(model, optimizer, x_train, y_train)
        print(f"Loss: {loss}")
    return model


# Evaluation function
def evaluate_model(model, x_test, y_test):
    y_pred = model(x_test)
    loss = jnp.mean((y_pred - y_test)**2)
    print(f"Test Loss: {loss}")


# test data
# Sample training data
x_train = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = jnp.array([0, 1, 1, 0])  # XOR dataset

x_test = jnp.array([[0.5, 0.5], [1.0, 0.5]])
y_test = jnp.array([1, 0])

# Run training and evaluation
trained_model = train_model(x_train, y_train)
evaluate_model(trained_model, x_test, y_test)
