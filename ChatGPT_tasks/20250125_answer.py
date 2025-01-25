import jax
import jax.numpy as jnp
import optax
from flax import nnx

# Sample training data (XOR dataset)
x_train = jnp.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = jnp.array([[0], [1], [1], [0]])  # XOR labels

x_test = jnp.array([[0.5, 0.5], [1.0, 0.5]])
y_test = jnp.array([[1], [0]])


# Define the model
class LogisticRegression(nnx.Module):

    def __init__(self, d_in: int, d_out: int, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(in_features=d_in,
                                  out_features=d_out,
                                  rngs=rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.sigmoid(x)
        return x


# Training function
def train_model(x_train, y_train):
    model = LogisticRegression(d_in=x_train.shape[1],
                               d_out=1,
                               rngs=nnx.Rngs(42))
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=0.001))

    def train_step(model, x, y):

        def loss_fun(model, x, y):
            y_pred = model(x)
            loss = optax.sigmoid_binary_cross_entropy(logits=y_pred, labels=y)
            return loss.mean()

        loss, grads = nnx.value_and_grad(loss_fun)(model, x, y)
        optimizer.update(grads)
        return loss

    for i in range(100):
        loss = train_step(model, x_train, y_train)
        print(f"Loss: {loss}")
    return model


# Evaluation function
def evaluate_model(model, x_test, y_test):
    y_pred = model(x_test)
    loss = optax.sigmoid_binary_cross_entropy(logits=y_pred, labels=y_test)
    return loss


# Run training and evaluation
trained_model = train_model(x_train, y_train)
evaluate_model(trained_model, x_test, y_test)
