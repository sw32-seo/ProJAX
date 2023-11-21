"""
# The problem
Dividing the dataset into batches can have an incomplete batch at the end.
Dropping the last batch is not an option, because it would mean that we are not using all the data.
Forming a batch with less data is also not an option, because this will trigger a recompliation of `eval_step()`.
The problem is getting worse when we are using multiple devices,
because the number of samples in a batch is divided by the number of devices.

# The solution: padding
Using `flax.jax_utils.pad_shard_unpad()` can solve this problem.

# Adding "infinite padding"
We could add "infinite padding" to the dataset, on each of the hosts independently, and continuing processing eamples until all hosts run out of unpadded examples.
"""

