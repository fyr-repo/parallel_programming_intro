import tensorflow as tf
import time
import os

# Check if TPU is available
try:
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
    tf.config.experimental_connect_to_cluster(resolver)
    # This is the TPU initialization code that has to be at the beginning.
    tf.tpu.experimental.initialize_tpu_system(resolver)
    print("All devices: ", tf.config.list_logical_devices('TPU'))
except:
    print("TPU not available")

tpu_strategy = tf.distribute.TPUStrategy(resolver)

# Define the dimensions and values of the arrays
rows_cols = 1000
value = 55.55

# Create the input arrays filled with the specified value
array1 = tf.constant([[value] * rows_cols] * rows_cols, dtype=tf.float32)
array2 = tf.constant([[value] * rows_cols] * rows_cols, dtype=tf.float32)

start_time = time.time()

with tpu_strategy.scope():
    result = tf.linalg.matmul(array1, array2)

# result = strategy.run(matmul_fn, args=(matrix_a, matrix_b))

end_time = time.time()

processing_time = end_time - start_time

# Print the result (will be executed on the TPU)
print("Matrix multiplication result:")
print(result)
print(f"Time taken for processing: {processing_time:.4f} seconds")

# 0.0009 seconds