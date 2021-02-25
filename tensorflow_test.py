import tensorflow as tf

#print tensorflow version
print(tf.__version__)

# Print all available devices
print("physical_devices:")
tf.config.list_physical_devices(
    device_type=None
)

#GPUs
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs (physical):", len(physical_devices))

logical_devices = tf.config.list_logical_devices('GPU')
print("Num GPUs (logical):", len(logical_devices))

