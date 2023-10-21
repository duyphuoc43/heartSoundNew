import tensorflow as tf

# Kiểm tra số lượng GPU sẵn có
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if len(gpu_devices) > 0:
    print("TensorFlow hỗ trợ GPU.")
else:
    print("TensorFlow không hỗ trợ GPU.")
# import tensorflow as tf
# print("Is GPU available: ", tf.test.is_gpu_available())





