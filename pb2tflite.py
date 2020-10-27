# #enviroment  pb2tflite

# # -*- coding:utf-8 -*-
# import tensorflow as tf

# in_path = "mobilenetv2False2020-06-24-13-59-35.pb"
# #in_path = '/media/hkuit164/Sherry/fastpose-master/parameters/pose_2d/tiny/pose2d.pb'
# #out_path = "./model/fastpose_tiny.tflite"
# out_path = "mobilenetv2False2020-06-24-13-59-35.tflite"

# # 模型输入节点
# input_tensor_name = ["Image"]
# input_tensor_shape = {"Image":[1,224,224,3]}
# # 模型输出节点
# classes_tensor_name = ["merge_Output/Output_pointwise/BatchNorm/FusedBatchNorm"]

# converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)
# #converter.post_training_quantize = True
# tflite_model = converter.convert()

# with open(out_path, "wb") as f:
#     f.write(tflite_model)

import tensorflow as tf

in_path = "yogaallFalse2020-09-14-10-47-14.pb"
#in_path = '20200427_112613.pb'
#out_path = "hrglass_256_64.tflite"
out_path = "yogaallFalse2020-09-14-10-47-14.tflite"

# 模型输入节点
input_tensor_name = ["Image"]
input_tensor_shape = {"Image":[1,224,224,3]}
# 模型输出节点
classes_tensor_name = ["Output"]

#converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, #classes_tensor_name, input_shapes=input_tensor_shape)

converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)

#converter.post_training_quantize = True
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]


tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)
