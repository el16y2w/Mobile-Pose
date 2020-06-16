#enviroment  pb2tflite

# -*- coding:utf-8 -*-
import tensorflow as tf

in_path = "mobilenetv2False2020-05-05-10-59-15.pb"
#in_path = '/media/hkuit164/Sherry/fastpose-master/parameters/pose_2d/tiny/pose2d.pb'
#out_path = "./model/fastpose_tiny.tflite"
out_path = "mobilenetv2False2020-05-05-10-59-15.tflite"

# 模型输入节点
input_tensor_name = ["Image"]
input_tensor_shape = {"Image":[1,224,224,3]}
# 模型输出节点
classes_tensor_name = ["Output"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(in_path, input_tensor_name, classes_tensor_name, input_shapes=input_tensor_shape)
#converter.post_training_quantize = True
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
converter.allow_custom_ops=True
tflite_model = converter.convert()

with open(out_path, "wb") as f:
    f.write(tflite_model)
