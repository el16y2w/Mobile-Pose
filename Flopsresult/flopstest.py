import tensorflow as tf
from src.models.posenet import  PoseNet
from src.models.posenetv2 import PoseNetv2
from src.models.posenetv3 import PoseNetv3
from src.models.hrglassnetv3 import HourGlassNet
from src.models.efficientnetlite0 import EfficientNetLite0
from src.models.efficientnetlite1 import EfficientNetLite1
from src.models.shufflenet import Shufflenetget
from src.models.senet import Senetget
import os
import time
from Config import config
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
def exportflops(inputshape ,model_type, outputFile="/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/PoseEstimation/flopsresult/flopstest.pb", outputName="Output"):
    output = outputFile
    with tf.Session() as sess:
        net, outputshape = build(inputshape ,model_type)
        sess.run(tf.global_variables_initializer())
        input_graph_def = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(
        #tf.compat.v1.graph_util.extract_sub_graph
            sess,
            input_graph_def,
            [outputName]
        )
    with tf.gfile.GFile(outputFile, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    graph = load_pb(outputFile)
    flops, layers = stats_graph(graph)
    return flops, layers, model_type, inputshape, outputshape

def build(inputshape ,model_type, is4Train=True):
    if model_type == "hourglass":
        model = HourGlassNet(is4Train=is4Train)
    else:
        if model_type == "mobilenetv1":
            model = PoseNet(inputshape, is4Train=is4Train)
        elif model_type == "mobilenetv3":
            model = PoseNetv3(inputshape, is4Train=is4Train)
        elif model_type == "efficientnet0":
            model = EfficientNetLite0(inputshape, is4Train=is4Train)
        elif model_type == "efficientnet1":
            model = EfficientNetLite1(inputshape, is4Train=is4Train)
        elif model_type == "shufflenet":
            model = Shufflenetget(inputshape)
        elif model_type == "senet":
            model = Senetget(inputshape)
        else:
            model = PoseNetv2(inputshape, is4Train=is4Train)

        output = model.getOutput()
        output = output.get_shape()

    return model, output


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    return flops.total_float_ops, flops.children

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == '__main__':
    i = 0
    flops, layers, model_type, inputshape, outputshape =exportflops(config.inputshapeforflops[i], config.model[i])
    txt_file = open(os.path.join('/media/hkuit104/24d4ed16-ee67-4121-8359-66a09cede5e7/PoseEstimation/flopsresult',model_type+"res.txt" + time_str), "w+")
    if config.model[i] == 'mobilenetv3':
        txt_file.write("modelname: {}, v3_version: {}, inputshape: {}, outputshape: {}, Offset: {}, Gauthreshold: {}, Gausigma: {}, Flops: {}".
                   format(model_type,config.v3_version,inputshape,outputshape,config.offset,config.threshold,config.sigma,flops))
    else:
        txt_file.write("modelname: {}, inputshape: {}, outputshape: {}, Offset: {}, Gauthreshold: {}, Gausigma: {}, Flops: {}, Layers Flops:{}".
                   format(model_type,inputshape,outputshape,config.offset,config.threshold,config.sigma,flops, layers))
    txt_file.close()
