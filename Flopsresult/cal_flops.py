import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
from src.models.Layerprovider import LayerProvider
class PoseNet():

    def __init__(self, is4Train=True, mobilenetVersion=0.75, totalJoints=13):

        #tf.reset_default_graph()

        lProvider = LayerProvider(is4Train)

        adaptChannels = lambda totalLayer: int(mobilenetVersion * totalLayer)

        self.inputImage = tf.placeholder(tf.float32, shape=(1, 144, 144, 3), name='Image')

        output = lProvider.convb(self.inputImage, 3, 3, adaptChannels(32), 2, "1-conv-32-2-1", relu=True)
        #print("1-conv-32-2-1 : " + str(output.shape))

        # architecture description

        channels = [64, 128, 128, 256, 256, 512, 512, 512, 512, 512, 512, 1024, 1024]
        strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 1, 1]
        dilations = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 2]

        intermediateSupervision = [6]

        self.intermediateSupervisionOutputs = []

        for layerId in range(len(channels) - 1):

            filters = adaptChannels(channels[layerId])
            currStrides = strides[layerId]
            currDilation = dilations[layerId]

            layerDescription = str(layerId) + "-dw/pw-" + str(filters) + "-" + str(currStrides) + "-" + str(
                currDilation)

            output = lProvider.separable_conv(output, filters, 3, currStrides, dilation=currDilation,
                                              scope=layerDescription)

            print(layerDescription + " : " + str(output.shape))

            if layerId in intermediateSupervision:
                interSeg = lProvider.pointwise_convolution(output, totalJoints, scope=str(layerId) + "-inter-output-1")
                interSeg = tf.sigmoid(interSeg)
                interReg = lProvider.pointwise_convolution(output, totalJoints * 2,
                                                           scope=str(layerId) + "-inter-output-2")
                interOutput = tf.concat([interSeg, interReg], 3, name=str(layerId) + "-inter-output")
                self.intermediateSupervisionOutputs.append(interOutput)

        # last layer ===
        activationFunc = None
        filters = adaptChannels(channels[len(channels) - 1])
        currStrides = strides[len(channels) - 1]
        currDilation = dilations[len(channels) - 1]
        layerDescription = str(len(channels) - 1) + "-dw/pw-" + str(filters) + "-" + str(currStrides) + "-" + str(
            currDilation)

        output = lProvider.separable_conv(output, filters, 3, currStrides, dilation=currDilation, activationFunc=None,
                                          scope=layerDescription)

        seg = lProvider.pointwise_convolution(output, totalJoints, scope=str(layerId) + "-output-1")
        seg = tf.sigmoid(seg)
        reg = lProvider.pointwise_convolution(output, totalJoints * 2, scope=str(layerId) + "-output-2")
        self.output = tf.concat([seg, reg], 3, name="Output")

    def getInput(self):
        return self.inputImage

    def getIntermediateOutputs(self):
        return self.intermediateSupervisionOutputs[:]

    def getOutput(self):
        return self.output


def buildPoseNet(checkpointFile=None, is4Train=False):
    #tf.reset_default_graph()

    modelDir = "parameters/pose_2d/tiny/"

    model = PoseNet(is4Train=is4Train)
    
    interStages = model.getIntermediateOutputs()

    output = model.getOutput()
    #print(output.get_shape())

    inputImage = model.getInput()

    outputStages = interStages
    outputStages.append(output)

    print("total stages : " + str(len(outputStages)))

    #trainer = Trainer(inputImage, output, [output], dataTrainProvider, dataValProvider, modelDir, Trainer.posenetLoss)

    #if not isinstance(checkpointFile, type(None)):
        #trainer.restore(checkpointFile)

    return model,output

def export(t, checkpointFile=None, outputFile="/media/hkuit164/Sherry/fastpose/model/train.pb", outputName="Output"):
    if t == 'pb':
        with tf.Session() as sess:
            net, output = buildPoseNet(checkpointFile)
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
    elif t == 'ckpt':
        with tf.Graph().as_default() as graph:
            net, output = buildPoseNet(checkpointFile)
            stats_graph(graph)

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))

def load_pb(pb):
    with tf.gfile.GFile(pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

if __name__ == '__main__':
        #checkpointFile = '/media/hkuit164/Sherry/fastpose/parameters/pose_2d/tiny/checkpoints/model-936700'
        t='pb'
        export(t)
        if t == 'pb':
            pb = "/media/hkuit164/Sherry/fastpose/model/train.pb"
            graph = load_pb(pb)
            stats_graph(graph)
            
