from src.models.Layerprovider import LayerProvider
import tensorflow as tf
import config



class finallayerforoffsetoption(object):
    def __init__(self, offset = config.offset, pixel = config.pixelshuffle,conv = config.convb):
        self.lProvider = LayerProvider(config.isTrain)
        self.offset = offset
        self.pixel = pixel
        self.conv = conv

    def fornetworks(self,output, totalJoints):
        if self.offset == True:
            seg = self.lProvider.pointwise_convolution(output, totalJoints, scope= "output-1")
            seg = tf.sigmoid(seg)
            reg = self.lProvider.pointwise_convolution(output, totalJoints * 2, scope="output-2")
            self.output = tf.concat([seg, reg], 3, name="Output")
        else:
            self.output = self.lProvider.pointwise_convolution(output, totalJoints, scope="Output")
        return self.output

    def fornetworks_DUC(self,output,totalJoints):
        output = tf.nn.depth_to_space(output, 2) # pixel shuffle
        for i in range(len(self.pixel)):
            output = self.lProvider.convb(output, self.conv[i][0], self.conv[i][1], self.conv[i][2], self.conv[i][3],
                                          self.conv[i][4], relu=True)
            output = tf.nn.depth_to_space(output, self.pixel[i])
        if self.offset == False:
            output = tf.identity(output, "Output")
            # return a tensor with the same shape and contents as input.
        else:
            seg = self.lProvider.pointwise_convolution(output, totalJoints, scope="output-1")
            seg = tf.sigmoid(seg)
            reg = self.lProvider.pointwise_convolution(output, totalJoints * 2, scope="output-2")
            output = tf.concat([seg, reg], 3, name="Output")

        return output
