from src.models.Layerprovider1 import LayerProvider
import tensorflow as tf
from opt import opt
from Config import config_cmd as config




class finallayerforoffsetoption(object):
    def __init__(self, offset = opt.offset,pixel = config.pixelshuffle,conv = config.convb_13):
        self.lProvider = LayerProvider(opt.isTrainpre)
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
            output = self.lProvider.pointwise_convolution(output, totalJoints, scope="Output")
            self.output = tf.identity(output, "Output")
        return self.output

    def fornetworks_DUC(self,output,totalJoints):
        # output = tf.nn.depth_to_space(output, 2) #pixel shuffle
        # output = self.lProvider.convb(output,3,3,160,1,"psconv1",relu=True)
        # output = tf.nn.depth_to_space(output, 2)
        # output = self.lProvider.convb(output, 3, 3, 52, 1, "psconv2", relu=True)
        # output = tf.nn.depth_to_space(output, 2)
        output = tf.nn.depth_to_space(output, 2) # pixel shuffle
        for i in range(len(self.pixel)):
            if opt.totaljoints == 13:
                output = self.lProvider.convb(output, self.conv[i][0], self.conv[i][1], self.conv[i][2], self.conv[i][3],
                                          self.conv[i][4], relu=True)
            elif opt.totaljoints == 16:
                output = self.lProvider.convb(output, config.convb_16[i][0], config.convb_16[i][1], config.convb_16[i][2], config.convb_16[i][3],
                                              config.convb_16[i][4], relu=True)
            output = tf.nn.depth_to_space(output, self.pixel[i])
        if self.offset == False:
            output = tf.identity(output, "Output")
            # return a tensor with the same shape and contents as input.
        else:
            seg = self.lProvider.pointwise_convolution(output, totalJoints, scope="output-1")
            seg = tf.sigmoid(seg)
            reg = self.lProvider.pointwise_convolution(output, totalJoints * 2, scope="output-2")
            output = tf.concat([seg, reg], 3, name="Output")
            #output = tf.identity(output, "Output")

        return output
