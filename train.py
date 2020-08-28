import tensorflow as tf
from Config import config
import os
import time
from src.modelbuild.PoseBuild import train_pose
from Pose_eval import poseevalpckh

time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

class run:
    def __init__(self):
        self.tp = train_pose()
        self.train_anno = config.train_annotFile
        self.train_img = config.train_imageDir
        self.test_anno = config.test_annotFile
        self.test_img = config.test_imageDir
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.batch = config.batch
        self.trainornot = config.isTrain
        self.checkpoint = config.checkpoints_file
        self.model = config.model
        self.dataformat = config.dataformat
        self.outputmodel = config.modelname
        self.pixel = config.pixelshuffle
        self.convb = config.convb
        self.poseeval = poseevalpckh()
        self.activate_function = config.activate_function
        self.hm_lossselect = config.hm_lossselect
        self.grayimage = config.grayimage


    def train(self):
        i = 0
        for epoch in config.epoch:
            for lrs in config.lr:
                for i in range(len(self.model)):
                    txt_file = open(os.path.join(config.modeloutputFile, "res.txt" + time_str), "w+")
                    if self.model[i] == "mobilenetv3":
                        txt_file.write(
                            "pose_{}.pb: model :{}, v3_version:{}, widthscale:{}, {} epochs, {} lr, {} input size,"
                            " {} output size, isTrain={}\n, offset: {}, Gauthreshold: {}, GauSigma: {},Used dataset: {},"
                            " Duc:{}and{}, activate function: {}, loss : {}, gray or not: {}".
                            format(time_str, self.model[i], config.v3_version, config.v3_width_scale, epoch, lrs,
                                   config.inputSize[i],self.outputSize[i], config.isTrain, config.offset,config.threshold,
                                   config.sigma,config.dataset, self.pixel,self.convb,self.activate_function,self.hm_lossselect,
                                   self.grayimage))
                    else:
                        txt_file.write("pose_{}.pb: model type:{}, {} epochs, {} lr, {} input size, {} outputsize, "
                                       "isTrain={}\n, offset: {}, , Gauthreshold: {}, GauSigma: {}, Used dataset: {}, "
                                       "Duc:{}and{}, activate function: {}, loss : {}, gray or not: {}".
                                       format(time_str,self.model[i], epoch, lrs, config.inputSize[i],self.outputSize[i],
                                              config.isTrain, config.offset,config.threshold,config.sigma, config.dataset,
                                              self.pixel,self.convb,self.activate_function,self.hm_lossselect,
                                   self.grayimage))
                    txt_file.close()
                    self.tp.train_fastpose( self.trainornot, self.checkpoint, self.model[i],
                                   self.dataformat, epoch, lrs, time_str, self.outputmodel,config.inputSize[i],self.outputSize[i],
                                            config.inputshape[i])






if __name__ == '__main__':
    R = run()
    R.train()
