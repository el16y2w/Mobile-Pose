import tensorflow as tf
import config_cmd as config
import os
import time
from modelbuild.PoseBuild import train_pose
from Pose_eval import poseevalpckh
from opt import opt

time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.makedirs("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name), exist_ok=True)
exp_dir = os.path.join("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name))

class run:
    def __init__(self):
        self.tp = train_pose()
        self.train_anno = config.train_annotFile
        self.train_img = config.train_imageDir
        self.test_anno = config.test_annotFile
        self.test_img = config.test_imageDir
        self.inputSize = config.inputSize
        self.outputSize = config.outputSize
        self.batch = opt.batch
        self.trainornot = opt.isTrain
        self.checkpoint = opt.checkpoints_file
        self.model = opt.backbone
        self.dataformat = config.dataformat
        self.pixel = config.pixelshuffle
        self.convb = config.convb
        self.poseeval = poseevalpckh()
        self.activate_function = opt.activate_function
        self.hm_lossselect = opt.hm_lossselect
        self.grayimage = opt.grayimage


    def train(self):
        txt_file = open(os.path.join(exp_dir,  time_str+"_"+opt.backbone+".txt"), "w+")
        if self.model == "mobilenetv3":
            txt_file.write(
                "pose_{}.pb: model :{}, v3_version:{}, widthscale:{}, {} epochs, {} lr, {} input size,"
                " {} output size, isTrain={}\n, offset: {}, Gauthreshold: {}, GauSigma: {},Used dataset: {},"
                " Duc:{}and{}, activate function: {}, loss : {}, gray or not: {}".
                format(time_str, self.model, opt.v3_version, opt.v3_width_scale, opt.epoch, opt.lr,
                       config.inputSize,self.outputSize, self.trainornot, opt.offset,opt.gaussian_thres,
                       opt.gaussian_sigma,config.dataset, self.pixel,self.convb,self.activate_function,self.hm_lossselect,
                       self.grayimage))
        else:
            txt_file.write("pose_{}.pb: model type:{}, {} epochs, {} lr, {} input size, {} outputsize, "
                           "isTrain={}\n, offset: {}, , Gauthreshold: {}, GauSigma: {}, Used dataset: {}, "
                           "Duc:{}and{}, activate function: {}, loss : {}, gray or not: {}".
                           format(time_str,self.model, opt.epoch, opt.lr, config.inputSize,self.outputSize,
                                  self.trainornot, opt.offset, opt.gaussian_thres,
                                  opt.gaussian_sigma, config.dataset,
                                  self.pixel,self.convb,self.activate_function,self.hm_lossselect,
                       self.grayimage))
        txt_file.close()
        self.tp.train_fastpose( self.trainornot, self.checkpoint, self.model,
                        opt.epoch, opt.lr, time_str, config.inputSize,self.outputSize,
                                config.inputshape)



if __name__ == '__main__':
    R = run()
    R.train()
