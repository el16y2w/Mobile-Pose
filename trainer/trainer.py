from utils.drawer import Drawer
from utils.pose import Pose2D
from utils.pose import PoseConfig
from utils.bbox import BBox
from utils.interface import Pose2DInterface
from statistics import mean
import numpy as np
import os
import tensorflow as tf
import config
from loss import wing_loss
from loss import adaptivewingLoss
from loss import smooth_l1_loss



class Trainer:
    SAVE_EVERY = config.SAVE_EVERY
    TEST_EVERY = config.TEST_EVERY
    VIZ_EVERY = config.VIZ_EVERY
    num = config.datanumber

    def __init__(self, inputImage, output, outputStages, dataTrainProvider, dataValProvider, modelDir, lossFunc,
                 inputSize,datatpye,offsetset , time,sess=None):

        self.inputSize = inputSize
        self.dataTrainProvider, self.dataValProvider = dataTrainProvider, dataValProvider
        self.inputImage = inputImage
        self.output = output
        self.offsetornot = offsetset
        self.dataformat = datatpye
        self.time = time

        if self.offsetornot == True:
            self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], 13 * 3),
                                           name='heatmapGT')
        else:
            self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], 13),
                                         name='heatmapGT')

        #self.globalStep = tf.Variable(0, trainable=True)
        self.globalStep = tf.Variable(0, trainable=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.trainLoss = []
        self.updater = []
        self.sess = tf.Session(config=config) if isinstance(sess, type(None)) else sess
        self.learningRate = tf.placeholder(tf.float32, [], name='learningRate')
        self.lr = tf.train.exponential_decay(self.learningRate, global_step=self.globalStep,
                                   decay_steps=10000, decay_rate=0.95, staircase=True)
        self.opt = tf.train.AdamOptimizer(self.lr, epsilon=1e-8)
        for i in range(len(self.dataTrainProvider)):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            Loss = self._buildLoss(self.heatmapGT, outputStages, dataTrainProvider[i].getBatchSize(), lossFunc,
                                         "trainLoss")
            self.trainLoss.append(Loss)
            with tf.control_dependencies(update_ops):
                # self.train_op = tf.group(apply_gradient_op, variables_averages_op)
                upd = self.opt.minimize(self.trainLoss[i], self.globalStep)
            self.updater.append(upd)



        tf.summary.scalar("learningRate", self.lr)

        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=10)
        for i in range(len(self.dataformat)):
            self.savePath = os.path.join(modelDir, "checkpoints" + self.time)

            if datatpye[i] == 'yoga':
                tf.summary.scalar("trainLoss", self.trainLoss[i])

                self.fileWriter = tf.summary.FileWriter(os.path.join(modelDir, "logs_yoga"+ self.time), self.sess.graph)
            if datatpye[i] == 'coco':
                #self.savePath = os.path.join(modelDir, "checkpoints_coco" + self.time)
                tf.summary.scalar("trainLoss", self.trainLoss[i])

                self.fileWriter = tf.summary.FileWriter(
                    os.path.join(modelDir, "logs_coco" + self.time), self.sess.graph)
        self.fileWriter = tf.summary.FileWriter(
            os.path.join(modelDir, "logs_all" + self.time), self.sess.graph)
        self.summaryMerge = tf.summary.merge_all()

    def restore(self, checkpointPath):
        tf.train.Saver().restore(self.sess, checkpointPath)

    def setLearningRate(self, lr):
        self.sess.run(self.learningRate, feed_dict={self.learningRate: lr})

    def _buildLoss(self, heatmapGT, outputStages, batchSize, lossFunc, lossName):
        losses = []
        for idx, stage_out in enumerate(outputStages):
            loss = lossFunc(heatmapGT, stage_out, lossName + '_' + str(idx), batchSize)
            tf.summary.scalar(lossName + "_stage_" + str(idx), (tf.reduce_sum(loss) / batchSize))
            losses.append(loss)

        return (tf.reduce_sum(losses) / len(outputStages)) / batchSize

    # @staticmethod
    # def l2Loss(gt, pred, lossName, batchSize):
    #     return tf.nn.l2_loss(pred - gt, name=lossName)

    def posenetLoss_nooffset(gt, pred, lossName, batchSize):
        predHeat, gtHeat = pred[:, :, :, :len(PoseConfig.NAMES)], gt[:, :, :, :len(PoseConfig.NAMES)]
        heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        if config.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif config.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif config.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif config.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
        for recordId in range(batchSize):
            for jointId in range(len(PoseConfig.NAMES)):
                print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
                # ================================> decode <x,y> from gt heatmap
                inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
                pixId = tf.argmax(inlinedPix)
                x = tf.floormod(pixId, gtHeat.shape[2])
                y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)


        print("start building huber loss")
        print("huber loss built")
        tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
        return heatmapLoss

    def posenetLoss(gt, pred, lossName, batchSize):
        predHeat, gtHeat = pred[:, :, :, :len(PoseConfig.NAMES)], gt[:, :, :, :len(PoseConfig.NAMES)]
        if config.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif config.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif config.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif config.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)

        predOffX, gtOffX = pred[:, :, :, len(PoseConfig.NAMES):(2 * len(PoseConfig.NAMES))], gt[:, :, :,
                                                                                            len(PoseConfig.NAMES):(
                                                                                                   2 * len(
                                                                                             PoseConfig.NAMES))]
        predOffY, gtOffY = pred[:, :, :, (2 * len(PoseConfig.NAMES)):], gt[:, :, :, (2 * len(PoseConfig.NAMES)):]
        offsetGT, offsetPred = [], []
        offsetLoss = 0

        for recordId in range(batchSize):
            for jointId in range(len(PoseConfig.NAMES)):
                print(str(recordId) + "/" + str(batchSize) + " : " + str(jointId))
                # ================================> decode <x,y> from gt heatmap

                inlinedPix = tf.reshape(gtHeat[recordId, :, :, jointId], [-1])
                pixId = tf.argmax(inlinedPix)


                x = tf.floormod(pixId, gtHeat.shape[2])
                y = tf.cast(tf.divide(pixId, gtHeat.shape[2]), tf.int64)

                # ==============================> add offset loss over the gt pix

                offsetGT.append(gtOffX[recordId, y, x, jointId])
                offsetPred.append(predOffX[recordId, y, x, jointId])
                offsetGT.append(gtOffY[recordId, y, x, jointId])
                offsetPred.append(predOffY[recordId, y, x, jointId])

        print("start building huber loss")
        offsetGT = tf.stack(offsetGT, 0)
        offsetPred = tf.stack(offsetPred, 0)
        offsetLoss = 5 * tf.losses.huber_loss(offsetGT, offsetPred)
        print("huber loss built")

        tf.summary.scalar(lossName + "_heatmapLoss", heatmapLoss)
        tf.summary.scalar(lossName + "_offsetLoss", offsetLoss)
        return (heatmapLoss + offsetLoss)

    def _buildUpdater(self, loss, globalStep, lr):

        tf.summary.scalar("learningRate", lr)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            updater = tf.train.AdamOptimizer(lr, epsilon=1e-8).minimize(loss,globalStep)

        return updater


    def _toPose(self, output):

        totalJoints = len(PoseConfig.NAMES)

        if self.offsetornot == True:
            heatmap = output[:, :, :totalJoints]
            xOff = output[:, :, totalJoints:(totalJoints * 2)]
            yOff = output[:, :, (totalJoints * 2):]
        else:
            heatmap = output[:, :, :totalJoints]

        joints = np.zeros((totalJoints, 2)) - 1

        for jointId in range(totalJoints):
            inlinedPix = heatmap[:, :, jointId].reshape(-1)
            pixId = np.argmax(inlinedPix)


            outX = pixId % output.shape[1]
            outY = pixId // output.shape[1]
            if self.offsetornot == True:
                x = outX / output.shape[1] * self.inputImage.get_shape().as_list()[2] + xOff[outY, outX, jointId]
                y = outY / output.shape[0] * self.inputImage.get_shape().as_list()[1] + yOff[outY, outX, jointId]
            else:
                x = outX / output.shape[1] * self.inputImage.get_shape().as_list()[2]
                y = outY / output.shape[0] * self.inputImage.get_shape().as_list()[1]

            x = x / self.inputImage.get_shape().as_list()[2]
            y = y / self.inputImage.get_shape().as_list()[1]

            joints[jointId, 0] = x
            joints[jointId, 1] = y

        return Pose2D(joints)

    def _imageFeatureToImage(self, imageFeature):
        return (((imageFeature[:, :, :] + 1) / 2) * 255).astype(np.uint8)

    def _heatmapVisualisation(self, heatmaps):
        return ((heatmaps.sum(2) / heatmaps.sum(2).max()) * 255).astype(np.uint8)

    def average_gradients(self, tower_grads):
        """
        Get gradients of all variables.
        :param tower_grads:
        :return:
        """
        average_grads = []

        # get variable and gradients in differents gpus
        for grad_and_vars in zip(*tower_grads):
            # calculate the average gradient of each gpu
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def start(self, fromStep, totalSteps, lr, modeltype, time):
        result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "w")
        result.write(
            "model_name,inputsize,outputsize, Gauthreshold, GauSigma, datasetnumber,epochs, learning-rate, train_loss, test_acc\n")
        result.close()
        result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "a+")
        for i in range(fromStep, fromStep + totalSteps + 1):
            result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "a+")
            for j in range(config.datanumber):
                inputs, heatmaps = self.dataTrainProvider[j].drawn()
                res = self.sess.run([self.trainLoss[j], self.updater[j], self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps,
                                               self.learningRate: lr})
                self.fileWriter.add_summary(res[2], i)
                print(str(i) + " -- TRAIN" + str(j) + " : " + str(res[0]))
            a = str(res[0])
            result.write("{},{},{},{},{},{},{},{},{}\n".format(modeltype,self.inputSize[0], config.outputSize[0],
                                                            config.threshold,config.sigma, config.datanumber, i, lr, a))
            if i % Trainer.SAVE_EVERY == 0:
                checkpoint_path = os.path.join(self.savePath, 'model')
                self.saver.save(self.sess, checkpoint_path, global_step=i)

            if i % Trainer.TEST_EVERY == 0:
                inputs, heatmaps = self.dataValProvider[0].drawn()

                res = self.sess.run([self.output, self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0})

                fullscreen_bbox = BBox(0, 1, 0, 1)

                distances = []
                for batch_id in range(inputs.shape[0]):
                    pose_gt, _ = Pose2DInterface.our_approach_postprocessing(heatmaps[batch_id, :, :, :],
                                                                             fullscreen_bbox, self.inputSize)
                    pose_pred, _ = Pose2DInterface.our_approach_postprocessing(res[0][batch_id, :, :, :],
                                                                               fullscreen_bbox, self.inputSize)

                    # pose_pred
                    # all labeled gt joints are used in the loss,
                    # if not detected by the prediction joint location (-1,-1) => (0.5,0.5)
                    tmp = pose_pred.get_joints()
                    tmp[~pose_pred.get_active_joints(), :] = 0.5
                    pose_pred = Pose2D(tmp)

                    distances.append(pose_gt.distance_to(pose_pred))

                summary = tf.Summary(value=[tf.Summary.Value(tag="testset_accuracy", simple_value=mean(distances))])

                self.fileWriter.add_summary(summary, i)

            if i % Trainer.VIZ_EVERY == 0:
                inputs, heatmaps = self.dataValProvider[0].drawn()
                res = self.sess.run([self.output, self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0})

                currHeatmaps = res[0][0, :, :, :]
                currImage = self._imageFeatureToImage(inputs[0, :, :, :])
                currHeatmapViz = self._heatmapVisualisation(currHeatmaps)
                currHeatmapViz = currHeatmapViz.reshape((1, currHeatmapViz.shape[0], currHeatmapViz.shape[0], 1))
                currPose = self._toPose(currHeatmaps)
                skeletonViz = np.expand_dims(Drawer.draw_2d_pose(currImage, currPose), 0)

                tmp = tf.summary.image("skeleton_" + str(i), skeletonViz).eval(session=self.sess)
                self.fileWriter.add_summary(tmp, i)
                tmp = tf.summary.image("heatmap_predicted_" + str(i), currHeatmapViz).eval(session=self.sess)
                self.fileWriter.add_summary(tmp, i)