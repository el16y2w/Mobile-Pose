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
from datetime import datetime
import config


class Trainer:
    SAVE_EVERY = config.SAVE_EVERY
    TEST_EVERY = config.TEST_EVERY
    VIZ_EVERY = config.VIZ_EVERY
    num = config.datanumber

    def __init__(self, inputImage, output, outputStages, dataTrainProvider, dataValProvider, modelDir, lossFunc, time,
                 inputSize=config.inputSize[0],datatpye=config.dataformat,sess=None):

        self.inputSize = inputSize
        self.dataTrainProvider, self.dataValProvider = dataTrainProvider, dataValProvider
        self.inputImage = inputImage
        self.output = output

        self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], 13 * 3),
                                        name='heatmapGT')
        self.globalStep = tf.Variable(0, trainable=True)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config) if isinstance(sess, type(None)) else sess

        self.learningRate = tf.placeholder(tf.float32, [], name='learningRate')
        self.trainLoss = []
        self.updater = []
        for i in range(len(self.dataTrainProvider)):
            Loss = self._buildLoss(self.heatmapGT, outputStages, dataTrainProvider[i].getBatchSize(), lossFunc,
                                         "trainLoss")
            self.trainLoss.append(Loss)
            upda = self._buildUpdater(self.trainLoss[i], self.globalStep, self.learningRate)
            self.updater.append(upda)


        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(max_to_keep=10)
        if datatpye == 'yoga':

            self.savePath = os.path.join(modelDir, "checkpoints_yoga"+ time)

            for i in range(len(self.trainLoss)):
                tf.summary.scalar("trainLoss", self.trainLoss[i])

            self.fileWriter = tf.summary.FileWriter(os.path.join(modelDir, "logs_yoga"+ time), self.sess.graph)
        if datatpye == 'coco':
            self.savePath = os.path.join(modelDir, "checkpoints_coco" + time)

            for i in range(len(self.trainLoss)):
                tf.summary.scalar("trainLoss", self.trainLoss[i])

            self.fileWriter = tf.summary.FileWriter(
                os.path.join(modelDir, "logs_coco" + time), self.sess.graph)

        self.summaryMerge = tf.summary.merge_all()

    def restore(self, checkpointPath):
        tf.train.Saver().restore(self.sess, checkpointPath)

    # def setLearningRate(self, lr):
    #     self.sess.run(self.learningRate, feed_dict={self.learningRate: lr})

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

    @staticmethod
    def posenetLoss(gt, pred, lossName, batchSize):

        predHeat, gtHeat = pred[:, :, :, :len(PoseConfig.NAMES)], gt[:, :, :, :len(PoseConfig.NAMES)]
        predOffX, gtOffX = pred[:, :, :, len(PoseConfig.NAMES):(2 * len(PoseConfig.NAMES))], gt[:, :, :,
                                                                                             len(PoseConfig.NAMES):(
                                                                                                         2 * len(
                                                                                                     PoseConfig.NAMES))]
        predOffY, gtOffY = pred[:, :, :, (2 * len(PoseConfig.NAMES)):], gt[:, :, :, (2 * len(PoseConfig.NAMES)):]

        heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")

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
            updater = tf.train.AdamOptimizer(lr, epsilon=1e-8).minimize(loss)

        return updater

    def _toPose(self, output):

        totalJoints = len(PoseConfig.NAMES)

        heatmap = output[:, :, :totalJoints]
        xOff = output[:, :, totalJoints:(totalJoints * 2)]
        yOff = output[:, :, (totalJoints * 2):]

        joints = np.zeros((totalJoints, 2)) - 1

        for jointId in range(totalJoints):
            inlinedPix = heatmap[:, :, jointId].reshape(-1)
            pixId = np.argmax(inlinedPix)


            outX = pixId % output.shape[1]
            outY = pixId // output.shape[1]

            x = outX / output.shape[1] * self.inputImage.get_shape().as_list()[2] + xOff[outY, outX, jointId]
            y = outY / output.shape[0] * self.inputImage.get_shape().as_list()[1] + yOff[outY, outX, jointId]

            x = x / self.inputImage.get_shape().as_list()[2]
            y = y / self.inputImage.get_shape().as_list()[1]

            joints[jointId, 0] = x
            joints[jointId, 1] = y

        return Pose2D(joints)

    def _imageFeatureToImage(self, imageFeature):
        return (((imageFeature[:, :, :] + 1) / 2) * 255).astype(np.uint8)

    def _heatmapVisualisation(self, heatmaps):
        return ((heatmaps.sum(2) / heatmaps.sum(2).max()) * 255).astype(np.uint8)

    def start(self, fromStep, totalSteps, lr, modeltype,time):
        result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "w")
        result.write(
            "model_name, epochs, learning-rate, train_loss, test_acc\n")
        result.close()
        result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "a+")
        for i in range(fromStep, fromStep + totalSteps + 1):
            result = open(os.path.join(config.modeloutputFile, time + "training_result.csv"), "a+")
            for i in range(config.datanumber):
                inputs, heatmaps = self.dataTrainProvider[i].drawn()
                res = self.sess.run([self.trainLoss[i], self.updater[i], self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: lr})
                self.fileWriter.add_summary(res[2], i)
                print(str(i) + " -- TRAIN"+str(i)+" : " + str(res[0]))

            a = str(res[0])

            result.write("{},{},{},{}\n".format(modeltype, i, lr, a))

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
                result.write("{},{},{},{},{}\n".format(modeltype, i, lr,  a, mean(distances)))

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
        result.close()


