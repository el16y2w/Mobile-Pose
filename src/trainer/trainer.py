from src.utils.drawer import Drawer
from src.utils.pose import Pose2D
from src.utils.pose import PoseConfig
from src.utils.bbox import BBox
from src.utils.interface import Pose2DInterface
from statistics import mean
import numpy as np
import os
import time
import tensorflow as tf
from opt import opt
from src.utils.loss import wing_loss
from src.utils.loss import adaptivewingLoss
from src.utils.loss import smooth_l1_loss
from Config import config_cmd as config

exp_dir = os.path.join("Result/{}/{}".format(opt.modeloutputFile, opt.Model_folder_name))

class Trainer:
    SAVE_EVERY = opt.SAVE_EVERY
    TEST_EVERY = opt.TEST_EVERY
    VIZ_EVERY = opt.VIZ_EVERY
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
            self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], opt.totaljoints * 3),
                                           name='heatmapGT')
        else:
            self.heatmapGT = tf.placeholder(tf.float32, shape=(None, output.shape[1], output.shape[2], opt.totaljoints),
                                         name='heatmapGT')

        self.globalStep = tf.Variable(0, trainable=False)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.trainLoss = []
        self.updater = []
        self.sess = tf.Session(config=config) if isinstance(sess, type(None)) else sess
        self.learningRate = tf.placeholder(tf.float32, [], name='learningRate')
        self.lr = tf.train.exponential_decay(self.learningRate, global_step=self.globalStep,
                                   decay_steps=opt.decay_steps, decay_rate=opt.decay_rate, staircase=True)
        if opt.optimizer == "Adam":
            self.opt = tf.train.AdamOptimizer(self.lr, epsilon=opt.epsilon)
        elif opt.optimizer == "Momentum": #use_locking: 为True时锁定更新
            self.opt = tf.train.MomentumOptimizer(self.lr, momentum = opt.momentum, use_locking=False, name='Momentum', use_nesterov=False)
        elif opt.optimizer == "Gradient":
            self.opt = tf.train.GradientDescentOptimizer(self.lr,
                                                       use_locking=False, name='GrandientDescent')
        else:
            raise ValueError("Your optimizer name is wrong")

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
            self.savePath = os.path.join(exp_dir, opt.backbone + "checkpoints" + self.time)

            if datatpye[i] == 'yoga':
                tf.summary.scalar("trainLoss", self.trainLoss[i])

                self.fileWriter = tf.summary.FileWriter(os.path.join(exp_dir, opt.backbone+ "logs_yoga"+ self.time), self.sess.graph)
            if datatpye[i] == 'coco':
                #self.savePath = os.path.join(modelDir, "checkpoints_coco" + self.time)
                tf.summary.scalar("trainLoss", self.trainLoss[i])

                self.fileWriter = tf.summary.FileWriter(
                    os.path.join(exp_dir, opt.backbone + "logs_coco" + self.time), self.sess.graph)
        self.fileWriter = tf.summary.FileWriter(
            os.path.join(exp_dir, opt.backbone + "logs_all" + self.time), self.sess.graph)
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
        predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]
        totaljoints = opt.totaljoints

        if opt.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif opt.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
        else:
            raise ValueError("Your optimizer name is wrong")

        for recordId in range(batchSize):
            for jointId in range(totaljoints):
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
        predHeat, gtHeat = pred[:, :, :, :opt.totaljoints], gt[:, :, :, :opt.totaljoints]
        predOffX, gtOffX = pred[:, :, :, opt.totaljoints:(2 * opt.totaljoints)], gt[:, :, :, opt.totaljoints:(
                                                                                                     2 * opt.totaljoints)]
        predOffY, gtOffY = pred[:, :, :, (2 * opt.totaljoints):], gt[:, :, :, (2 * opt.totaljoints):]
        totaljoints = opt.totaljoints
        if opt.hm_lossselect == 'l2':
            heatmapLoss = tf.nn.l2_loss(predHeat - gtHeat, name=lossName + "_heatmapLoss")
        elif opt.hm_lossselect == 'wing':
            heatmapLoss = wing_loss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'adaptivewing':
            heatmapLoss = adaptivewingLoss(predHeat, gtHeat)
        elif opt.hm_lossselect == 'smooth_l1':
            heatmapLoss = smooth_l1_loss(None, predHeat, gtHeat)
        else:
            raise ValueError("Your optimizer name is wrong")
        offsetGT, offsetPred = [], []

        offsetLoss = 0

        for recordId in range(batchSize):
            for jointId in range(totaljoints):
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

        totalJoints = opt.totaljoints

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

    def cal_kps_acc(self,kps_acc):
        value = []
        acc = []
        j = 0
        for i in range(len(kps_acc[j])):
            for item in kps_acc:
                value.append(item[i])
            acc.append(mean(value))
            j = j+1
        return acc

    def start(self, fromStep, totalSteps, lr, modeltype, date):
        total_iterations = 0
        best_validation_accuracy = 1  # 当前最佳验证集准确率
        last_improvement = 0  # 上一次有所改进的轮次
        cur_lr = lr
        j_num = 0
        require_improvement = opt.require_improvement  # 如果在1000轮内没有改进，停止迭代

        result = open(os.path.join(exp_dir, opt.backbone + date + "_result.csv"), "w")
        result.write(
            "model_name,isTrain,checkpoints_file,offset,traindata,inputsize,outputsize,optimizer,opt_epilon,momentum,heatmaploss, epsilon_loss, "
            "loss_w,Gauthreshold, GauSigma, datasetnumber,Dataset,Totaljoints,epochs, learning_type,decayrate,learning-rate, require_improvement,j_min,j_max,test_epoch,"
            "training_time,train_loss, val_acc,head, "
            "lShoulder, rShoulder, lElbow, rElbow, lWrist, rWrist, lHip,rHip, lKnee, rKnee, lAnkle, rAnkle\n")
        result.close()
        if os.path.exists("Result/Yogapose" + '/' + "training_result.csv"):
            pass
        else:
            result_all = open(os.path.join(opt.train_all_result, "training_result.csv"), "w")
            result_all.write(
                "Index,Backbone,isTrain,checkpoints_file,offset,traindata,inputsize,outputsize,optimizer,opt_epilon,momentum,heatmaploss, epsilon_loss, "
                "loss_w,Gauthreshold, GauSigma, datasetnumber,Batch,Dataset,Totaljoints,Total_epochs,Stop_epoch, learning_type,learning-rate,decay_rate,require_improvement, "
                "j_min,j_max,test_epoch,training_time,train_loss, best_validation_accuracy, Dataset\n")
            result_all.close()

        for i in range(fromStep, fromStep + totalSteps + 1):
            start_time = time.time()
            total_iterations += 1

            result = open(os.path.join(exp_dir, opt.backbone + date + "_result.csv"), "a+")
            result_all = open(os.path.join(opt.train_all_result, "training_result.csv"), "a+")
            for j in range(config.datanumber):
                inputs, heatmaps = self.dataTrainProvider[j].drawn()
                res = self.sess.run([self.trainLoss[j], self.updater[j], self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps,
                                               self.learningRate: lr})
                self.fileWriter.add_summary(res[2], i)
                print(str(i) + " -- TRAIN" + str(j) + " : " + str(res[0]))

            training_time = time.time() - start_time
            train_loss = str(res[0])

            if opt.Early_stopping:
            # 每100轮迭代输出状态
                if (total_iterations % opt.test_epoch == 0) or (i == totalSteps - 1):

                    if Val_acc < best_validation_accuracy:  # 如果当前验证集准确率大于之前的最好准确率
                        best_validation_accuracy = Val_acc  # 更新最好准确率
                        last_improvement = total_iterations  # 更新上一次提升的迭代轮次
                        j_num = 0
                        checkpoint_path = os.path.join(self.savePath, 'model')
                        self.saver.save(self.sess, checkpoint_path, global_step=i)

                    else:
                        j_num += 1
                        if j_num <opt.j_min : pass
                        else:
                            if opt.lr_type == "exponential_decay":
                                lr = cur_lr * opt.decay_rate
                            elif opt.lr_type == "polynomial_decay":
                                global_step = min(i, 1000)
                                lr = (cur_lr - 0.00001) * (1 - global_step / 1000) + 0.00001
                                '''
                                global_step = min(global_step, decay_steps)
                                decayed_learning_rate = (learning_rate - end_learning_rate) *
                                                        (1 - global_step / decay_steps) ^ (power) +
                                                         end_learning_rate
                                '''
                            # elif opt.lr_type == "natural_exp_decay":
                            #     lr = cur_lr * tf.math.exp(-opt.decay_rate * i)
                            elif opt.lr_type == "inverse_time_decay":
                                lr = cur_lr / (1 + opt.decay_rate * i / 1000)
                            else:
                                raise ValueError("Your lr_type name is wrong")
                            cur_lr = lr
                # 如果在require_improvement轮次内未有提升
                if total_iterations - last_improvement > require_improvement or j_num > opt.j_max:
                    result_all.write(
                        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                        format(opt.Model_folder_name, modeltype, opt.isTrain, opt.checkpoints_file, opt.offset,
                               str(config.dataformat), self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon,
                               opt.momentum,
                               opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres, opt.gaussian_sigma,
                               config.datanumber, opt.batch, opt.dataset,opt.totaljoints,opt.epoch, total_iterations, opt.lr_type, lr, opt.decay_rate,
                               opt.require_improvement,
                               opt.j_min, opt.j_max, opt.test_epoch, training_time, train_loss, best_validation_accuracy,
                               config.dataset_comment))
                    print("Stop optimization")
                    break  # 跳出循环

            else:
                if i % Trainer.SAVE_EVERY == 0:
                    checkpoint_path = os.path.join(self.savePath, 'model')
                    self.saver.save(self.sess, checkpoint_path, global_step=i)


            if i % Trainer.TEST_EVERY == 0:
                inputs, heatmaps = self.dataValProvider[0].drawn()

                res = self.sess.run([self.output, self.summaryMerge],
                                    feed_dict={self.inputImage: inputs, self.heatmapGT: heatmaps, self.learningRate: 0})

                fullscreen_bbox = BBox(0, 1, 0, 1)

                distances = []
                distances_kps_acc = []
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

                    distances_all, distances_kps = pose_gt.distance_to(pose_pred)
                    distances.append(distances_all)
                    distances_kps_acc.append(distances_kps)

                kps_acc = self.cal_kps_acc(distances_kps_acc)
                summary = tf.Summary(value=[tf.Summary.Value(tag="testset_accuracy", simple_value=mean(distances))])
                Val_acc = mean(distances)
                print("Model_Folder:{}|--Epoch:{}|--isTrain:{}|--Earlystop:{}|--Train Loss:{}|--Val Acc:{}|--lr:{}".format(
                    str(opt.Model_folder_name),str(i),str(opt.isTrain),str(opt.Early_stopping),train_loss, str(Val_acc)[:4] ,str(cur_lr)))

                result.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                             format(modeltype,opt.isTrain,opt.checkpoints_file,opt.offset,str(config.dataformat), self.inputSize[0],
                                    config.outputSize[0],opt.optimizer,opt.epsilon,opt.momentum,opt.hm_lossselect,opt.epsilon_loss,
                                    opt.w,opt.gaussian_thres, opt.gaussian_sigma,config.datanumber, opt.dataset,opt.totaljoints,i, opt.lr_type,opt.decay_rate,lr,
                                    opt.require_improvement,opt.j_min,opt.j_max,opt.test_epoch,training_time,train_loss, Val_acc,kps_acc[0],
                                    kps_acc[1],kps_acc[2],kps_acc[3],kps_acc[4],kps_acc[5],kps_acc[6],kps_acc[7],kps_acc[8],kps_acc[9],
                                    kps_acc[10],kps_acc[11],kps_acc[12]))
                self.fileWriter.add_summary(summary, i)
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
        result_all.write(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                format(opt.Model_folder_name, modeltype, opt.isTrain, opt.checkpoints_file, opt.offset,
                       str(config.dataformat), self.inputSize[0], config.outputSize[0], opt.optimizer, opt.epsilon,
                       opt.momentum,
                       opt.hm_lossselect, opt.epsilon_loss, opt.w, opt.gaussian_thres, opt.gaussian_sigma,
                       config.datanumber, opt.batch, opt.dataset, opt.totaljoints, opt.epoch, total_iterations,
                       opt.lr_type, lr, opt.decay_rate,
                       opt.require_improvement,
                       opt.j_min, opt.j_max, opt.test_epoch, training_time, train_loss, best_validation_accuracy,
                       config.dataset_comment))
