import math
import tensorflow as tf
import torch
import cv2
import numpy as np
from opt import opt


def Smooth_l1_loss(predictions,labels,scope=tf.GraphKeys.LOSSES):
    with tf.variable_scope(scope):
        diff=tf.abs(labels-predictions)
        less_than_one=tf.cast(tf.less(diff,1.0),tf.float32)   #Bool to float32
        smooth_l1_loss=(less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)#同上图公式
        return tf.reduce_mean(smooth_l1_loss) #取平均值


def wing_loss(landmarks, labels, w=opt.w, epsilon=opt.epsilon_loss):
    """
    Arguments:
        landmarks, labels: float tensors with shape [batch_size, num_landmarks, 2].
        w, epsilon: a float numbers.
    Returns:
        a float tensor with shape [].
    """
    with tf.name_scope('wing_loss'):
        x = landmarks - labels
        c = w * (1.0 - math.log(1.0 + w / epsilon))
        absolute_x = tf.abs(x)
        losses = tf.where(
            tf.greater(w, absolute_x),
            w * tf.log(1.0 + absolute_x / epsilon),
            absolute_x - c
        )
        loss = tf.reduce_mean(tf.reduce_sum(losses, axis=[1, 2]), axis=0)
        return loss


def AdapWingLoss(pre_hm, gt_hm,w = opt.w ,epsilon = opt.epsilon_loss):
    # pre_hm = pre_hm.to('cpu')
    # gt_hm = gt_hm.to('cpu')
    theta = 0.5
    alpha = 2.1
    e = epsilon
    A = w * (1 / (1 + torch.pow(theta / e, alpha - gt_hm))) * (alpha - gt_hm) * torch.pow(theta / e, alpha - gt_hm - 1) * (1 / e)
    C = (theta * A - w * torch.log(1 + torch.pow(theta / e, alpha - gt_hm)))

    batch_size = gt_hm.size()[0]
    hm_num = gt_hm.size()[1]

    mask = torch.zeros_like(gt_hm)
    # W = 10
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for i in range(batch_size):
        img_list = []
        for j in range(hm_num):
            img_list.append(np.round(gt_hm[i][j].cpu().numpy() * 255))
        img_merge = cv2.merge(img_list)
        img_dilate = cv2.morphologyEx(img_merge, cv2.MORPH_DILATE, kernel)
        img_dilate[img_dilate < 51] = 1  # 0*W+1
        img_dilate[img_dilate >= 51] = 11  # 1*W+1
        img_dilate = np.array(img_dilate, dtype=np.int)
        img_dilate = img_dilate.transpose(2, 0, 1)
        mask[i] = torch.from_numpy(img_dilate)

    diff_hm = torch.abs(gt_hm - pre_hm)
    AWingLoss = A * diff_hm - C
    idx = diff_hm < theta
    AWingLoss[idx] = w * torch.log(1 + torch.pow(diff_hm / e, alpha - gt_hm))[idx]

    AWingLoss *= mask
    sum_loss = torch.sum(AWingLoss)
    all_pixel = torch.sum(mask)
    mean_loss = sum_loss / all_pixel

    return mean_loss



# @staticmethod
# def l2Loss(gt, pred, lossName, batchSize):
#     return tf.nn.l2_loss(pred - gt, name=lossName)

