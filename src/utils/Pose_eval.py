from statistics import mean
from opt import opt
import numpy as np

class pose_eval:
    def __init__(self):
        self.distThresh = 0.5
        self.pcks = []
        self.dist_kp = []


    def save_value(self,pose_gt,pose_pred):

        ALL_dist,KP_dist,mask = pose_gt.distance_to(pose_pred)
        match = ALL_dist <= self.distThresh
        pck = 1.0 * np.sum(match, axis=0) / len(ALL_dist)
        self.pcks.append(pck)
        self.dist_kp.append(KP_dist)
        return self.pcks,self.dist_kp


    def cal_eval(self):

        kps_acc = self.cal_kps_acc(self.dist_kp)

        return kps_acc


    def cal_kps_acc(self,kp_acc):
        value = []
        acc = []
        j = 0
        for i in range(len(kp_acc[j])):
            for item in kp_acc:
                value.append([item[i][0]])
            match_KP = np.array(value) <= self.distThresh
            pck_KP = 1.0 * np.sum(match_KP, axis=0) / len(value)
            acc.append(pck_KP)
            j = j+ 1
        return acc
