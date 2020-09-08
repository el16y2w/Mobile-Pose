from statistics import mean
from src.utils.pose import Pose2D

class pose_eval:
    def __init__(self):
        self.distances = []
        self.distances_kps_acc = []

    def cal_kps_acc(self ,kps_acc):
        value = []
        acc = []
        j = 0
        for i in range(len(kps_acc[j])):
            for item in kps_acc:
                value.append(item[i])
            acc.append(mean(value))
            j = j+ 1
        return acc

    def save_value(self,pose_gt,pose_pred):
        distances_all, distances_kps, mask = pose_gt.distance_to(pose_pred)
        self.distances.append(distances_all)
        self.distances_kps_acc.append(distances_kps)

        return self.distances


    def cal_eval(self):

        kps_acc = self.cal_kps_acc(self.distances_kps_acc)

        return kps_acc
