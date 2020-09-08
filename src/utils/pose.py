from src.utils.bbox import BBox
import numpy as np
from opt import opt

"""
Pose configuration
"""
class PoseConfig():
    # The joint order defined by the system
    if opt.totaljoints == 13:
        NAMES = ["head", "leftShoulder", "rightShoulder", "leftElbow", "rightElbow", "leftWrist", "rightWrist", "leftHip",
                 "rightHip", "leftKnee", "rightKnee", "leftAnkle", "rightAnkle"]

        HEAD, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 1, 2, 3, 4, 5, 6
        L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 7, 8, 9, 10, 11, 12
        # The available bones
        BONES = [(1, 3), (3, 5), (2, 4), (4, 6), (7, 9), (9, 11), (8, 10), (10, 12), (1,2), (1,7), (2,8)]

    elif opt.dataset == "MPII":
    #for mpii
        MPIINAMES =  ["r_ankle", "r_knee", "r_hip","l_hip", "l_knee", "l_ankle","pelvis", "throax","upper_neck", "head_top",
                      "r_wrist", "r_elbow", "r_shoulder","l_shoulder", "l_elbow", "l_wrist"]

        MPIIBONES =  [(0, 1),(1, 2),(2, 6),(7, 12),(12, 11), (11, 10),(5, 4),(4, 3),(3, 6),(7, 13),(13, 14),(14, 15),(6, 7),
                      (7, 8),(8, 9)]
        MPIIr_ankle,MPIIr_knee,MPIIr_hip,MPIIl_hip,MPIIl_knee,MPIIl_ankle, MPIIpelvis,MPIIthroax = 0,1,2,3,4,5,6,7
        MPIIupper_neck,MPIIhead_top,MPIIr_wrist,MPIIr_elbow,MPIIr_shoulder,MPIIl_shoulder,MPIIl_elbow,MPIIl_wrist = 8,9,10,11,12,13,14,15

    else:
        raise ValueError("Your dataset name is wrong")


    """Return the total number of joints """
    @staticmethod
    def get_total_joints():
        return opt.totaljoints

    """Return the total number of bones """
    @staticmethod
    def get_total_bones():
        if opt.totaljoints == 13:
            return len(PoseConfig.BONES)
        elif opt.dataset == "MPII":
            return len(PoseConfig.MPIIBONES)
        else:
            raise ValueError("Your dataset name is wrong")


"""
Wrap a 2D pose (numpy array of size <PoseConfig.get_total_joints(),2> )
"""
class Pose2D:
    # HEAD,L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST,L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE,
    # The joints isn't in the same order in the differents datasets
    FROM_MPII_PERMUTATION = [0,1,2,3,4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    FROM_MPII_PERMUTATION_13 = [9, 13, 12, 14, 11, 15, 10, 3, 2, 4, 1, 5, 0]
    FROM_COCO_PERMUTATION = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    FROM_CROWDPOSE_PERMUTATION = [12, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    FROM_OCHUMAN_PERMUTATION = [0, 5, 6, 7, 8, 9, 10 ,11, 12, 13, 14, 15, 16]
    TO_HUMAN_36_PERMUTATION = [8, 10, 12, 7, 9, 11, 0, 1, 3, 5, 2, 4, 6]
    # FROM_COCO2_PERMUTATION = [0, 4, 1, 5, 2, 6, 3, 10, 7, 11, 8, 12, 9]
    # FROM_POSE2D_PERMUTATION = [0, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]


    def __init__(self, npArray):

        if len(npArray.shape) != 2 or npArray.shape[0] != PoseConfig.get_total_joints() or npArray.shape[1] != 2:
            raise Exception("Pose 2D only accepts numpy array with shape : <total joints, 2 DIM>")

        self.joints = npArray

        self.is_active_mask = []

        for joint_id in range(PoseConfig.get_total_joints()):
            self.is_active_mask.append(not np.array_equal(self.joints[joint_id, [0, 1]], [-1, -1]))

        self.is_active_mask = np.array(self.is_active_mask)


    """Build a 2D pose from a numpy coco ordered content"""
    @staticmethod
    def build_from_coco(npArray,datatype):

        if datatype == "coco" or datatype =="yoga":
            joints = npArray[Pose2D.FROM_COCO_PERMUTATION, :]
        elif datatype == "coco_mpii_13":
            joints = npArray[Pose2D.FROM_MPII_PERMUTATION_13,:]
        elif datatype == "coco_mpii":
            joints = npArray[Pose2D.FROM_MPII_PERMUTATION,:]
        elif datatype == "coco_crowd":
            joints = npArray[Pose2D.FROM_CROWDPOSE_PERMUTATION,:]
        elif datatype == "coco_ochuman":
            joints = npArray[Pose2D.FROM_OCHUMAN_PERMUTATION,:]
        elif datatype == "coco_human36":
            joints = npArray[Pose2D.TO_HUMAN_36_PERMUTATION,:]
        else:
            raise ValueError("Your dataset name is wrong")

        return Pose2D(joints)


    """Return the 2D joints as numpy array"""
    def get_joints(self):
        return self.joints.copy()


    """Return the total number of labeled joints (x and y position are != -1)"""
    def total_labeled_joints(self):
        return self.is_active_mask.sum()

    """Return the mask of labeled joints (x and y position are != -1)"""
    def get_active_joints(self):
        return self.is_active_mask.copy()

    """Return true if the given joint_id is labeled"""
    def is_active_joint(self, joint_id):
        return self.is_active_mask[joint_id]


    def distance_to(self, that):
        if opt.totaljoints == 13:
            mask_1 = that.get_active_joints()
            mask_2 = self.get_active_joints()
            mask = mask_1 & mask_2
        elif opt.dataset == "MPII":
            mask = [True,True,True,True,True,True,True,True,True,True,True,True,True,True,True,True]

        j1 = self.get_joints()[mask,:]
        j2 = that.get_joints()[mask, :]

        return np.sqrt(((j1 -j2)**2).sum(1)).mean(), np.sqrt(((self.get_joints() -that.get_joints())**2).sum(1)) ,mask

    def get_gravity_center(self):
        return self.joints[self.is_active_mask, :].mean(0)


    """Transform the pose in a bounding box or return the 100%, 100% box if impossible"""
    def to_bbox(self):

        if self.is_active_mask.sum() < 3:
            return BBox(0, 1, 0, 1)

        min_x, max_x = self.joints[self.is_active_mask, 0].min(), self.joints[self.is_active_mask, 0].max()
        min_y, max_y = self.joints[self.is_active_mask, 1].min(), self.joints[self.is_active_mask, 1].max()

        return BBox(min_x, max_x, min_y, max_y)


    """Return the pose in absolute coordinate if recorded from the given bbox"""
    def to_absolute_coordinate_from(self, bbox):

        joints = self.joints.copy()

        joints[self.is_active_mask, 0] = joints[self.is_active_mask, 0] * (bbox.get_max_x() - bbox.get_min_x()) + bbox.get_min_x()
        joints[self.is_active_mask, 1] = joints[self.is_active_mask, 1] * (bbox.get_max_y() - bbox.get_min_y()) + bbox.get_min_y()

        return Pose2D(joints)


    """Return the pose in the coordinate of the given bbox"""
    def to_relative_coordinate_into(self, bbox):

        joints = self.joints.copy()

        scale_x = bbox.get_max_x() - bbox.get_min_x()
        scale_y = bbox.get_max_y() - bbox.get_min_y()

        joints[self.is_active_mask, 0] = (joints[self.is_active_mask, 0] - bbox.get_min_x()) / scale_x
        joints[self.is_active_mask, 1] = (joints[self.is_active_mask, 1] - bbox.get_min_y()) / scale_y

        return Pose2D(joints)




    # def __str__(self):
    #     return self.joints.__str__()