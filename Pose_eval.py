# -*- coding: utf-8 -*-
# @Time    : 18-7-10 上午9:41
# @Author  : zengzihua@huya.com
# @FileName: benchmark.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np
import json
import cv2
import os
import math
import time
from dataprovider.test.interface import AnnotatorInterface
import config

class poseevalpckh:
    def __init__(self):
        self.anno = json.load(open(config.Groundtru_annojson))
        self.testimg_path = config.testimg_path

    def eval(self,modelpath,inputsize):
        print("Total test example=%d" % len(self.anno['images']))
        images_anno = []
        keypoint_annos = {}
        transform = list(zip(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ))
        for img_info in self.anno['images']:
            images_anno.append(img_info['image_name'])

            prev_xs = img_info['keypoints'][0::2]
            prev_ys = img_info['keypoints'][1::2]

            new_kp = []
            for idx, idy in transform:
                new_kp.append(
                    (prev_xs[idx], prev_ys[idy])
                )

            keypoint_annos[img_info['image_name']] = new_kp

        pred_coords, use_times = poseevalpckh.infer(self.testimg_path, modelpath,inputsize, images_anno)

        scores = []
        for img_id in keypoint_annos.keys():
            groundtruth_anno = keypoint_annos[img_id]

            head_gt = groundtruth_anno[0]

            neck_gt = ((groundtruth_anno[1][0] + groundtruth_anno[2][0]) / 2, (groundtruth_anno[1][1]+ groundtruth_anno[2][1]) / 2)

            threshold = math.sqrt((head_gt[0] - neck_gt[0]) ** 2 + (head_gt[1] - neck_gt[1]) ** 2)
            curr_score = []
            #frame, key_points = Drawer.draw_scene(poses, ids, fps, None)
            if img_id in pred_coords.keys():

                for index, coord in enumerate(pred_coords[img_id]):
                    pred_x, pred_y = coord

                    gt_x, gt_y = groundtruth_anno[index]

                    d = math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)
                    if d > threshold:
                        curr_score.append(0)
                    else:
                        curr_score.append(1)
                scores.append(np.mean(curr_score))

        print("PCKh=%.2f" % (np.mean(scores) * 100))
        return use_times, "PCKh=%.2f" % (np.mean(scores) * 100)


    def infer(img_path, model, inputsize,images_anno, shape = config.evalmodelshape):
        annotator = AnnotatorInterface.build(model,inputsize,max_persons=1)
        res = {}

        use_times = []
        for img_id in images_anno:
            ori_img = cv2.imread(os.path.join('/media/hkuit104/MB155_4/PoseEstimation/poseval', img_id))
            ori_shape = ori_img.shape
            shape = shape
            inp_img = cv2.resize(ori_img, (shape[0], shape[1]))
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_RGB2GRAY)
            inp_img = cv2.cvtColor(inp_img, cv2.COLOR_GRAY2RGB)
            st = time.time()
            persons = annotator.update(inp_img)#{'id': person_id,'bbox':None,'pose_2d':None,'pose_3d':None,'confidence':np.array([0.25 for _ in range(PoseConfig.get_total_joints())]),'hash':self.person_hash_provider}
            #print(persons)
            for p in persons:
                pose = p['pose_2d'].get_joints()
                pose[:, 0] = (pose[:, 0]* ori_shape[1])
                pose[:, 1] = (pose[:, 1]* ori_shape[0])
                #bbox = [p['bbox'].get_min_x(inp_img), p['bbox'].get_min_y(inp_img), p['bbox'].get_max_x(inp_img), p['bbox'].get_max_y(inp_img)]
                res[img_id] = pose
            infer_time = 1000 * (time.time() - st)
            print("img_id = %s, cost_time = %.2f ms" % (img_id, infer_time))
            use_times.append(infer_time)
        print("Average inference time = %.2f ms" % np.mean(use_times))
        #result['Average inference time'].append(np.mean(use_times))
        annotator.terminate()
        return res, np.mean(use_times)



if __name__ == '__main__':
    dataset = config.testdataset
    inputsize = 224#config.modelinputseze[0]
    modelpath = './poseval/models/224/'
    resultpath = './poseval/results/'
    eval = poseevalpckh()
    models = os.walk(modelpath)
    for path, dir_list, file_list in models:
        for file_name in file_list:
            avertime, PCKH = eval.eval(modelpath+file_name,inputsize)
            txt_file = open(os.path.join(resultpath, str(file_name)[:-3]+"_"+dataset+".txt" ), "w+")
            txt_file.write("{}.pb: Average inference time :{}, PCKh:{}".
                    format(str(file_name)[:-3], avertime, str(PCKH)))
            txt_file.close()
    
