from datetime import time
import cv2
from dataprovider.test.interface import AnnotatorInterface
from utils.drawer import Drawer
import time
import os
import json
import config



"""
Read the movie located at moviePath, perform the 2d pose annotation and display
Run from terminal : python demo_2d.py [movie_file_path] [max_persons_detected]
with all parameters optional.
Keep holding the backspace key to speed the video 30x
"""



def start_video(movie_path, max_persons):

    annotator = AnnotatorInterface.build(max_persons=max_persons)

    cap = cv2.VideoCapture(movie_path)

    while(True):

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        tmpTime = time.time()
        persons = annotator.update(frame)
        fps = int(1/(time.time()-tmpTime))

        poses = [p['pose_2d'] for p in persons]

        ids = [p['id'] for p in persons]
        frame,key_points = Drawer.draw_scene(frame, poses, ids, fps, cap.get(cv2.CAP_PROP_POS_FRAMES))

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if cv2.waitKey(33) == ord(' '):
            curr_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(curr_frame + 30))

    annotator.terminate()
    cap.release()
    cv2.destroyAllWindows()

class image_detection:
    def __init__(self,src_folder,dest_folder, max_persons):
        self.model = config.testmodel
        self.inputsize = config.modelinputseze[0]
        self.src_img_ls = [os.path.join(src_folder, img_name) for img_name in os.listdir(src_folder)]
        self.dest_img_ls = [os.path.join(dest_folder, img_name) for img_name in os.listdir(src_folder)]
        self.annotator = AnnotatorInterface.build(self.model,self.inputsize,max_persons=max_persons)
        self.idx = 0
        self.keypoints_json = []
        self.result = {}
        self.result_all = {}
        self.result_all['images'] = []
        self.json = open(src_folder + ".json", "w")
        self.tmpTime = time.time()


    def __process_img(self, img_path, dest_path):
        img = cv2.imread(img_path)
        ori_img = img
        img = cv2.resize(img,(640,480))
        persons = self.annotator.update(ori_img)
        fps = int(1 / (time.time() - self.tmpTime))
        poses = [p['pose_2d'] for p in persons]
        ids = [p['id'] for p in persons]
        frame, key_points = Drawer.draw_scene(ori_img, poses, ids, fps, None)
        #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(dest_path, frame)
        cv2.imshow('frame', frame)

        if len(poses)>0:
            self.__write_json(ori_img, poses[0].joints )

        cv2.waitKey(600)


    def __write_json(self, image, keypoints):
        keypoints[:,0]=(keypoints[:,0]*image.shape[0])
        keypoints[:, 1] = (keypoints[:, 1] * image.shape[0])
        for i in range(len(keypoints)):
            for j in range(len(keypoints[0])):
                self.keypoints_json.append(keypoints[i][j])
        self.result['image_name'] = str(self.src_img_ls[self.idx])
        self.result['id'] = self.idx
        self.result['keypoints'] = self.keypoints_json
        self.result_all['images'].append( self.result)
        self.result = {}
        self.keypoints_json = []

    def process(self):
        for self.idx in range(len(self.src_img_ls)):
            print("Processing image {}".format(self.idx))
            self.__process_img(self.src_img_ls[self.idx], self.dest_img_ls[self.idx])
            cv2.destroyAllWindows()
            # try:
            #     self.__process_img(self.src_img_ls[self.idx], self.dest_img_ls[self.idx])
            # except:
            #     print("wrong image is:", self.src_img_ls[self.idx])
        self.json.write(json.dumps(self.result_all))


if __name__ == "__main__":
    print("start frontend")
    #for webcame or video
    # max_persons = 1
    # default_media = 0
    # cam_num = 0
    # video = 'test3.mp4'
    # start_video(video, max_persons)

    #for output json
    max_persons = 1
    src_folder = 'testdata/moveitval'
    dest_folder = src_folder + "_kps"
    os.makedirs(dest_folder, exist_ok=True)
    image_detection(src_folder, dest_folder,max_persons).process()
    print('end')





