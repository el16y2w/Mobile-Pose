from datetime import datetime
#prepare json file dataset
image_dir = 'img/tree'
json_dir = 'img/tree.json'


#Train model
#select data
dataformat = ['yoga']#coco_mpii_13,coco_mpii_16,yoga,coco
datanumber = 1 #multiple
dataset_comment = "coco_ochuman"

#model type :
# mobilenetv1: inputsize(160,160), ouput size (10,10)
# mobilenetv2: inputsize(224,224), output size (14,14)
# mobilenetv3: inputsize(224,224), output size (7,7)
# hourglass: inputsize(256,256), output size (64,64)
# efficientnet : inputsize(224,224), output size(56,56)
# efficientnet : inputsize(224,224), output size(56,56)

#data prepare
# train_annotFile = "img/crowdpose/annotations/crowdpose_val.json"
# train_imageDir = "img/crowdpose/images"
# test_annotFile = "img/crowdpose/annotations/crowdpose_val.json"
# test_imageDir = "img/crowdpose/images"
# train_annotFile = "img/ochuman/ochuman_coco_format_test_range_0.00_1.00.json"
# train_imageDir = "img/ochuman/images"
# test_annotFile = "img/ochuman/ochuman_coco_format_test_range_0.00_1.00.json"
# test_imageDir = "img/ochuman/images"
# train_annotFile = "img/mpiitrain.json"
# train_imageDir = ""
# test_annotFile = "img/mpiitrain.json"
# test_imageDir = ""
# train_annotFile1 = "coco/person_keypoints_train2017.json"
# train_imageDir1 = "coco/images"
# test_annotFile1 = "coco/person_keypoints_val2017.json"
# test_imageDir1 = "coco/images"
train_annotFile = "img/ai_add_searchedyoga_train.json"
train_imageDir = "img/ai_add_searchedyoga_train"
test_annotFile = "img/ai_add_searchedyoga_test.json"
test_imageDir = "img/ai_add_searchedyoga_test"
dataprovider_trainanno =[train_annotFile]
dataprovider_trainimg = [train_imageDir]
dataprovider_testanno = [test_annotFile]
dataprovider_testimg = [test_imageDir]
inputSize = (224,224)
inputshape = (None, 224, 224, 3)
outputSize = (56,56)

inputshapeforflops = [(1, 224, 224, 3)]

#For DUC
pixelshuffle = [2]
convb_13 = [[3,3,52,1,"psconv1"]]
convb_16 = [[3,3,64,1,"psconv1"]]

#comment :images number
total_images = 200000
dataset = 'senet, block=16'
