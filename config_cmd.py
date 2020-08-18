from datetime import datetime
#prepare json file dataset
image_dir = 'img/tree'
json_dir = 'img/tree.json'


#Train model
#select data
dataformat = ['yoga']
datanumber = 1 #multiple

#model type :
# mobilenetv1: inputsize(160,160), ouput size (10,10)
# mobilenetv2: inputsize(224,224), output size (14,14)
# mobilenetv3: inputsize(224,224), output size (7,7)
# hourglass: inputsize(256,256), output size (64,64)
# efficientnet : inputsize(224,224), output size(56,56)
# efficientnet : inputsize(224,224), output size(56,56)

#data prepare
train_annotFile = "img/tree.json"
train_imageDir = "img/tree"
test_annotFile = "img/tree.json"
test_imageDir = "img/tree"
# train_annotFile1 = "coco/person_keypoints_train2017.json"
# train_imageDir1 = "coco/images"
# test_annotFile1 = "coco/person_keypoints_val2017.json"
# test_imageDir1 = "coco/images"
# train_annotFile2 = "img/moveittest.json"
# train_imageDir2 = "img/moveittest"
# test_annotFile2 = "img/moveitval.json"
# test_imageDir2 = "img/moveitval"
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
convb = [[3,3,52,1,"psconv1"]]

#comment :images number
total_images = 200000
dataset = 'senet, block=16'
