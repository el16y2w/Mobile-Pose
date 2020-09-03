from datetime import datetime
#prepare json file dataset
image_dir = 'img/tree'
json_dir = 'img/tree.json'


#Train model
#select data
dataformat = ['coco']
datanumber = 1 #multiple
dataset_comment = "MPIIimages and yoga"

#model type :
# mobilenetv1: inputsize(160,160), ouput size (10,10)
# mobilenetv2: inputsize(224,224), output size (14,14)
# mobilenetv3: inputsize(224,224), output size (7,7)
# hourglass: inputsize(256,256), output size (64,64)
# efficientnet : inputsize(224,224), output size(56,56)
# efficientnet : inputsize(224,224), output size(56,56)

#data prepare
train_annotFile = "img/mpiitrain.json"
train_imageDir = ""
test_annotFile = "img/mpiitrain.json"
test_imageDir = ""

# train_annotFile1 = "coco/person_keypoints_train2017.json"
# train_imageDir1 = "coco/images"
# test_annotFile1 = "coco/person_keypoints_val2017.json"
# test_imageDir1 = "coco/images"
# train_annotFile1 = "img/single_yoga2_train.json"
# train_imageDir1 = "img/single_yoga2_train"
# test_annotFile1 = "img/single_yoga2_test.json"
# test_imageDir1 = "img/single_yoga2_test"
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
