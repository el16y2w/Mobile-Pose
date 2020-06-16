from datetime import datetime
#prepare json file dataset
image_dir = 'img/tree'
json_dir = 'img/tree.json'


#Train model
#select data
dataformat = ['yoga']
datanumber = 1 #multiple

#offset choose
offset = True #False

#gaussian setting
threshold = 12
sigma = 6

#model type :
# mobilenetv1: inputsize(160,160), ouput size (10,10)
# mobilenetv2: inputsize(224,224), output size (14,14)
# mobilenetv3: inputsize(224,224), output size (7,7)
# hourglass: inputsize(256,256), output size (64,64)
# efficientnet : inputsize(224,224), output size(56,56)
# efficientnet : inputsize(224,224), output size(56,56)
model = ['mobilenetv2']

#For mobilev3 parameters
v3_version = "small" #or large
v3_width_scale = 1 #[0.35,0.5,0.75,1,1.25]


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
inputSize = [(224,224)]
inputshape = [(None, 224, 224, 3)]
outputSize = [(56,56)]
batch = 8
inputshapeforflops = [(1, 224, 224, 3)]

#for tensorboard
SAVE_EVERY = 100
TEST_EVERY = 10
VIZ_EVERY = 300

#trainning
fromStep = 0
epoch = [3]
lr = [0.001]
isTrain = False # False
#checkpoints_file = 'output/shuffle/model-141200'  #load pretrain checkpoints
checkpoints_file = None
#save dir
checkpoinDir = 'output'  # checkpoint save dir
modeloutputFile = 'output'

#For DUC
pixelshuffle = [2]
convb = [[3,3,52,1,"psconv1"]]

#name
modelname = 'output/'

#comment :images number
total_images = 200000
dataset = 'senet, block=16'

#for test
testmodel = "mobilnetv2False2020-05-11-12-17-45.pb"
modelinputseze = (224,224)
input_node_name =  "Image:0"
output_node_name = "Output:0"

#For pose eval
testdataset = "single_yoga2_test"
Groundtru_annojson = 'poseval/data/gt/single_yoga2_test_gt.json'
testimg_path = 'poseval/img/single_yoga2_test'
evalmodelshape = (224,224)