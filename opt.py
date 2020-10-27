import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


"----------------------------- Training options -----------------------------"
tf.app.flags.DEFINE_integer("batch", 8, "batch size")
tf.app.flags.DEFINE_integer("epoch", 10, "Current epoch")
tf.app.flags.DEFINE_integer("fromStep", 1, "Initial epoch")
tf.app.flags.DEFINE_integer("SAVE_EVERY", 50, "tensorboard save")
tf.app.flags.DEFINE_integer("TEST_EVERY", 1, "tensorboard test")
tf.app.flags.DEFINE_integer("VIZ_EVERY", 50, "tensorboard viz")

tf.app.flags.DEFINE_string("Model_folder_name", 'A_1', "Model_folder_name")

tf.app.flags.DEFINE_boolean("Early_stopping",False,"early stop or not")
tf.app.flags.DEFINE_boolean('isTrain', False, 'trainable or not')
tf.app.flags.DEFINE_boolean('isTrainpre',False,'if pre train,set false')
tf.app.flags.DEFINE_boolean('offset', True, 'offset')

tf.app.flags.DEFINE_string("backbone", 'mobilenetv2', "backbone:mobilenetv1/mobilenetv2"
                                                      "/mobilenetv3/hourglass/efficientnet/resnet18")
tf.app.flags.DEFINE_string("modeloutputFile", 'trash', "model output dir")
tf.app.flags.DEFINE_string("checkpoints_file", None, " checkpoints file")
tf.app.flags.DEFINE_string("checkpoinsaveDir", 'trash', " checkpoints save dir")
tf.app.flags.DEFINE_string("train_all_result", 'Result/trash', "model name")



"----------------------------- Data options -----------------------------"
tf.app.flags.DEFINE_boolean("checkanno", False,"check annotation")

tf.app.flags.DEFINE_string("dataset",'MPII_13',"choose data format:MPII_13/MPII/COCO/YOGA")
tf.app.flags.DEFINE_integer("totaljoints", 13, "MPII16/MPII_13/COCO13/YOGA13")

tf.app.flags.DEFINE_integer("inputResH", 224, "Input image height")
tf.app.flags.DEFINE_integer("inputResW", 224, "Input image width")
tf.app.flags.DEFINE_integer("outputResH", 56, "Output image height")
tf.app.flags.DEFINE_integer("outputResW", 56, "Output image width")

tf.app.flags.DEFINE_boolean('grayimage', False, 'image type')


"----------------------------- Hyperparameter options -----------------------------"
#lr
tf.app.flags.DEFINE_string("lr_type", "polynomial_decay","exponential_decay|polynomial_decay|inverse_time_decay|cosine_decay")
tf.app.flags.DEFINE_float("lr", 0.00095, "learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.98, "learning rate decay rate")
tf.app.flags.DEFINE_integer("decay_steps", 5000, "learning rate decay steps")

#optimizer
tf.app.flags.DEFINE_float("epsilon", 1e-8, "epsilon")
tf.app.flags.DEFINE_string("optimizer", 'Gradient', "Adam/Momentum/Gradient")
tf.app.flags.DEFINE_float("momentum", 0.9, "Momentum value")

tf.app.flags.DEFINE_integer("gaussian_thres", 12, "gaussian threshold")
tf.app.flags.DEFINE_integer("gaussian_sigma", 6, "gaussian sigma")
tf.app.flags.DEFINE_integer("v3_width_scale", 1, "for mobilenetv3:0.35,0.5,0.75,1,1.25")
tf.app.flags.DEFINE_string("v3_version", 'small', "for mobilenetv3:small/large")

#ACTIVATE
tf.app.flags.DEFINE_string("activate_function", 'relu', "swish/relu")

#loss
tf.app.flags.DEFINE_integer("epsilon_loss", 2, "wing_loss(2)/AdapWingLoss(1)")
tf.app.flags.DEFINE_integer("w", 2, "wing_loss(10)/AdapWingLoss(14)")
tf.app.flags.DEFINE_string("hm_lossselect", 'l2', "l2/wing/adaptivewing/smooth_l1")

#EARLY STOPPING
tf.app.flags.DEFINE_integer("j_min", 5, "if j_num>j_min,begin to decay lr")
tf.app.flags.DEFINE_integer("j_max", 10, "if j_num>j_max,stop training")
tf.app.flags.DEFINE_integer("test_epoch", 50, "每50轮迭代输出状态test")
tf.app.flags.DEFINE_integer("require_improvement",300, "如果在#轮内没有改进，停止迭代")

#model compression
tf.app.flags.DEFINE_integer("depth_multiplier", 1, "control the output channel")


"----------------------------- Eval options -----------------------------"
tf.app.flags.DEFINE_string("testdataset", "single_yoga2_test", "testdataset")
tf.app.flags.DEFINE_integer("inputsize", 224, "Input image")
tf.app.flags.DEFINE_string("Groundtru_annojson", 'poseval/data/gt/single_yoga2_test_gt.json', "Groundtru_annojson")
tf.app.flags.DEFINE_string("modelpath", './poseval/models/280/', "testing models path")
tf.app.flags.DEFINE_string("testing_path", 'poseval/img/single_yoga2_test', "testing dataset path")
tf.app.flags.DEFINE_string("resultpath", './poseval/results/', "result output path")



"----------------------------- Realtime testing options -----------------------------"
tf.app.flags.DEFINE_string("testmodel", "mobilenetv2False2020-09-22-09-45-20.pb", "testmodel")
tf.app.flags.DEFINE_string("input_node_name", "Image:0", "input_node_name")
tf.app.flags.DEFINE_string("output_node_name", "Output:0", "output_node_name")
tf.app.flags.DEFINE_integer("modelinputsize", 224, "Input image")

opt = FLAGS


