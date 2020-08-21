import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


"----------------------------- Training options -----------------------------"
tf.app.flags.DEFINE_integer("batch", 8, "batch size")
tf.app.flags.DEFINE_integer("epoch", 2000, "Current epoch")
tf.app.flags.DEFINE_integer("fromStep", 0, "Initial epoch")
tf.app.flags.DEFINE_integer("SAVE_EVERY", 10, "tensorboard save")
tf.app.flags.DEFINE_integer("TEST_EVERY", 1, "tensorboard test")
tf.app.flags.DEFINE_integer("VIZ_EVERY", 10, "tensorboard viz")

tf.app.flags.DEFINE_string("Model_folder_name", 'A_0', "Model_folder_name")

tf.app.flags.DEFINE_boolean('isTrain', True, 'trainable or not')
tf.app.flags.DEFINE_boolean('offset', True, 'offset')

tf.app.flags.DEFINE_string("backbone", 'mobilenetv2', "backbone:mobilenetv1/mobilenetv2"
                                                      "/mobilenetv3/hourglass/efficientnet")
tf.app.flags.DEFINE_string("modeloutputFile", 'Yogapose', "model output dir")
tf.app.flags.DEFINE_string("checkpoints_file", None, " checkpoints file")
tf.app.flags.DEFINE_string("checkpoinsaveDir", 'Yogapose', " checkpoints save dir")
tf.app.flags.DEFINE_string("train_all_result", 'Result/Yogapose', "model name")



"----------------------------- Data options -----------------------------"
tf.app.flags.DEFINE_integer("inputResH", 224, "Input image height")
tf.app.flags.DEFINE_integer("inputResW", 224, "Input image width")
tf.app.flags.DEFINE_integer("outputResH", 56, "Output image height")
tf.app.flags.DEFINE_integer("outputResW", 56, "Output image width")

tf.app.flags.DEFINE_boolean('grayimage', False, 'image type')


"----------------------------- Hyperparameter options -----------------------------"
#lr
tf.app.flags.DEFINE_string("lr_type", "exponential_decay","exponential_decay|polynomial_decay|natural_exp_decay|cosine_decay_restarts")
tf.app.flags.DEFINE_float("lr", 0.001, "learning rate")
tf.app.flags.DEFINE_float("decay_rate", 0.95, "learning rate decay rate")
tf.app.flags.DEFINE_integer("decay_steps", 1000, "learning rate decay steps")

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
tf.app.flags.DEFINE_integer("require_improvement",300, "如果在#轮内没有改进，停止迭代")



"----------------------------- Eval options -----------------------------"
tf.app.flags.DEFINE_string("testdataset", "single_yoga2_test", "testdataset")
tf.app.flags.DEFINE_integer("inputsize", 224, "Input image")
tf.app.flags.DEFINE_string("Groundtru_annojson", 'poseval/data/gt/single_yoga2_test_gt.json', "Groundtru_annojson")
tf.app.flags.DEFINE_string("modelpath", './poseval/models/280/', "testing models path")
tf.app.flags.DEFINE_string("testing_path", 'poseval/img/single_yoga2_test', "testing dataset path")
tf.app.flags.DEFINE_string("resultpath", './poseval/results/', "result output path")



"----------------------------- Realtime testing options -----------------------------"
tf.app.flags.DEFINE_string("testmodel", "mobilnetv2False2020-05-11-12-17-45.pb", "testmodel")
tf.app.flags.DEFINE_string("input_node_name", "Image:0", "input_node_name")
tf.app.flags.DEFINE_string("output_node_name", "Output:0", "output_node_name")
tf.app.flags.DEFINE_integer("modelinputsize", 224, "Input image")


opt = FLAGS


