import tensorflow as tf
from dataprovider.pose_2d.interface import Pose2DInterface
from dataprovider.object_detection.interface import YoloInterface
import config

"""
Provide an instantiated model interface containing a method "predict" used for the inference.
"""

class ModelFactory:


    """
    Build the human detector model
    """
    @staticmethod
    def build_object_detection_interface():

        conf_thresh, nms_thresh = 0.25, 0.1

	    # default : yolo_tiny_single_class | also platinium-tiny (30% padding)
        config_file = "dataprovider/object_detection/tiny/yolo-voc.cfg"
        model_parameters = "dataprovider/object_detection/tiny/final.weights"

        return YoloInterface(config_file, model_parameters, conf_thresh, nms_thresh)


    """
    Build the 2 dimensional pose model
    """
    @staticmethod
    def build_pose_2d_interface(model = config.testmodel,inputsize = config.modelinputseze[0],inputnode =config.input_node_name,
                                outputnode = config.output_node_name ):

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)

        graph_file = model

        input_node_name, output_node_name = inputnode, outputnode

        input_size = inputsize

        #subject_padding = 0.3
        subject_padding = 0.4

        post_processing = Pose2DInterface.our_approach_postprocessing


        return Pose2DInterface(session, graph_file, post_processing, input_size, subject_padding, input_node_name, output_node_name)



