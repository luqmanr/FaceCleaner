from __future__ import print_function

import os
import operator
import logging
import settings
import utils
import tensorflow as tf

# Communication to TensorFlow server via gRPC
from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2


log = logging.getLogger(__name__)

def __get_tf_server_connection_params__():
    '''
    Returns connection parameters to TensorFlow Server

    :return: Tuple of TF server name and server port
    '''
    server_name = utils.get_env_var_setting('TF_SERVER_NAME', settings.DEFAULT_TF_SERVER_NAME)
    server_port = utils.get_env_var_setting('TF_SERVER_PORT', settings.DEFAULT_TF_SERVER_PORT)

    return server_name, server_port

def __create_prediction_request__(image):
    '''
    Creates prediction request to TensorFlow server for GAN model

    :param: Byte array, image for prediction
    :return: PredictRequest object
    '''
    # create predict request
    request = predict_pb2.PredictRequest()

    # Call GAN model to make prediction on the image
    request.model_spec.name = settings.GAN_MODEL_NAME
    request.model_spec.signature_name = settings.GAN_MODEL_SIGNATURE_NAME
    request.inputs[settings.GAN_MODEL_INPUTS_KEY].CopyFrom(
        tf.contrib.util.make_tensor_proto(image, shape=[1]))

    return request

def __open_tf_server_channel__(server_name, server_port):
    '''
    Opens channel to TensorFlow server for requests

    :param server_name: String, server name (localhost, IP address)
    :param server_port: String, server port
    :return: Channel stub
    '''
    channel = implementations.insecure_channel(
        server_name,
        int(server_port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    return stub

def __make_prediction_and_prepare_results__(stub, request):
    '''
    Sends Predict request over a channel stub to TensorFlow server

    :param stub: Channel stub
    :param request: PredictRequest object
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    log.info('asdasdasdasdsadaasdsads')
    log.info(type(request))
    
    try:
        result = stub.Predict(request, 60.0)  # 60 secs timeout
        log.info(result)
        boxes = result.outputs['detection_boxes'].float_val
        classes = result.outputs['detection_classes'].float_val
        scores = result.outputs['detection_scores'].float_val    
    except Exception as e: 
        log.info(e)
        
    a = [('e', 27.910110473632812), ('g', 22.20656967163086)]
    return a

def make_prediction(image):
    '''
    Predict the house number on the image using GAN model

    :param image: Byte array, images for prediction
    :return: List of tuples, 3 most probable digits with their probabilities
    '''
    # get TensorFlow server connection parameters
    log.info(type(image))
    server_name, server_port = __get_tf_server_connection_params__()
    log.info('Connecting to TensorFlow server %s:%s', server_name, server_port)

    # open channel to tensorflow server
    stub = __open_tf_server_channel__(server_name, server_port)
    log.info('asdasdasdasd')

    # create predict request
    request = __create_prediction_request__(image)
    log.info('asdasdasdasdsadas')

    # make prediction
    return __make_prediction_and_prepare_results__(stub, request)

