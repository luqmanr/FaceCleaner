import io
import json
import time
import logging


from flask import request
from flask_restplus import Resource
from api.restplus import api
from api.gan.logic.tf_serving_client import make_prediction
from werkzeug.datastructures import FileStorage

log = logging.getLogger(__name__)
# create dedicated namespace for GAN client
ns = api.namespace('gan_client', description='Operations for GAN client')

# Flask-RestPlus specific parser for image uploading
UPLOAD_KEY = 'inputs'
UPLOAD_LOCATION = 'files'
upload_parser = api.parser()
upload_parser.add_argument(UPLOAD_KEY,
                           location=UPLOAD_LOCATION,
                           type=FileStorage,
                           required=True)


@ns.route('/prediction')
class GanPrediction(Resource):
    @ns.doc(description='Predict the house number on the image using GAN model. ' +
            'Return 3 most probable digits with their probabilities',
            responses={
                200: "Success",
                400: "Bad request",
                500: "Internal server error"
                })
    @ns.expect(upload_parser)
    def post(self):
        try:
            image_file = request.files[UPLOAD_KEY]
            log.info(type(image_file))
            image = io.BytesIO(image_file.read())
            log.info(type(image))
            
        except Exception as inst:
            return {'message': 'something wrong with incoming request. ' +
                               'Original message: {}'.format(inst)}, 400

        try:
            start = time.time()
            results = make_prediction(image.read())
            end = time.time()
            log.info('Process time taken :%s', end-start)
            results_json = [{'classes': res[0], 'scores': res[1]} for res in results]
            return {'prediction_result': results_json}, 200

        except Exception as inst:
            return {'message': 'internal error: {}'.format(inst)}, 500
