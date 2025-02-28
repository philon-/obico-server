#!/usr/bin/env python

import flask
from flask_compress import Compress
from flask import abort, make_response, request, jsonify
from os import path, environ
import sentry_sdk
from sentry_sdk.integrations.flask import FlaskIntegration
import cv2
import numpy as np
import requests

from auth import token_required
from lib.detection_model import load_net, detect

SESSION_TTL_SECONDS = int(environ.get('SESSION_TTL_SECONDS', 120))
ML_THRESH = float(environ.get('ML_THRESH', 0.08))
ML_NMS = float(environ.get('ML_NMS', 0.45))
ML_API_PORT = float(environ.get('ML_API_PORT', 3333))
ML_API_HOST = environ.get('ML_API_HOST', '0.0.0.0')
ML_OUTPUT_TEXT = bool(environ.get('ML_OUTPUT_TEXT', False))

# Sentry
if environ.get('SENTRY_DSN'):
    sentry_sdk.init(
        dsn=environ.get('SENTRY_DSN'),
        integrations=[FlaskIntegration(), ],
    )

app = flask.Flask(__name__)
Compress(app)

status = dict()

# SECURITY WARNING: don't run with debug turned on in production!
app.config['DEBUG'] = environ.get('DEBUG') == 'True'

model_dir = path.join(path.dirname(path.realpath(__file__)), 'model')
net_main = load_net(path.join(model_dir, 'model.cfg'), path.join(model_dir, 'model.meta'))

@app.route('/p/', methods=['GET'])
@token_required
def get_p():
    if 'img' in request.args:
        try:
            resp = requests.get(request.args['img'], stream=True, timeout=(0.1, 5))
            resp.raise_for_status()
            img_array = np.array(bytearray(resp.content), dtype=np.uint8)
            img = cv2.imdecode(img_array, -1)
            detections = detect(net_main, img, thresh=ML_THRESH, nms=ML_NMS)
            return jsonify({'detections': detections})
        except Exception as err:
            sentry_sdk.capture_exception()
            app.logger.error(f"Failed to get image {request.args} - {err}")
            abort(
                make_response(
                    jsonify(
                        detections=[],
                        message=f"Failed to get image {request.args} - {err}",
                    ),
                    400,
                )
            )
    elif environ.get('ML_INPUT_IMAGE'):
        try:
            img = cv2.imread(environ.get('ML_INPUT_IMAGE'))
            detections = detect(net_main, img, thresh=ML_THRESH, nms=ML_NMS)
            if environ.get('ML_OUTPUT_IMAGE'):
                for d in detections:
                    (xc, yc, w, h) = map(int, d[2])
                    cv2.rectangle(img,
                        (xc-w//2,yc-h//2), (xc+w//2,yc+h//2),
                        (0, 255, 0), 2)
                    
                    if ML_OUTPUT_TEXT:
                        cv2.putText(img, str(round(d[1],2)), (xc-w//2,yc-h//2-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.75, (0, 255, 0), 2, cv2.LINE_AA)
                try:    
                    cv2.imwrite(environ.get('ML_OUTPUT_IMAGE'), img)
                except Exception as err: # Fail silently
                    app.logger.error(f"Failed to write output image {environ.get('ML_OUTPUT_FILENAME')} - {err}")
            return jsonify({'detections': detections})
        except Exception as err:
            sentry_sdk.capture_exception()
            app.logger.error(f"Failed to open input image {environ.get('ML_INPUT_FILENAME')} - {err}")
            abort(
                make_response(
                    jsonify(
                        detections=[],
                        message=f"Failed to open input image {environ.get('ML_INPUT_FILENAME')} - {err}",
                    ),
                    400,
                )
            )  
    else:
        app.logger.warn(f"Invalid request params: {request.args}")
        abort(
            make_response(
                jsonify(
                    detections=[], message=f"Invalid request params: {request.args}"
                ),
                422,
            )
        )


@app.route('/hc/', methods=['GET'])
def health_check():
    return 'ok' if net_main is not None else 'error'

if __name__ == "__main__":
    app.run(host=ML_API_HOST, port=ML_API_PORT, threaded=False)
