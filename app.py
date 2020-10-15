import os
from PIL import Image
from flask import Flask, request, Response, render_template
import cv2
import io
from camera import VideoCamera
import json
from hand_tracker import HandTracker

app = Flask(__name__)

PALM_MODEL_PATH = "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "hand_landmark.tflite"
ANCHORS_PATH = "anchors.csv"

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
) 
# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response

def runModel(img, threshold):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = detector(image,threshold)

    output_json = list()
    for points_ in points:
        if points_ is not None:
            
            #save outputs into json
            model_output_json = {}
            model_output_json['x'] = list(points_[:,0])
            model_output_json['y'] = list(points_[:,1])

            output_json.append(model_output_json)
    return output_json

@app.route('/')
def index():
    # return Response('ETRI Object Detection Test 2019.09.27 #8')
    return render_template('index.html')


def gen(cam_):
    while True:
        #get camera frame
        frame_output = cam_.getFrame()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame_output + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


@app.route('/test')
def test():
    PATH_TO_TEST_IMAGES_DIR = '.'  # cwh
    TEST_IMAGE_PATHS = os.path.join(PATH_TO_TEST_IMAGES_DIR, 'multi_hands.png')

    img = cv2.imread(TEST_IMAGE_PATHS)
    threshold = 0.7
    return runModel(img, threshold)

@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image'].read()  # get the image

        threshold = request.form.get('threshold')
        if threshold is None:
         threshold = 0.5
        else:
         threshold = float(threshold)

        img = np.array(Image.open(io.BytesIO(image_file)))
        return runModel(img, threshold)

    except Exception as e:
        print('POST /image error: %e' % e)
        return e

if __name__ == '__main__':
     app.run(debug=True, host='0.0.0.0', port=8080)
