from flask import Flask
import flask

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import copy
from helpers import clean_data_results, verifySize, return_data_openalpr, get_coord
import boto3 # a low-level client representing S3
from config import S3_BUCKET, S3_KEY, S3_SECRET
import tempfile

application = Flask(__name__)

@application.route('/')
@application.route('/index')
def index():
    return "Hello, World"    

@application.route('/detect')
def detect():
    def get_s3_client():
        print(S3_KEY)
        print(S3_SECRET)
        if S3_KEY and S3_SECRET:
            print("returning boto3 client")
            return boto3.client(
                's3',
                'us-west-2',
                aws_access_key_id = S3_KEY,
                aws_secret_access_key = S3_SECRET
            )
        else:
            return boto3.client('s3')

    s3 = get_s3_client()
    bucket = S3_BUCKET 

    file = s3.get_object(Bucket=bucket, Key='image.png')
    
    response = flask.Response(file['Body'].read(),
                        mimetype='image/gif')
    print(response)
    print(flask.make_response(file['Body'].read()))

    tmp = tempfile.NamedTemporaryFile()
    with open(tmp.name, 'wb') as f:
        s3.download_fileobj(bucket,'image.png', f)
        img=mpimg.imread(tmp.name)
        im = cv2.imread(tmp.name)
        print(img)
        print(im)
    

    # IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/us/us3.jpg'

    # # data_des = return_data_openalpr(IMAGE_PATH)
    
    
    # # min_xcoord, min_ycoord, max_xcoord, max_ycoord = get_coord(data_des)   
    # min_xcoord, min_ycoord, max_xcoord, max_ycoord = 207, 209, 258, 296
    # img = cv2.imread(IMAGE_PATH)
    # img_crop = img[int(min_ycoord):int(max_ycoord),int(min_xcoord):int(max_xcoord)]
    # height_img_crop = max_ycoord - min_ycoord
    
    # # image processing colors
    # img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    # ddepth = cv2.CV_8U
    # ret, th = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    
    # # find contours in image
    # img_contours = copy.copy(th)
    # contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # contours_after_size_verification = []
    # for contour in contours:
    #     if verifySize(th, contour, height_img_crop):
    #         contours_after_size_verification.append(contour)

    # for contour in contours_after_size_verification:
    #         x, y, w, h = cv2.boundingRect(contour)
    #         paddingw = int(w/5)
    #         paddingh = int(h/5)
    #         img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
    #         resized_image = cv2.resize(img_crop,(34,85))
    #         # feed in image to machine learning model
    
    data = {"ok": True,
            "status": 201}
    return flask.jsonify(data )

@application.route("/predict", methods=["POST"])
def predict():
    print("Predicting")
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    data["success"] = True
    print(flask.request.data.image)
    if flask.request.files.get("image"):
        print("got image")
    elif flask.request.form["image"]:
        print("image in second form")
    #     # read the image in PIL format
    #     image = flask.request.files["image"].read()
    #     print(image)
    #     print("read image")
    #     image = Image.open(io.BytesIO(image))
    #     print(image)
    #     print("opened image")
    else:
        print("didnt get image")
    #     # preprocess the image and prepare it for classification
    print("after if statement")
    # #     # # classify the input image and then initialize the list
    # #     # # of predictions to return to the client
    # #     # preds = model.predict(image)
    # #     # results = imagenet_utils.decode_predictions(preds)
    # #     # data["predictions"] = []

    # #     # # loop over the results and add them to the list of
    # #     # # returned predictions
    # #     # for (imagenetID, label, prob) in results[0]:
    # #     #     r = {"label": label, "probability": float(prob)}
    # #     #     data["predictions"].append(r)

    # #     # indicate that the request was a success

    # # return the data dictionary as a JSON response
    return flask.jsonify(data)

if __name__ == '__main__':
    application.run(debug=True)