from flask import Flask
import flask
from PIL import Image
import cv2
import matplotlib.image as mpimg
import copy
from helpers import *
import boto3 # a low-level client representing S3
from config import S3_BUCKET, S3_KEY, S3_SECRET
import tempfile
import numpy as np
from keras.preprocessing import image
from zeep import Client

application = Flask(__name__)

@application.route('/')
@application.route('/index')
def index():
    return "Hello, World"    

@application.route('/detect')
def detect():
    # def get_s3_client():

    #     if S3_KEY and S3_SECRET:
    #         print("returning boto3 client")
    #         return boto3.client(
    #             's3',
    #             'us-west-2',
    #             aws_access_key_id = S3_KEY,
    #             aws_secret_access_key = S3_SECRET
    #         )
    #     else:
    #         return boto3.client('s3')

    # s3 = get_s3_client()
    # bucket = S3_BUCKET 

    # file = s3.get_object(Bucket=bucket, Key='image.png')
    
    # response = flask.Response(file['Body'].read(),
    #                     mimetype='image/gif')
    # print(file['Body'])

    # tmp = tempfile.NamedTemporaryFile()
    # with open(tmp.name, 'wb') as f:
    #     s3.download_fileobj(bucket,'image.png', f)
    #     print(f)
    #     print(tmp.name)
    #     img=mpimg.imread(tmp.name)
    #     im = cv2.imread(tmp.name)
    #     print(img)
    #     print(im)
    
    # print("temp ", tmp.name)

    IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/us/us3.jpg'

    # data_des = return_data_openalpr(tmp.name)
    # data_des = return_data_openalpr(IMAGE_PATH)
    data_des = {
        'coordinates': [{'y': 209, 'x': 207}, {'y': 212, 'x': 296}, {'y': 258, 'x': 296}, {'y': 255, 'x': 207}], 
        'state': 'il', 
        'plate': '9185914'
        }
    # min_xcoord, min_ycoord, max_xcoord, max_ycoord = get_coord(data_des)   
    min_xcoord, min_ycoord, max_xcoord, max_ycoord = 207, 209, 258, 296
    img = cv2.imread(IMAGE_PATH)
    img_crop = img[int(min_ycoord):int(max_ycoord),int(min_xcoord):int(max_xcoord)]
    height_img_crop = max_ycoord - min_ycoord
    
    # image processing colors
    img_gray = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)
    ddepth = cv2.CV_8U
    ret, th = cv2.threshold(img_gray,80,255,cv2.THRESH_BINARY_INV)
    
    # find contours in image
    img_contours = copy.copy(th)
    contours, hierarchy = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours_after_size_verification = []
    for contour in contours:
        if verifySize(th, contour, height_img_crop):
            contours_after_size_verification.append(contour)
    contour_num = 0
    x_locations = []
    pred_chars = {}
    for contour in contours_after_size_verification:
            x, y, w, h = cv2.boundingRect(contour)
            paddingw = int(w/3)
            paddingh = int(h/4)
            img_crop = cv2.getRectSubPix(img_contours, (w+paddingw,h+paddingh), (x+w/2,y+h/2))
            x_locations.append(x)
            resized_image = cv2.resize(img_crop,(224,224))
            cv2.imwrite("contour" + str(contour_num) + ".png",resized_image)
            img = cv2.imread("contour" + str(contour_num) +  ".png")
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis = 0)
            print(img.shape)

            # feed in image to machine learning model
            model = load_model_p()
            model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

            #predict the result
            result = model.predict(img)
            predictions = np.argmax(result, axis=1) #in an array
            character = list(CATEGORIES.keys())[list(CATEGORIES.values()).index(predictions[0])]
            print(character)      
            pred_chars[character] = x  
            
    print(pred_chars)
    sorted_chars = sorted(pred_chars.items(), key=lambda kv: kv[1])
    pred_plate = [j[0] for j in sorted_chars]
    pred_plate = ''.join(pred_plate)
    data = {"plate": data_des["plate"],
            "state": data_des["state"],
            "predicted": pred_plate}

    return flask.jsonify(data)

# @application.route("/predict", methods=["POST"])
@application.route("/predict")

def predict():
    # need to provide license plate from front end
    data = {'username': 'cyft369',
        'RegistrationNumber': 'YY07XHH'}

    # client = Client("http://www.regcheck.org.uk/api/reg.asmx?WSDL")
    # result = client.service.Check(data['RegistrationNumber'], data['username'])
    # vehicleJson = result.vehicleJson
    # cleaned_data = clean_vehicle_info_data(vehicleJson)
    
    # return flask.jsonify(cleaned_data)
    return flask.jsonify(clean_vehicle_info_data(fake_clean_data()))

if __name__ == '__main__':
    application.run(debug=True)