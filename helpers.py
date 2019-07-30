import cv2
import requests
import json
import base64
# from config import OPENALPR_SECRET_KEY
from keras.models import load_model
import config
OPENALPR_SECRET_KEY = 'sk_45a5804f6619b15dd88a84b4'

CATEGORIES = {'1': 0, '2': 1, '3': 2, '4': 3, '5': 4, '6': 5, '7': 6, '8': 7, '9': 8, 
                'A': 9, 'B': 10, 'C': 11, 'D': 12, 'E': 13, 'F': 14, 'G': 15, 'H': 16, 
                'I': 17, 'J': 18, 'K': 19, 'L': 20, 'M': 21, 'N': 22, 'O': 23, 'P': 24, 
                'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29, 'W': 30, 'X': 31, 'Y': 32, 
                'Z': 33}

def clean_data_results(data):
        print("in here")
        des_data = {
            'coordinates': data["coordinates"],
            'state': data["region"],
            'plate': data["plate"]
        }
        return des_data

def verifySize(th, contour, img_height):
    # char sizes are 45x77
    # char sizes are 2.5"x2.5*(3 to 4)
    x, y, w, h = cv2.boundingRect(contour)
    min_char_to_plate_aspect = 40/224
    aspect = 2.5/(2.5*2.5) #based on eyeballing
    charAspect = w/h
    error = 0.6
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error
    max_char_to_plate_aspect = min_char_to_plate_aspect+min_char_to_plate_aspect*error
    if charAspect > minAspect and charAspect < maxAspect and (h/img_height)>=min_char_to_plate_aspect:       
        return True
    else:
        return False

def return_data_openalpr(IMAGE_PATH):
    print("in here")
    with open(IMAGE_PATH, 'rb') as image_file:
        img_base64 = base64.b64encode(image_file.read())

    print(OPENALPR_SECRET_KEY)
    url = 'https://api.openalpr.com/v2/recognize_bytes?recognize_vehicle=1&country=us&secret_key=%s' % (OPENALPR_SECRET_KEY)
    r = requests.post(url, data = img_base64)

    # print(json.dumps(r.json(), indent=2))
    data = json.loads(r.text)
    print(data)
    data_des = clean_data_results(data["results"][0])
    print("data list ", data_des)
    return data_des

def get_coord(data_des):
        min_xcoord = min(pair["x"] for pair in data_des["coordinates"])
        min_ycoord = min(pair["y"] for pair in data_des["coordinates"])
        max_xcoord = max(pair["x"] for pair in data_des["coordinates"])
        max_ycoord = max(pair["y"] for pair in data_des["coordinates"])
        return min_xcoord, min_ycoord, max_xcoord, max_ycoord

def load_model_p():
        model = load_model('resnet50-weights.04-1.35.hdf5')
        return model

def clean_vehicle_info_data(data):
        cleaned_data = {
                "description":data["Description"],
                "registration_year":data["RegistrationYear"],
                "car_make":data["CarMake"]["CurrentTextValue"],
                "car_model":data["CarModel"]["CurrentTextValue"],
                "engine_size":data["EngineSize"]["CurrentTextValue"],
                "body_style":data["BodyStyle"]["CurrentTextValue"],
                "vehicle_identification_number":data["VechileIdentificationNumber"],
        }
        return cleaned_data
def fake_clean_data():
        return {
                "Description": "PEUGEOT 307 X-LINE",
                "RegistrationYear": "2007",
                "CarMake": {
                "CurrentTextValue": "PEUGEOT"
                },
                "CarModel": {
                "CurrentTextValue": "307 X-LINE"
                },
                "MakeDescription": {
                "CurrentTextValue": "PEUGEOT"
                },
                "ModelDescription": {
                "CurrentTextValue": "307 X-LINE"
                },
                "EngineSize": {
                "CurrentTextValue": "1360"
                },
                "BodyStyle": {
                "CurrentTextValue": "Motorbike"
                },
                "FuelType": {
                "CurrentTextValue": "PETROL"
                },
                "Variant": "",
                "Colour": "SILVER",
                "VehicleIdentificationNumber": "VF33CKFUC84922414",
                "KType": "",
                "EngineNumber": "FE040407358",
                "ImageUrl": "https://www.regcheck.org.uk/image.aspx/@UEVVR0VPVCAzMDcgWC1MSU5FfG1vdG9yY3ljbGU="
                }
