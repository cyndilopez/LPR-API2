import cv2
import requests
import json
import matplotlib.pyplot as plt
import base64
from config import OPENALPR_SECRET_KEY
def clean_data_results(data):
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
    error = 0.4
    minHeight = 30
    minAspect = 0.2
    maxAspect = aspect+aspect*error
    if charAspect > minAspect and charAspect < maxAspect and (h/img_height)>=min_char_to_plate_aspect:       
        return True
    else:
        return False

def return_data_openalpr(IMAGE_PATH):

    with open(IMAGE_PATH, 'rb') as image_file:
        img_base64 = base64.b64encode(image_file.read())

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

# {
#   "uuid": "",
#   "data_type": "alpr_results",
#   "epoch_time": 1563554171947,
#   "processing_time": {
#     "total": 576.9929999951273,
#     "plates": 94.29983520507812,
#     "vehicles": 480.1889999944251
#   },
#   "img_height": 375,
#   "img_width": 500,
#   "results": [
#     {
#       "plate": "9185914",
#       "confidence": 93.7087631225586,
#       "region_confidence": 99,
#       "vehicle_region": {
#         "y": 0,
#         "x": 64,
#         "height": 375,
#         "width": 375
#       },
#       "region": "il",
#       "plate_index": 0,
#       "processing_time_ms": 36.85621643066406,
#       "candidates": [
#         {
#           "matches_template": 1,
#           "plate": "9185914",
#           "confidence": 93.7087631225586
#         },
#         {
#           "matches_template": 1,
#           "plate": "918591",
#           "confidence": 80.28701782226562
#         }
#       ],
#       "coordinates": [
#         {
#           "y": 209,
#           "x": 207
#         },
#         {
#           "y": 212,
#           "x": 296
#         },
#         {
#           "y": 258,
#           "x": 296
#         },
#         {
#           "y": 255,
#           "x": 207
#         }
#       ],
#       "vehicle": {
#         "orientation": [
#           {
#             "confidence": 99.11315155029297,
#             "name": "180"
#           },
#           {
#             "confidence": 0.8563453555107117,
#             "name": "225"
#           },
#           {
#             "confidence": 0.03050280176103115,
#             "name": "135"
#           },
#           {
#             "confidence": 2.5378501504746964e-06,
#             "name": "0"
#           },
#           {
#             "confidence": 6.952996045583859e-08,
#             "name": "45"
#           },
#           {
#             "confidence": 2.7187482487533998e-08,
#             "name": "missing"
#           },
#           {
#             "confidence": 1.9718989108241658e-08,
#             "name": "270"
#           },
#           {
#             "confidence": 9.061150052502853e-09,
#             "name": "315"
#           },
#           {
#             "confidence": 9.841935044718753e-10,
#             "name": "90"
#           }
#         ],
#         "color": [
#           {
#             "confidence": 99.07832336425781,
#             "name": "white"
#           },
#           {
#             "confidence": 0.8844082951545715,
#             "name": "silver-gray"
#           },
#           {
#             "confidence": 0.030218694359064102,
#             "name": "gold-beige"
#           },
#           {
#             "confidence": 0.00497695105150342,
#             "name": "blue"
#           },
#           {
#             "confidence": 0.0012463860912248492,
#             "name": "black"
#           },
#           {
#             "confidence": 0.0005867808358743787,
#             "name": "yellow"
#           },
#           {
#             "confidence": 0.00012517017603386194,
#             "name": "green"
#           },
#           {
#             "confidence": 7.873159484006464e-05,
#             "name": "red"
#           },
#           {
#             "confidence": 3.4619952202774584e-05,
#             "name": "orange"
#           },
#           {
#             "confidence": 4.2736951400002e-06,
#             "name": "brown"
#           }
#         ],
#         "make": [
#           {
#             "confidence": 99.99864959716797,
#             "name": "ford"
#           },
#           {
#             "confidence": 0.0007877767784520984,
#             "name": "mercury"
#           },
#           {
#             "confidence": 0.0004813948180526495,
#             "name": "chevrolet"
#           },
#           {
#             "confidence": 5.5553202400915325e-05,
#             "name": "cadillac"
#           },
#           {
#             "confidence": 6.3391953517566435e-06,
#             "name": "shelby"
#           },
#           {
#             "confidence": 2.6975640139426105e-06,
#             "name": "buick"
#           },
#           {
#             "confidence": 2.258077529404545e-06,
#             "name": "mazda"
#           },
#           {
#             "confidence": 1.3764256436843425e-06,
#             "name": "aston-martin"
#           },
#           {
#             "confidence": 9.72486418504559e-07,
#             "name": "dodge"
#           },
#           {
#             "confidence": 9.36862875278166e-07,
#             "name": "scion"
#           }
#         ],
#         "body_type": [
#           {
#             "confidence": 99.99334716796875,
#             "name": "sedan-sports"
#           },
#           {
#             "confidence": 0.00621750857681036,
#             "name": "sedan-standard"
#           },
#           {
#             "confidence": 0.0003934820997528732,
#             "name": "motorcycle"
#           },
#           {
#             "confidence": 1.8562377590569668e-05,
#             "name": "truck-standard"
#           },
#           {
#             "confidence": 8.484757927362807e-06,
#             "name": "sedan-convertible"
#           },
#           {
#             "confidence": 1.520516661912552e-06,
#             "name": "sedan-wagon"
#           },
#           {
#             "confidence": 4.905758714812691e-07,
#             "name": "suv-crossover"
#           },
#           {
#             "confidence": 2.757722405988261e-08,
#             "name": "suv-standard"
#           },
#           {
#             "confidence": 1.8569892290543066e-08,
#             "name": "tractor-trailer"
#           },
#           {
#             "confidence": 5.603315589297608e-09,
#             "name": "antique"
#           }
#         ],
#         "year": [
#           {
#             "confidence": 88.11061096191406,
#             "name": "2005-2009"
#           },
#           {
#             "confidence": 10.45720386505127,
#             "name": "2010-2014"
#           },
#           {
#             "confidence": 1.4172221422195435,
#             "name": "2000-2004"
#           },
#           {
#             "confidence": 0.013416082598268986,
#             "name": "1995-1999"
#           },
#           {
#             "confidence": 0.0014636542182415724,
#             "name": "1990-1994"
#           },
#           {
#             "confidence": 7.90458798292093e-05,
#             "name": "1985-1989"
#           },
#           {
#             "confidence": 3.0213254831323866e-06,
#             "name": "1980-1984"
#           },
#           {
#             "confidence": 1.779733565854258e-06,
#             "name": "2015-2019"
#           },
#           {
#             "confidence": 2.5024817773555696e-08,
#             "name": "missing"
#           }
#         ],
#         "make_model": [
#           {
#             "confidence": 99.29950714111328,
#             "name": "ford_mustang"
#           },
#           {
#             "confidence": 0.6989838480949402,
#             "name": "ford_gt"
#           },
#           {
#             "confidence": 0.0009800600819289684,
#             "name": "chevrolet_camaro"
#           },
#           {
#             "confidence": 0.00012154193973401561,
#             "name": "mercury_grand-marquis"
#           },
#           {
#             "confidence": 7.879921031417325e-05,
#             "name": "cadillac_sts"
#           },
#           {
#             "confidence": 7.13369736331515e-05,
#             "name": "toyota_celica"
#           },
#           {
#             "confidence": 6.624715751968324e-05,
#             "name": "ford_f-150"
#           },
#           {
#             "confidence": 5.879783202544786e-05,
#             "name": "ford_flex"
#           },
#           {
#             "confidence": 4.6700752136530355e-05,
#             "name": "ford_ranger"
#           },
#           {
#             "confidence": 3.954941712436266e-05,
#             "name": "cadillac_seville"
#           }
#         ]
#       },
#       "matches_template": 1,
#       "requested_topn": 10
#     }
#   ],
#   "credits_monthly_used": 3,
#   "version": 2,
#   "credits_monthly_total": 1000,
#   "error": False,
#   "regions_of_interest": [
#     {
#       "y": 0,
#       "x": 0,
#       "height": 375,
#       "width": 500
#     }
#   ],
#   "credit_cost": 1
# }