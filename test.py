import requests
import base64
import json
OPENALPR_SECRET_KEY = 'sk_45a5804f6619b15dd88a84b4'
def clean_data_results(data):
        des_data = {
            'coordinates': data["coordinates"],
            'state': data["region"],
            'plate': data["plate"]
        }
        return des_data
        
def return_data_openalpr(IMAGE_PATH):

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

IMAGE_PATH = '/Users/cylopez/Documents/projects/license-plate-recognition/plates_mania_lp/california/extracted.png/image_2_20.png'
# data_des = return_data_openalpr(tmp.name)
data_des = return_data_openalpr(IMAGE_PATH)