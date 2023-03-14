#grabbing an image by dropping in ui and then converting the image to base64 (ie image to string)
import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d
import warnings
warnings = warnings.filterwarnings("ignore")

__class_name_to_number = {}
__class_number_to_name = {}

__model = None

def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image(file_path, image_base64_data)

    result = []
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(__model.predict(final)[0]),
            'class_probability': np.around(__model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': __class_name_to_number
        })

    return result

def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open(r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\artifacts\cricket_players_class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open(r'D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\artifacts\cricket_player_image_classification_model.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image(image_path, image_base64_data):               # function to create crop images by looping over every image
    face_cascade = cv2.CascadeClassifier(r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\opencv\haarcascades\haarcascade_frontalface_default.xml")
    
    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]                #roi - region of interest
        cropped_faces.append(roi_color)

    return cropped_faces
        

def get_b64_test_image_for_dhoni():
    with open(r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\b64.txt") as f:
        return f.read()

if __name__ == "__main__":
    load_saved_artifacts()

    print(classify_image(get_b64_test_image_for_dhoni(), None))


    # print(classify_image(None, r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\test_images\ms_dhoni.jpg"))
    # print(classify_image(None, r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\test_images\rohit_sharma.jpg"))
    # print(classify_image(None, r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\test_images\sachin_tendulkar.jpg"))
    # print(classify_image(None, r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\test_images\virat_kohli.jpg"))
    # print(classify_image(None, r"D:\PYTHON DATA SCIENCE INTERNSHIP\DS notes\mymlproject\server\web_server\test_images\yuvraj_singh.jpg"))
