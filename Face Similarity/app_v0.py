from unittest import result
import cv2
from datetime import datetime
from PIL import Image
import io
import numpy as np
import requests

from flask import Flask
from flask import request

from frs_stream import FRS_Stream
import face_recognition

import re
import requests
import json

import logging
import os
import time

logger = logging.getLogger(__name__)
if (logger.hasHandlers()):
    logger.handlers.clear()

formatter = logging.Formatter('%(asctime)s | %(name)s |  %(levelname)s | %(message)s')
logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

logger.info("App started")

app = Flask('Face-Similarity')

model = FRS_Stream()
ocr_api_key = 'K83433112388957'
ocr_lang = 'eng'
ocr_overlay = 'true'

@app.route("/",methods=['GET'])
def hello():
    return "Face similarity model is Up and Running!" , 200


def frsProcessing(image):
    try:
        face_prediction = model.predict(image)
        embeddings = face_prediction['embeddings']
        detected_boxes = face_prediction['detected_boxes']            
        return embeddings, detected_boxes

    except Exception as E:
        error = "An Exception Occured: {}".format(E)
        logger.error(error)
        return False


def cvtFormatBox(de_box):
    detect_box = []
    for bx in de_box:
        detect_box.append([int(val) for val in bx])
    return detect_box


def callingFaceSearch(image_array):
    embeddings, detected_boxes = frsProcessing(image_array)
    detected_boxes  = cvtFormatBox(detected_boxes)
    return embeddings, detected_boxes


def transformImage(image_bytes):
    img = Image.open(image_bytes)
    arr = np.uint8(img)

    image_shape = arr.shape
    if image_shape[2] != 3: # convering image to 3 channels for extracting embeddings
        arr = arr[:,:,:3]
    return arr


def exractEmbeddings(license_image, test_image):
    lic_image = transformImage(license_image)
    tes_image = transformImage(test_image)
    
    lic_image_box = face_recognition.face_locations(lic_image)
    tes_image_box = face_recognition.face_locations(tes_image)
    
    lic_encoding = face_recognition.face_encodings(lic_image, lic_image_box)
    tes_encoding = face_recognition.face_encodings(tes_image, tes_image_box)

    return lic_encoding, tes_encoding


def searchSimilarity(emb_license_image, emb_test_image):
    check = []
    for emb_test in emb_test_image:
        check.extend(face_recognition.compare_faces(emb_license_image, emb_test))
    return check


def img_ocr(img_path):       
    ## To read image_path for ocr
    payload = {'isOverlayRequired': ocr_overlay,
        'apikey': ocr_api_key,
        'language': ocr_lang,
        }

    with open(img_path, 'rb') as f:
        r = requests.post('https://api.ocr.space/parse/image',
            files={img_path: f},
            data=payload,
            )

    con = json.loads(r.text)
    return con['ParsedResults'][0]['ParsedText']


def cleanLicense(ocr_text):
    cln_text = ocr_text.replace("\r", "\t")
    cln_text = cln_text.replace("\t", " ")
    cln_text = cln_text.replace("\n", "")

    license_number = None
    word_list = cln_text.split() # string to list

    for word in word_list:
        if (re.search('^[0-9]+$', word)) and len(word) == 8: # if word is a number
            license_number = word
    
    return license_number, cln_text


def extractLicense(license_image):
    lic_image = transformImage(license_image)
    lic_saved = Image.fromarray(lic_image)
    lic_saved_path = "license_saved_image.jpeg"
    lic_saved.save(lic_saved_path)

    ocr_text = img_ocr(lic_saved_path)

    if os.path.isfile(lic_saved_path):
        os.remove(lic_saved_path)
    
    return cleanLicense(ocr_text)


@app.route("/search",methods=['POST'])
def search_embeddings():
    if request.method == "POST":
        logger.info("Face Similarity Started!")
        try:
            main_image = request.files['main_image']
            test_image = request.files['test_image']
        except:
            logger.error("Error 400: Bad Input")
            return "Error 400: Bad Input", 400

        try:
            logger.info("Reading license image!")
            # emb_license_image, license_image_boxes = callingFaceSearch(main_image)
            
            logger.info("Reading similarity-test image!")
            # emb_test_image, test_image_boxes = callingFaceSearch(test_image)

            emb_license_image, emb_test_image = exractEmbeddings(main_image, test_image)

            matched = searchSimilarity(emb_license_image, emb_test_image)
            
            lic_text, ocr_text = extractLicense(main_image)
    
            if True in [val for val in matched]:
                return {"license_matched" : True, "license_number" : lic_text,\
                    "license_text": ocr_text }    

            return {"license_matched" : False, "license_number" : lic_text,\
                    "license_text": ocr_text }

        except Exception as E:
            error = "An Exception Occured: {}".format(E)
            logger.error(error)
            return error,500
                
    else:
        error = "Error 405: Method Not Allowed"
        logger.error(error)
        return error, 405


if __name__ == "__main__":
    app.run(debug=False)