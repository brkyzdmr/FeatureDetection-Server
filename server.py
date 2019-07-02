import base64
import cv2 as cv
import jsonpickle
import numpy as np
import os
import time
from PIL import Image
from flask import Flask, request, Response
from resizeimage import resizeimage
from siftdetector import detect_keypoints

app = Flask(__name__)


# @app.route('/', methods=['POST'])
def standard():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    cv.imwrite("target.jpeg", img)

    millis1 = int(round(time.time() * 1000))
    path = os.getcwd() + "/data_20"
    target_path = os.getcwd() + "/target.jpeg"
    target_des = find_descriptor(target_path)

    max_percent, max_percent_img = match_descriptors(path, target_des)
    millis2 = int(round(time.time() * 1000))
    time_milis = (millis2 - millis1)

    print("Max percent : ", max_percent, " And filename : ", max_percent_img)
    print("Geçen zaman = ", time_milis)

    target_img = os.getcwd() + "/dataset/" + max_percent_img + ".jpeg"
    with open(target_img, 'rb') as open_file:
        byte_content = open_file.read()
    base64_bytes = base64.b64encode(byte_content)
    base64_img = base64_bytes.decode('utf-8')

    response = {'image_destination': max_percent_img,
                'process_time': time_milis,
                'accuracy': max_percent,
                'image': base64_img
                }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


def get_projected_des(target_des):
    path = os.getcwd() + "/projection_vectors/projection_vec_6.txt"
    projection_vec = get_projection_vector(path)
    projected_des = np.dot(target_des, projection_vec.T)

    return projected_des

def get_projection_vector(path):
    projection_vec = np.loadtxt(path, dtype=float)

    return projection_vec


@app.route('/', methods=['POST'])
def random_projection():
    r = request
    nparr = np.fromstring(r.data, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    cv.imwrite("target.jpeg", img)

    millis1 = int(round(time.time() * 1000))
    path = os.getcwd() + "/projected_data/projected_data_6"
    target_path = os.getcwd() + "/target.jpeg"
    target_des = find_descriptor(target_path)

    target_des = get_projected_des(target_des)

    max_percent, max_percent_img = match_descriptors(path, target_des)
    millis2 = int(round(time.time() * 1000))
    time_milis = (millis2 - millis1)

    print("Max percent : ", max_percent, " And filename : ", max_percent_img)
    print("Geçen zaman = ", time_milis)

    target_img = os.getcwd() + "/dataset/" + max_percent_img + ".jpeg"
    with open(target_img, 'rb') as open_file:
        byte_content = open_file.read()
    base64_bytes = base64.b64encode(byte_content)
    base64_img = base64_bytes.decode('utf-8')

    response = {'image_destination': max_percent_img,
                'process_time': time_milis,
                'accuracy': max_percent,
                'image': base64_img
                }

    response_pickled = jsonpickle.encode(response)

    return Response(response=response_pickled, status=200, mimetype="application/json")


def resize_image(path):
    fd_img = open(path, "rb")
    img = Image.open(fd_img)
    img = resizeimage.resize_width(img, 200)
    img.save(path, img.format)
    fd_img.close()


def find_descriptor(path):
    resize_image(path)
    kp, target_des = detect_keypoints(path, 0.01)

    return target_des


def get_descriptors(path):
    descriptors = {}

    print(path)

    for r, d, f in os.walk(path):
        for file in f:
            if '.txt' in file:
                file_name = path + "/" + file
                destination = file.split(".txt")[0]
                descriptors[destination] = np.loadtxt(file_name, dtype=float)

    return descriptors


def match_descriptors(path, target_des):
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv.FlannBasedMatcher(index_params, search_params)
    count = 0
    maxpercent = 0

    descriptors = get_descriptors(path)

    for file_name, descriptor in descriptors.items():
        matches = flann.knnMatch(np.asarray(target_des, np.float32), np.asarray(descriptor, np.float32), k=2)
        good_points = []
        ratio = 0.6

        for m, n in matches:
            if m.distance < ratio * n.distance:
                good_points.append(m)

        if len(matches) != 0:
            percent = len(good_points) / len(matches) * 100
        else:
            percent = 0
        print(count, " - ", file_name, " : ", percent)
        count = count + 1
        if percent >= maxpercent:
            maxpercent = percent
            maxpercentImg = file_name

    return maxpercent, maxpercentImg


def main():
    app.run(host="0.0.0.0", port=5000)


if __name__ == "__main__":
    main()
