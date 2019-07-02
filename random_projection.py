from siftdetector import detect_keypoints
import cv2 as cv
import numpy as np
import os
from server import get_descriptors


def create_projection_vector(x=1, y=128):
    projection_vec = np.random.uniform(low=-1, high=1, size=(x, y))
    np.savetxt("projection_vec_2.txt", projection_vec, fmt='%f')


def get_projection_vector(path):
    projection_vec = np.loadtxt(path, dtype=float)

    return projection_vec


def calculate_projections(projection_vec, descriptors):
    for file_name, descriptor in descriptors.items():
        projection = np.dot(descriptor, projection_vec.T)
        projection_path = os.getcwd() + "/projected_data_2/" + file_name + ".txt"
        np.savetxt(projection_path, projection, fmt='%f')


def main():
    projection_vec = get_projection_vector("projection_vec_2.txt")
    descriptors = get_descriptors(os.getcwd() + "/data_20")

    calculate_projections(projection_vec, descriptors)

if __name__ == "__main__":
    main()