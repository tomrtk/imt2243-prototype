"""


"""
import cv2
import numpy as np
import sys, os

sys.path.append(os.path.join(os.getcwd(),'python/'))
from scipy.misc import imread
import darknet as dn
import pdb

class ObjectDetection:

    def detectObject():
        pass


class VisualDetection(ObjectDetection):

    def __init__(self, weightsPath, configPath, dataPath, streamPath=0):
        self.__weightsPath = weightsPath
        self.__configPath = configPath
        self.__dataPath = dataPath
        self.__streamPath = streamPath


    def __enter__(self):
        net = dn.load_net(self.__configPath, self.__weightsPath, 0)
        meta = dn.load_meta(__dataPath)
        return self

    def __exit__(self, exc_type, exe_value, traceback):
        self.vs.release()

        return True

    def detectObject():

        while True:
            (ok, frame) = vs.read() # get next frame

            if not ok:
                break

            (H, W) = frame.shape[:2] # get size of frame

            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                    swapRB=True, crop=False)

            frame = yolo(blob)

            cv2.imshow("output", frame)
            print("output")

    def yolo(blob):



        return frame


class Object:
    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z


def main():

    with VisualDetection('../yolo-coco/yolov3.weights', '../yolo-coco/yolov3.cfg', '../test_tracking.mp4') as od:
        print("Detekterer")
        od.detectObject()


main()

