"""
    Object detection class and subclasses.

    For prototype only visual object detection implemented.

"""
import cv2
import time
import numpy as np

class ObjectDetection:  # Dummy class for prototype

    def detectObject():
        pass


class VisualDetection(ObjectDetection):

    def __init__(self, weightsPath, configPath, labelPath, streamPath=0):
        """Constructor

        Setup it's own data from parameters and predefined values.

        @Arguments:
            weightsPath: Pre-trained yolo modell weights
            configPath: yolov3.cfg file for modell
            labelPath: coco.names file with names of objects
            streamPath: path to video, camera or webcam ip/url.
                        Defaults to built in webcam.

        """
        self.__weightsPath = weightsPath
        self.__configPath = configPath
        self.__streamPath = streamPath
        self.__confi = 0.5
        self.__thres = 0.3

        # load the COCO class labels our YOLO model was trained on
        self.LABELS = open(labelPath).read().strip().split("\n")

        (self.W, self.H) = (None, None)

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
	dtype="uint8")


    def __enter__(self):
        """__enter__

        Function called when creating object using 'with'

        """
        # Setup darknet & opencv using parameter set in __init__
        self.net = cv2.dnn.readNetFromDarknet(self.__configPath,
                                             self.__weightsPath)
        self.vs = cv2.VideoCapture(self.__streamPath)
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return self

    def __exit__(self, exc_type, exe_value, traceback):
        """
        Function called when exit a 'with' block for an object to cleanup

        """
        cv2.destroyAllWindows()
        self.vs.release()

        return True

    def detectObject(self):
        """Detects objects in video stream

        Read next frame in self.vs and detect objects in it

        For prototype image is also showed from this function. Would
        be moved to a show video stream function.
        """

        # Counter for frames and time to calc FPS
        frames = 0
        start = time.time()

        ok = True
        while( ok ):
            ok, frame = self.vs.read() # get next frame

            # if no frame
            if not ok:
                print("Error: no frame to prosess")
                break

            # Exit logic, if 'q' is pressed, breaks out of loop
            key = cv2.waitKeyEx(1) & 0xff
            if key == ord('q'):
                break

            if self.W is None or self.H is None:
                (self.H, self.W) = frame.shape[:2] # get size of frame


            # convert frame to req. format and call detection
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416),
                    swapRB=True, crop=False)

            out_det = self.yolo(blob, frame)

            # Display video and detection
            cv2.imshow("output", out_det)

            # Calculate FPS and print to terminal
            frames += 1
            fps = frames / ( time.time() - start )
            print("FPS: {0:.1f} ".format(fps))

    def yolo(self, blob, frame):
        """YOLO detection on frame,

        Tries to detect objects from YOLO-COCO weights and if found and
        confidence above threshold limits __confi and __thres, draw
        bounding boxes.

        TODO: Split up function

        """

        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        # initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
        boxes = []
        confidences = []
        classIDs = []


        # loop over each of the layer outputs
        for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > self.__confi:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([self.W, self.H,
                                                        self.W, self.H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

        # apply non-maxima suppression to suppress weak, overlapping
        # bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences,
                                self.__confi, self.__thres)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the frame
                color = [int(c) for c in self.COLORS[classIDs[i]]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(self.LABELS[classIDs[i]],
                                            confidences[i])
                cv2.putText(frame, text,
                            (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, color, 2)

        return frame


class Object:
    """Place holder class, not implemented

    """
    def __init__(self, x, y, z):
        self.__x = x
        self.__y = y
        self.__z = z



def main():

    # creates a detection object and start detection
    with VisualDetection("../yolo-coco/yolov3.weights",
                         "../yolo-coco/yolov3.cfg",
                         "../yolo-coco/coco.names"  ) as od:

        print("Detect object in video...")
        od.detectObject()

    del od


main() # Call main function to start

