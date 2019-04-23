# Object detection prototype

Prototype implementation of object-oriented object detection from video using
OpenCV and Darknet YOLOv3 object detection and pre-trained model from Darknet.

Part of a project in planning, requirements specification and
design of an autonomous ship control software spring 2019 in
the course IMT2243 - Software Engineering at NTNU Gjøvik.

## Requirements

* Python
* [OpenCV](https://opencv.org/)
* [Numpy](https://www.numpy.org/)
* [Darknet](http://pjreddie.com/darknet/)


### Install instructions:

Make OpenCV from source.

Install Numpy and clone the repo:

```
pip3 install numpy

git clone https://github.com/tomrtk/imt2243-prototype.git
cd prototype

```

[Download and make Darknet with OpenCV and Cuda(if suported)](https://pjreddie.com/darknet/install/)

[Apply this patch to make Darknet to compile with OpenCV 4(april 2019)](https://patch-diff.githubusercontent.com/raw/pjreddie/darknet/pull/1348.patch)


Minimal install of Darknet, no direct OpenCV and CUDA suport
. From link above:

```
git clone https://github.com/pjreddie/darknet
cd darknet
make

mkdir yolo-coco
cd yolo-coco
wget https://pjreddie.com/media/files/yolov3.weights

cp -i ../darknet/cfg/yolov3.cfg .
cp -i ../darknet/data/coco.names .

```


## Demo

![Illustration of detection](https://github.com/tomrtk/imt2243-prototype/raw/master/demo/fig.png "Demo")

To run, `python3 src/objectdetection.py`, press `q` to exit.

Change path to video in `def main()` to run on other videos.

## Other references used

https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
