# USAGE
# python object_tracker.py --prototxt deploy.prototxt --model res10_300x300_ssd_iter_140000.caffemodel

# import the necessary packages
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.video import VideoStream
from imutils.video import FPS
import matplotlib.pyplot as plt
# %matplotlib notebook

import numpy as np
import argparse
import imutils
import time
import cv2


from datetime import datetime
import math
import time
from data import inputs
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import *
import os
import json
import csv

RESIZE_FINAL = 227
MAX_BATCH_SZ = 128
tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                               'What processing unit to execute inference on')
tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                              'Checkpoint basename')
tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')
FLAGS = tf.app.flags.FLAGS
def make_batch(image, coder, multicrop):
    """Process a single image file."""

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 100]
    # print(image)
    result, img_str = cv2.imencode('.jpg', image, encode_param)
    #
    image = coder.decode_jpeg(img_str.tostring(order='C'))
    crops = []

    if multicrop is False:
        print('Running a single image')
        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image = standardize_image(crop)

        crops.append(image)
    else:
        print('Running multi-cropped image')
        h = image.shape[0]
        w = image.shape[1]
        hl = h - RESIZE_FINAL
        wl = w - RESIZE_FINAL

        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        crops.append(standardize_image(crop))
        crops.append(tf.image.flip_left_right(crop))

        corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
        for corner in corners:
            ch, cw = corner
            cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
            crops.append(standardize_image(cropped))
            flipped = tf.image.flip_left_right(cropped)
            crops.append(standardize_image(flipped))

    image_batch = tf.stack(crops)

    return image_batch

def classify(sess, label_list, softmax_output, coder, images, image_file):
    # print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, FLAGS.single_look)
    batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})

    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)

    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))

    return best_choice


# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.5,
# 	help="minimum probability to filter weak detections")
# args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
def predict(video_file):
    ct = CentroidTracker()
    (H, W) = (None, None)

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")

    # initialize the video stream and allow the camera sensor to warmup
    print("[INFO] starting video stream...")
    stream = cv2.VideoCapture(video_file)
    fps = FPS().start()
    #vs = VideoStream("one.mp4").start()
    #time.sleep(2.0)

    # loop over the frames from the video stream
    while True:
    	# read the next frame from the video stream and resize it
    	(grabbed,frame) = stream.read()
    	if not grabbed:
                    break
    	frame = imutils.resize(frame, width=400)

    	# if the frame dimensions are None, grab them
    	if W is None or H is None:
    		(H, W) = frame.shape[:2]

    	# construct a blob from the frame, pass it through the network,
    	# obtain our output predictions, and initialize the list of
    	# bounding box rectangles
    	blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
    		(104.0, 177.0, 123.0))
    	net.setInput(blob)
    	detections = net.forward()
    	rects = []

    	# loop over the detections
    	for i in range(0, detections.shape[2]):
    		# filter out weak detections by ensuring the predicted
    		# probability is greater than a minimum threshold
    		if detections[0, 0, i, 2] > 0.5:
    			# compute the (x, y)-coordinates of the bounding box for
    			# the object, then update the bounding box rectangles list

    			# cv2.imshow("aaa",detections[0, 0, i, 0:3])
    			box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
    			# cv2.imshow("det",box)
    			rects.append(box.astype("int"))

    			# draw a bounding box surrounding the object so we can
    			# visualize it

    			(startX, startY, endX, endY) = box.astype("int")
    			det = frame[startY:endY, startX:endX]
    			cv2.rectangle(frame, (startX, startY), (endX, endY),
    				(0, 255, 0), 2)

    	# update our centroid tracker using the computed set of bounding
    	# box rectangles
    			objects = ct.update(rects,frame)

    	# i = imutils.resize(det, width=100)
    	# cv2.imwrite("1.jpg",i)
    	# loop over the tracked objects
    	for (objectID, centroid) in objects.items():
    		# print(objectID)
    		# draw both the ID of the object and the centroid of the
    		# object on the output frame
    		text = "ID {}".format(objectID)

    		cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
    			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    		cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    	# show the output frame
    	# cv2.imshow("Frame", frame)
    	# plt.imshow(frame)
    	# plt.show();
    	key = cv2.waitKey(1) & 0xFF
    	fps.update()

    	# if the `q` key was pressed, break from the loop
    	if key == ord("q"):
    		break
    # f = cv2.imread("t.jpg")
    # ct.pics.append([3,f])
    # do a bit of cleanup
    cv2.destroyAllWindows()
    fps.stop()
    # print("Total persons = "+str(len(ct.pics)))
    # for i in ct.pics:
    # 	p = imutils.resize(i[1], width=100)
    # 	cv2.imwrite(""+str(i[0])+".jpg",p)

    # tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            # tf.reset_default_graph()
            # FLAGS.model_type = "inception"
            # FLAGS.class_type = "gender"
            # FLAGS.model_dir = "21936"

            label_list = ['Male','Female']
            nlabels = len(label_list)

            # print('Executing on %s' % FLAGS.device_id)
            model_fn = select_model("inception")


            with tf.device("cpu:0"):

                images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
                soft = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])

                logits = model_fn(nlabels, images, 1, False)
                init = tf.global_variables_initializer()
                requested_step = None
                checkpoint_path = '%s' % ("21936")
                model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
                saver = tf.train.Saver()
                saver.restore(sess, model_checkpoint_path)
                softmax_output = tf.nn.softmax(logits)
                coder = ImageCoder()
                print("Total detected = "+str(len(ct.pics)))
                m = 0
                f = 0
                for i in ct.pics:
                    p = imutils.resize(i[1], width=100)
                    cv2.imwrite(""+str(i[0])+".jpg",p)
                    best_choice = classify(sess, label_list, softmax_output, coder, images, p)
                    if best_choice[0]=="Male":
                        m = m+1
                    else:
                        f = f+1
    return [len(ct.pics),m,f]
