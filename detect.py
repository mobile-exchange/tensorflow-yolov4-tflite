import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
from core.yolov4 import YOLO, decode, filter_boxes
from absl import app, flags, logging
from absl.flags import FLAGS
import core.utils as utils
from core.yolov4 import filter_boxes
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
import time
import os

flags.DEFINE_string('framework', 'tf', '(tf, tflite, trt')
flags.DEFINE_string('weights', './checkpoints/yolov4-tiny.h5',
                    'path to weights file')
flags.DEFINE_integer('size', 640, 'resize images to')
flags.DEFINE_boolean('tiny', True, 'yolo or yolo-tiny')
flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')
flags.DEFINE_string('source', '0', 'image source')
flags.DEFINE_float('iou', 0.45, 'iou threshold')
flags.DEFINE_float('score', 0.2, 'score threshold')

def yolo_for_inference():
    STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

    input_layer = tf.keras.layers.Input([FLAGS.size, FLAGS.size, 3])
    feature_maps = YOLO(input_layer, NUM_CLASS, FLAGS.model, FLAGS.tiny)
    bbox_tensors = []
    prob_tensors = []

    for i, fm in enumerate(feature_maps):
        if i == 0:
            output_tensors = decode(fm, FLAGS.size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
        else:
            output_tensors = decode(fm, FLAGS.size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, FLAGS.framework)
    bbox_tensors.append(output_tensors[0])
    prob_tensors.append(output_tensors[1])

    pred_bbox = tf.concat(bbox_tensors, axis=1)
    pred_prob = tf.concat(prob_tensors, axis=1)

    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=FLAGS.score, input_shape=tf.constant([FLAGS.size, FLAGS.size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)

    model = tf.keras.Model(input_layer, pred)
    # utils.load_weights(model, FLAGS.weights, FLAGS.model, FLAGS.tiny)
    model.load_weights(FLAGS.weights)
    model.summary()

    # if os.path.exists(FLAGS.weights):
    #     model.load_weights(FLAGS.weights)
    # else:
    #     model.save(FLAGS.weights)

    return model

def preprocess(frame, input_size):
    frame_size = frame.shape[:2]
    image_data = cv2.resize(frame, (input_size, input_size))
    image_data = image_data / 255.
    return tf.constant(np.asarray([image_data]).astype(np.float32))
    # return image_data[np.newaxis, ...].astype(np.float32)

@tf.function
def infer(model, X):
    pred_bbox = model(X)
    for key, value in pred_bbox.items():
        boxes = value[:, :, 0:4]
        pred_conf = value[:, :, 4:]

    boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
        boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
        scores=tf.reshape(
            pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
        max_output_size_per_class=50,
        max_total_size=50,
        iou_threshold=FLAGS.iou,
        score_threshold=FLAGS.score
    )
    return boxes, scores, classes, valid_detections

def main(_argv):
    input_size = FLAGS.size
    source = FLAGS.source

    try:
        source = int(source)
    except:
        pass

    model = yolo_for_inference()
    vid = cv2.VideoCapture(source)

    frame_id = 0
    while True:
        return_value, frame = vid.read()
        if return_value:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            if frame_id == vid.get(cv2.CAP_PROP_FRAME_COUNT):
                print("Video processing complete")
                break
            raise ValueError("No image! Try with another video format")
        

        X = preprocess(frame, input_size)
        boxes, scores, classes, valid_detections = infer(model, X)
        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        image = utils.draw_bbox(frame, pred_bbox)
        result = np.asarray(image)
        result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("result", result)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        frame_id += 1

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
