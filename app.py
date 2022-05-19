import cv2
import imutils
import numpy as np
from flask import Flask, Response
import random
import colorsys
import os

useCuda = os.environ.get("USE_CUDA", False)
video_url = os.environ.get("VIDEO_URL")
frame_width = 1200

app = Flask(__name__)

with open("model/coco.names","r", encoding="utf-8") as f:
    labels = f.read().strip().split("\n")

yolo_config_path = "model/yolov4.cfg"
yolo_weights_path = "model/yolov4.weights"

net = cv2.dnn.readNetFromDarknet(yolo_config_path, yolo_weights_path)

if useCuda:
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# OpenCV
cap = cv2.VideoCapture(video_url)

def get_random_bright_colors(size):
    for i in range(0,size-1):
        h,s,l = random.random(), 0.5 + random.random()/2.0, 0.4 + random.random()/5.0
        r,g,b = [int(256*i) for i in colorsys.hls_to_rgb(h,l,s)]
        yield (r,g,b)

def gen_frames(net, video_url, confidence_threshold, overlapping_threshold, labels = None, frame_resize_width=None):
    # List of colors to represent each class label with distinct bright color
    colors = list(get_random_bright_colors(len(labels)))

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    try:
        if not cap.isOpened():
            print("Error opening video stream or file")
            return

        yolo_width_height = (416, 416)
        while True:
            success, frame = cap.read()
            if not success:
                break

            if frame_resize_width:
                frame = imutils.resize(frame, width=frame_resize_width)
            (H, W) = frame.shape[:2]
    
            # Construct blob of frames by standardization, resizing, and swapping Red and Blue channels (RBG to RGB)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, yolo_width_height, swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(output_layers)
            boxes = []
            confidences = []
            classIDs = []
            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]
                    if confidence > confidence_threshold:
                        # Scale the bboxes back to the original image size
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # Remove overlapping bounding boxes
            bboxes = cv2.dnn.NMSBoxes(
                boxes, confidences, confidence_threshold, overlapping_threshold)
            if len(bboxes) > 0:
                for i in bboxes.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])
                    color = [int(c) for c in colors[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
                    
                    # draw bounding box title background
                    text_offset_x = x
                    text_offset_y = y
                    text_color = (255, 255, 255)
                    (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=1.0, thickness=1)[0]
                    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 80, text_offset_y - text_height + 4))
                    cv2.rectangle(frame, box_coords[0], box_coords[1], color, cv2.FILLED)
                    
                    # draw bounding box title
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        cap.release()
    

@app.route('/feed')
def video_feed():
    return Response(gen_frames(net, video_url, 0.6, 0.1, labels,frame_width), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)