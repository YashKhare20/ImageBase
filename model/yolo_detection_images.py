import numpy as np
import time
import cv2
import os

def load_model(config_path, weights_path):
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    return net


class YoloModel:
    def __init__(self):
        self.confthres = 0.5
        self.nmsthres = 0.1
        self.yolo_path = "./"
        self.modelFolderName = 'model/coco'
        self.testImgNames = []
        self.testImgs = []
        self.LABELS = []
        self.currLabelsPath = ''

    def init_values(self, folder_name):
        self.modelFolderName = 'model/' + folder_name
        self.LABELS = open(self.modelFolderName + '/' + 'obj.names').read().strip().split("\n")

    def get_labels(self, labels_path):
        # load the COCO class labels our YOLO model was trained on
        # labelsPath = os.path.sep.join([yolo_path, "yolo_v3/coco.names"])
        # lpath=os.path.sep.join([yolo_path, labels_path])
        self.LABELS = open(labels_path).read().strip().split("\n")
        return self.LABELS

    @staticmethod
    def get_colors(labels):
        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
        return colors

    def get_weights(self, weights_path):
        # derive the paths to the YOLO weights and model configuration
        weights_path = os.path.sep.join([self.yolo_path, weights_path])
        return weights_path

    def get_prediction(self, image, net, labels, colors, index):
        (H, W) = image.shape[:2]
        orignalImage = image
        # determine only the *output* layer names that we need from YOLO
        ln = net.getLayerNames()
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

        # construct a blob from the input image and then perform a forward
        # pass of the YOLO object detector, giving us our bounding boxes and
        # associated probabilities
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        net.setInput(blob)
        start = time.time()
        layer_outputs = net.forward(ln)
        end = time.time()

        # show timing information on YOLO
        print("[INFO] YOLO took {:.6f} seconds".format(end - start))

        # initialize our lists of detected bounding boxes, confidences, and
        # class IDs, respectively
        boxes = []
        confidences = []
        class_ids = []

        # loop over each of the layer outputs
        for output in layer_outputs:
            # loop over each of the detections
            for detection in output:
                # extract the class ID and confidence (i.e., probability) of
                # the current object detection
                scores = detection[5:]
                # print(scores)
                class_id = np.argmax(scores)
                # print(class_id)
                confidence = scores[class_id]

                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.confthres:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")

                    # use the center (x, y)-coordinates to derive the top and
                    #  left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))

                    # update our list of bounding box coordinates, confidences,
                    # and class IDs
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # apply non-maxima suppression to suppress weak, overlapping bounding
        # boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.confthres,
                                self.nmsthres)

        allBoundingBoxes = {}
        label_list = []
        boundingBoxes = {}
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                tmp = []
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])

                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format(labels[class_ids[i]], confidences[i])
                if labels[class_ids[i]] not in label_list:
                    label_list.append(labels[class_ids[i]])

                tmp = self.bnd_box_to_yolo_line([x, y, x + w, y + h, class_ids[i], W, H])

                if labels[class_ids[i]] in allBoundingBoxes:
                    allBoundingBoxes[labels[class_ids[i]]].append(tmp)
                    boundingBoxes[labels[class_ids[i]]].append(boxes[i])
                else:
                    allBoundingBoxes[labels[class_ids[i]]] = [tmp]
                    boundingBoxes[labels[class_ids[i]]] = [boxes[i]]

                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # print(allBoundingBoxes)
        return [image, label_list, boundingBoxes, self.testImgNames[index], [W, H], allBoundingBoxes]

    def bnd_box_to_yolo_line(self, box):
        # box = literal_eval(coco_format_box)
        x_min = box[0]
        x_max = box[2]
        y_min = box[1]
        y_max = box[3]
        id = box[4]
        width = box[5]
        height = box[6]

        x_center = float((x_min + x_max)) / 2 / width
        y_center = float((y_min + y_max)) / 2 / height

        width = float((x_max - x_min)) / width
        height = float((y_max - y_min)) / height

        # PR387

        '''box_name = box[4]
        if box_name not in class_list:
            class_list.append(box_name)
            '''

        # return id, x_center, y_center, w, h
        yolo_box = [id, x_center, y_center, width, height]
        return yolo_box

    def runModel(self, image, ind):
        # load our input image and grab its spatial dimensions
        # image = cv2.imread(img)
        print('in runModel')
        labelsPath = self.modelFolderName + '/' + 'obj.names'
        self.currLabelsPath = self.modelFolderName + '/' + 'obj.names'
        cfgpath = self.modelFolderName + '/' + 'obj.cfg'
        wpath = self.modelFolderName + '/' + 'obj.weights'
        Lables = self.get_labels(labelsPath)
        Weights = self.get_weights(wpath)
        nets = load_model(cfgpath, Weights)
        Colors = self.get_colors(Lables)
        res = self.get_prediction(image, nets, Lables, Colors, ind)
        return res
