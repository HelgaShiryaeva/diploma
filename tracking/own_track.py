from utils import detector_utils as detector_utils
import cv2
from scipy.spatial import distance
import numpy as np
import datetime
import argparse
from collections import OrderedDict


class IouTracker:
    def __init__(self, maxLost=30, minActive=10, minIou=0.1):  # maxLost: maximum object lost counted when the object is being tracked
        self.nextObjectID = 0  # ID of next object
        self.objects = OrderedDict()  # stores ID:Locations
        self.lost = OrderedDict()  # stores ID:Lost_count
        self.active = OrderedDict()   # stores ID:Active_count
        self.finished = OrderedDict()  # stores ID:Active_count
        self.maxLost = maxLost  # maximum number of frames object was not detected.

    def addObject(self, new_object_location):
        self.objects[self.nextObjectID] = new_object_location  # store new object location
        self.lost[self.nextObjectID] = 0  # initialize frame_counts for when new object is undetected

        self.nextObjectID += 1

    def removeObject(self, objectID):  # remove tracker data after object is lost
        del self.objects[objectID]
        del self.lost[objectID]

    def update(self, detections):
        if len(detections) == 0:  # if no object detected in the frame
            lost_ids = list(self.lost.keys())
            for objectID in lost_ids:
                self.lost[objectID] += 1
                if self.lost[objectID] > self.maxLost:
                    self.removeObject(objectID)

            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(detections)):
                self.addObject(detections[i])
        else:
            objectIDs = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            D = distance.cdist(previous_object_locations,
                               detections)  # pairwise distance between previous and current

            row_idx = D.min(axis=1).argsort()  # (minimum distance of previous from current).sort_as_per_index

            cols_idx = D.argmin(axis=1)[row_idx]  # index of minimum distance of previous from current

            assignedRows, assignedCols = set(), set()

            for (row, col) in zip(row_idx, cols_idx):

                if row in assignedRows or col in assignedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = detections[col]
                self.lost[objectID] = 0

                assignedRows.add(row)
                assignedCols.add(col)

            unassignedRows = set(range(0, D.shape[0])).difference(assignedRows)
            unassignedCols = set(range(0, D.shape[1])).difference(assignedCols)

            if D.shape[0] >= D.shape[1]:
                for row in unassignedRows:
                    objectID = objectIDs[row]
                    self.lost[objectID] += 1

                    if self.lost[objectID] > self.maxLost:
                        self.removeObject(objectID)

            else:
                for col in unassignedCols:
                    self.addObject(detections[col])

        return self.objects


detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.2,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default=0,
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=640,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=480,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 10

    maxLost = 15  # maximum number of object losts counted when the object is being tracked
    tracker = IouTracker(maxLost=maxLost)

    model_info = {
        "object_names": {0: 'background', 1: 'hand'},
        "confidence_threshold": 0.5,
        "threshold": 0.4
        }
    np.random.seed(12345)
    bbox_colors = {key: np.random.randint(0, 255, size=(3,)).tolist() for key in model_info['object_names'].keys()}
    (H, W) = (im_height, im_width)  # input image height and width for the network
    writer = None

    while True:
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        ret, image = cap.read()
        if W is None or H is None: (H, W) = image.shape[:2]
        # image_np = cv2.flip(image_np, 1)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)

        boxes, confidences, classIDs = detector_utils.detect_objects(image,
                                                                     detection_graph, sess)

        if len(boxes) is not 0:
            print()

        # draw bounding boxes on frame
        detector_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                        confidences, boxes, im_width, im_height,
                                        image)

        detections_bbox = []  # bounding box for detections
        for i, box in enumerate(boxes):
            if confidences[i] > args.score_thresh:
                (left, right, top, bottom) = (boxes[i][1] * im_width, boxes[i][3] * im_width,
                                              boxes[i][0] * im_height, boxes[i][2] * im_height)
                # x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
                (left, right, top, bottom) = (int(left), int(right), int(top), int(bottom))

                detections_bbox.append((left, top, right, bottom))

                label = "{}:{:.4f}".format(model_info["object_names"][classIDs[i]], confidences[i])
                (label_width, label_height), baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                y_label = max(top, label_height)
                cv2.rectangle(image, (int(left), int(y_label - label_height)),
                          (int(left + label_width), int(y_label + baseLine)), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (left, y_label), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        objects = tracker.update(detections_bbox) # update tracker based on the newly detected objects
        print(len(detections_bbox))
        for (objectID, centroid) in objects.items():
            text = "ID {}".format(objectID)
            cv2.putText(image, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                       0.5, (0, 255, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        #cv2.imshow("image", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30, (int(W), int(H)), True)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        # Calculate Frames per second (FPS)
        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if args.display > 0:
            # Display FPS on frame
            if args.fps > 0:
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                 image)

            cv2.imshow('Hand tracking',
                       cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))
    writer.release()
