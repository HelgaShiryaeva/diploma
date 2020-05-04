from utils import detector_utils as detector_utils
import cv2
from scipy.spatial import distance
import numpy as np
import datetime
import argparse
from collections import OrderedDict


class IouTracker:
    def __init__(self, active_threshold=10, non_active_threshold=10):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.potential_active = OrderedDict()
        self.lost = OrderedDict()
        self.active = OrderedDict()
        self.finished = OrderedDict()
        self.non_active_threshold = non_active_threshold
        self.active_threshold = active_threshold

    def add_object(self, new_object_location):
        self.objects[self.next_object_id] = new_object_location
        self.lost[self.next_object_id] = 0
        self.next_object_id += 1

    def remove_object(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]

    def update(self, detections):
        if len(detections) == 0:
            lost_ids = list(self.lost.keys())
            for object_id in lost_ids:
                self.lost[object_id] += 1
                if self.lost[object_id] > self.non_active_threshold:
                    self.remove_object(object_id)

            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(detections)):
                self.add_object(detections[i])
        else:
            object_ids = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            dist = distance.cdist(previous_object_locations, detections)

            row_idx = dist.min(axis=1).argsort()
            cols_idx = dist.argmin(axis=1)[row_idx]
            assigned_rows, assigned_cols = set(), set()

            for (row, col) in zip(row_idx, cols_idx):

                if row in assigned_rows or col in assigned_cols:
                    continue

                object_id = object_ids[row]
                self.objects[object_id] = detections[col]
                self.lost[object_id] = 0

                assigned_rows.add(row)
                assigned_cols.add(col)

            unassigned_rows = set(range(0, dist.shape[0])).difference(assigned_rows)
            unassigned_cols = set(range(0, dist.shape[1])).difference(assigned_cols)

            if dist.shape[0] >= dist.shape[1]:
                for row in unassigned_rows:
                    object_id = object_ids[row]
                    self.lost[object_id] += 1

                    if self.lost[object_id] > self.non_active_threshold:
                        self.remove_object(object_id)

            else:
                for col in unassigned_cols:
                    self.add_object(detections[col])

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
    width, height = (cap.get(3), cap.get(4))

    non_active_threshold = 8
    active_threshold = 10
    tracker = IouTracker(active_threshold=active_threshold, non_active_threshold=non_active_threshold)

    writer = None

    while True:
        ret, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, confidences, classIDs = detector_utils.detect_objects(image, detection_graph, sess)

        detections_bbox = detector_utils.draw_box_on_image(args.score_thresh, confidences, boxes, width, height, image)

        objects = tracker.update(detections_bbox)

        for (objectID, centroid) in objects.items():
            text = "id{}".format(objectID)
            cv2.putText(image, text, (centroid[0] - 5, centroid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30, (int(width), int(height)), True)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        num_frames += 1
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = num_frames / elapsed_time

        if args.display > 0:
            if args.fps > 0:
                detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), image)

            cv2.imshow('Hand tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ", elapsed_time, "fps: ", str(int(fps)))
    writer.release()
