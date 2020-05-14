from utils import detector_utils as detector_utils
from utils.tracker import Tracker
import cv2
import datetime
import argparse

detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for displaying bounding boxes')
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
        default=1080,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=720,
        help='Height of the frames in the video stream.')
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    start_time = datetime.datetime.now()
    num_frames = 0
    width, height = (cap.get(3), cap.get(4))

    non_active_threshold = 15
    active_threshold = 8
    tracker = Tracker(active_threshold=active_threshold, non_active_threshold=non_active_threshold)

    writer = None

    while True:
        start_time = datetime.datetime.now()
        ret, image = cap.read()

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, confidences, class_ids = detector_utils.detect_objects(image, detection_graph, sess)

        boxes, confidences = detector_utils.filter_boxes(args.score_thresh, confidences, boxes, width, height)

        detector_utils.draw_box_on_image(confidences, boxes, image)

        objects = tracker.update(boxes)

        for (object_id, centroid) in objects.items():
            text = "id{}".format(object_id)
            cv2.putText(image, text, (centroid[0] - 5, centroid[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.circle(image, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter("output.avi", fourcc, 30, (int(width), int(height)), True)
        writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = 1 / elapsed_time

        detector_utils.draw_fps_on_image("FPS : " + str(int(fps)), image)

        cv2.imshow('Hand tracking', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    writer.release()
