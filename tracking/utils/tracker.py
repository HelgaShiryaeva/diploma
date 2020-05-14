from sklearn.utils.linear_assignment_ import linear_assignment
import numpy as np
from collections import OrderedDict


class Tracker:
    def __init__(self, active_threshold=10, non_active_threshold=10, iou_threshold=0.3):
        self.next_object_id = 0
        self.objects = OrderedDict()
        self.potential_active = OrderedDict()
        self.lost = OrderedDict()
        self.active = OrderedDict()
        self.finished = OrderedDict()
        self.non_active_threshold = non_active_threshold
        self.active_threshold = active_threshold
        self.iou_threshold = iou_threshold

    def add_object(self, new_object_location):
        self.objects[self.next_object_id] = new_object_location
        self.lost[self.next_object_id] = 0
        self.potential_active[self.next_object_id] = 1
        self.next_object_id += 1

    def remove_object(self, object_id):
        del self.objects[object_id]
        del self.lost[object_id]
        del self.potential_active[object_id]

    def update(self, detections):
        if len(detections) == 0:
            lost_ids = list(self.lost.keys())
            for object_id in lost_ids:
                self.lost[object_id] += 1
                self.potential_active[object_id] = 0
                if self.lost[object_id] > self.non_active_threshold:
                    self.remove_object(object_id)

            return self.objects

        if len(self.objects) == 0:
            for i in range(0, len(detections)):
                self.add_object(detections[i])
        else:
            object_ids = list(self.objects.keys())
            previous_object_locations = np.array(list(self.objects.values()))

            matches, unmatched_detections, unmatched_trackers = \
                self.associate_detections_to_trackers(detections, previous_object_locations, self.iou_threshold)

            for (row, col) in matches:
                object_id = object_ids[col]
                self.objects[object_id] = detections[row]
                self.lost[object_id] = 0
                self.potential_active[object_id] += 1

            for row in unmatched_trackers:
                object_id = object_ids[row]
                self.potential_active[object_id] -= 1
                self.lost[object_id] += 1

                if self.lost[object_id] > self.non_active_threshold:
                    self.remove_object(object_id)

            for col in unmatched_detections:
                self.add_object(detections[col])

        active_objects = dict(
            filter(
            lambda elem: self.potential_active[elem[0]] > self.active_threshold, self.objects.items()
            )
        )

        return active_objects


    def iou(self, bbox1, bbox2):
        (left1, top1, right1, bottom1) = bbox1
        (left2, top2, right2, bottom2) = bbox2

        left = max(left1, left2)
        top = max(top1, top2)
        right = min(right1, right2)
        bottom = min(bottom1, bottom2)

        if bottom < top or right < left:
            return 0.0

        intersection_area = (bottom - top) * (right - left)
        bb1_area = (bottom1 - top1) * (right1 - left1)
        bb2_area = (bottom2 - top2) * (right2 - left2)

        union = float(bb1_area + bb2_area - intersection_area)
        iou = intersection_area / union

        assert iou >= 0.0
        assert iou <= 1.0

        return iou

    def associate_detections_to_trackers(self, detections, trackers, iou_threshold=0.4):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)

        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self.iou(det, trk)
        matched_indices = linear_assignment(-iou_matrix)

        unmatched_detections = []
        for d, det in enumerate(detections):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t, trk in enumerate(trackers):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

