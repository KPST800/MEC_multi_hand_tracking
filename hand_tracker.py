import csv
import cv2
import numpy as np
import tensorflow as tf
''' modified from:https://github.com/metalwhale/hand_tracking '''
def non_max_suppression_fast(boxes, probabilities=None, overlap_threshold=0.3):
    """
    Algorithm to filter bounding box proposals by removing the ones with a too low confidence score
    and with too much overlap.
    Source: https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
    :param boxes: List of proposed bounding boxes
    :param overlap_threshold: the maximum overlap that is allowed
    :return: filtered boxes
    """
    # if there are no boxes, return an empty list
    if boxes.shape[1] == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0] - (boxes[:, 2] / [2])  # center x - width/2
    y1 = boxes[:, 1] - (boxes[:, 3] / [2])  # center y - height/2
    x2 = boxes[:, 0] + (boxes[:, 2] / [2])  # center x + width/2
    y2 = boxes[:, 1] + (boxes[:, 3] / [2])  # center y + height/2

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = boxes[:, 2] * boxes[:, 3]  # width * height
    idxs = y2


    # if probabilities are provided, sort on them instead
    if probabilities is not None:
        idxs = probabilities

    # sort the indexes
    idxs = np.argsort(idxs)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlap_threshold)[0])))
    # return only the bounding boxes that were picked
    return pick
class HandTracker():
    r"""
    Class to use Google's Mediapipe HandTracking pipeline from Python.
    So far only detection of a single hand is supported.
    Any any image size and aspect ratio supported.
    Args:
        palm_model: path to the palm_detection.tflite
        joint_model: path to the hand_landmark.tflite
        anchors_path: path to the csv containing SSD anchors
    Ourput:
        (21,2) array of hand joints.
    Examples::
        >>> det = HandTracker(path1, path2, path3)
        >>> input_img = np.random.randint(0,255, 256*256*3).reshape(256,256,3)
        >>> keypoints, bbox = det(input_img)
    """

    def __init__(self, palm_model, joint_model, anchors_path,
                box_enlarge=1.5, box_shift=0.2):
        self.box_shift = box_shift
        self.box_enlarge = box_enlarge

        self.interp_palm = tf.lite.Interpreter(palm_model)
        self.interp_palm.allocate_tensors()
        self.interp_joint = tf.lite.Interpreter(joint_model)
        self.interp_joint.allocate_tensors()

        # reading the SSD anchors
        with open(anchors_path, "r") as csv_f:
            self.anchors = np.r_[
                [x for x in csv.reader(csv_f, quoting=csv.QUOTE_NONNUMERIC)]
            ]
        # reading tflite model paramteres
        output_details = self.interp_palm.get_output_details()
        input_details = self.interp_palm.get_input_details()
        

        self.in_idx = input_details[0]['index']
        self.out_reg_idx = output_details[0]['index']
        self.out_clf_idx = output_details[1]['index']

        self.in_idx_joint = self.interp_joint.get_input_details()[0]['index']
        self.out_idx_joint = self.interp_joint.get_output_details()[0]['index']

        # 90° rotation matrix used to create the alignment trianlge
        self.R90 = np.r_[[[0,1],[-1,0]]]

        # trianlge target coordinates used to move the detected hand
        # into the right position
        self._target_triangle = np.float32([
                        [128, 128],
                        [128,   0],
                        [  0, 128]
                    ])
        self._target_box = np.float32([
                        [  0,   0, 1],
                        [256,   0, 1],
                        [256, 256, 1],
                        [  0, 256, 1],
                    ])
        self.kp_orig_list = list()
        self.img_landmarks = list()
        self.Mtrs = list()
        self.source_multi = list()

    def _get_triangle(self, kp0, kp2, dist=1):
        """get a triangle used to calculate Affine transformation matrix"""

        dir_v = kp2 - kp0
        dir_v /= np.linalg.norm(dir_v)

        dir_v_r = dir_v @ self.R90.T
        return np.float32([kp2, kp2+dir_v*dist, kp2 + dir_v_r*dist])

    @staticmethod
    def _triangle_to_bbox(source):
        # plain old vector arithmetics
        bbox = np.c_[
            [source[2] - source[0] + source[1]],
            [source[1] + source[0] - source[2]],
            [3 * source[0] - source[1] - source[2]],
            [source[2] - source[1] + source[0]],
        ].reshape(-1,2)
        return bbox

    @staticmethod
    def _im_normalize(img):
        #  return np.ascontiguousarray(2 * ((img / 255) - 0.5).astype('float32'))
        return np.ascontiguousarray(((img - 127.5) /  127.5).astype('float32'))

    @staticmethod
    def _sigm(x):
        return 1 / (1 + np.exp(-x)  + 0.00000000001 )

    @staticmethod
    def _pad1(x):
        return np.pad(x, ((0,0),(0,1)), constant_values=1, mode='constant')


    def predict_joints(self, img_norms, num_imgs):
        in_shape = (num_imgs,256,256,3)
        self.interp_joint.resize_tensor_input(self.interp_joint.get_input_details()[0]['index'], in_shape)
        self.interp_joint.allocate_tensors()
        self.interp_joint.set_tensor(self.interp_joint.get_input_details()[0]['index'], img_norms.reshape(in_shape))
        self.interp_joint.invoke()

        joints = self.interp_joint.get_tensor(self.out_idx_joint)
        return joints.reshape(num_imgs,-1,2)

    def detect_hand(self, img_norm,input_threshold):
        # assert -1 <= img_norm.min() and img_norm.max() <= 1,\
        "img_norm should be in range [-1, 1]"
        # assert img_norm.shape == (256, 256, 3),\
        "img_norm shape must be (256, 256, 3)"

        # predict hand location and 7 initial landmarks
        self.interp_palm.set_tensor(self.in_idx, img_norm[None])
        self.interp_palm.invoke()

        """
        out_reg shape is [number of anchors, 18]
        Second dimension 0 - 4 are bounding box offset, width and height: dx, dy, w ,h
        Second dimension 4 - 18 are 7 hand keypoint x and y coordinates: x1,y1,x2,y2,...x7,y7
        """
        out_reg = self.interp_palm.get_tensor(self.out_reg_idx)[0]
        """
        out_clf shape is [number of anchors]
        it is the classification score if there is a hand for each anchor box
        """
        out_clf = self.interp_palm.get_tensor(self.out_clf_idx)[0,:,0]

        # finding the best prediction
        probabilities = self._sigm(out_clf)
        detecion_mask = probabilities > input_threshold
        candidate_detect = out_reg[detecion_mask]
        candidate_anchors = self.anchors[detecion_mask]
        probabilities = probabilities[detecion_mask]

        if candidate_detect.shape[0] == 0:
            print("No hands found")
            return None, None, None

        # Pick the best bounding box with non maximum suppression
        # the boxes must be moved by the corresponding anchor first
        moved_candidate_detect = candidate_detect.copy()
        moved_candidate_detect[:, :2] = candidate_detect[:, :2] + (candidate_anchors[:, :2] * 256)
        box_ids = non_max_suppression_fast(moved_candidate_detect[:, :4], probabilities)

        # Pick the first detected hand. Could be adapted for multi hand recognition
        # box_ids = box_ids[0]

        for box_id in box_ids:
            # bounding box offsets, width and height
            dx,dy,w,h = candidate_detect[box_id, :4]
            center_wo_offst = candidate_anchors[box_id,:2] * 256

            # 7 initial keypoints
            keypoints = center_wo_offst + candidate_detect[box_id,4:].reshape(-1,2)
            side = max(w,h) * self.box_enlarge

            # now we need to move and rotate the detected hand for it to occupy a
            # 256x256 square
            # line from wrist keypoint to middle finger keypoint
            # should point straight up
            # TODO: replace triangle with the bbox directly
            source = self._get_triangle(keypoints[0], keypoints[2], side)
            source -= (keypoints[0] - keypoints[2]) * self.box_shift
            # debug_info = {
            #     "detection_candidates": candidate_detect,
            #     "anchor_candidates": candidate_anchors,
            #     "selected_box_id": box_ids,
            # }
            
            self.source_multi.append(source)

        return self.source_multi

    def preprocess_img(self, img):
        # fit the image into a 256x256 square
        shape = np.r_[img.shape]
        pad = (shape.max() - shape[:2]).astype('uint32') // 1
        img_pad = np.pad(
            img,
            ((pad[0],pad[0]), (pad[1],pad[1]), (0,0)),
            mode='constant')
        # img_small = cv2.resize(img, (256, 256))
        img_small = np.ascontiguousarray(img)

        img_norm = self._im_normalize(img_small)
        return img_pad, img_norm, pad


    def __call__(self, img, input_threshold):
        img_pad, img_norm, pad = self.preprocess_img(img)

        sources = self.detect_hand(img_norm, input_threshold)
        
        for source in sources:
            if source is None:
                return None, None

        # calculating transformation from img_pad coords
        # to img_landmark coords (cropped hand image)
        scale = max(img.shape) / 256
        
        num_imgs = 0
        for source in sources:
            Mtr = cv2.getAffineTransform(source * scale,self._target_triangle)
            img_landmark = cv2.warpAffine(self._im_normalize(img_pad), Mtr, (256,256))
        
            self.Mtrs.append(Mtr)
            self.img_landmarks.extend(img_landmark)
            num_imgs = num_imgs + 1
                
        joints = self.predict_joints(np.array(self.img_landmarks),num_imgs)

        for (Mtr, joints_) in zip(self.Mtrs, joints):
            # adding the [0,0,1] row to make the matrix square
            Mtr = self._pad1(Mtr.T).T
            Mtr[2,:2] = 0

            Minv = np.linalg.inv(Mtr)

            # projecting keypoints back into original image coordinate space
            kp_orig = (self._pad1(joints_) @ Minv.T)[:,:2]
            # box_orig = (self._target_box @ Minv.T)[:,:2]
            kp_orig -= pad[::-1]
            # box_orig -= pad[::-1]
            self.kp_orig_list.append(kp_orig)
        return self.kp_orig_list
