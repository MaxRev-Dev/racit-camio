import time

import numpy as np
import cv2 as cv
import mediapipe as mp
from scipy import stats


class PoseDetectorMP:
    def __init__(self, model, max_hands=1, input_width=640, verbose=False, draw_landmarks=True):
        self.model = model
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=0,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.5,
        )
        self.input_width = input_width
        self.verbose = verbose
        self.draw_landmarks = draw_landmarks
        self._coors_buffer = np.zeros((4, 3), dtype=np.float32)
        self._last_log_time = 0.0
        self._last_hand_count = 0
        self._last_status = None
        self._log_prefix = "[PoseDetector]"
        self._input_shape_logged = False
        self._projection_warning_logged = False
        if self.verbose:
            width_msg = f"{self.input_width}px" if self.input_width else "full width"
            self._log(f"Max hands: {max_hands}, inference width: {width_msg}")

    def detect(self, frame, H, _):
        frame_height, frame_width = frame.shape[:2]
        scaled_frame = frame
        if self.input_width and frame_width > self.input_width:
            scale = self.input_width / float(frame_width)
            new_size = (int(frame_width * scale), max(1, int(frame_height * scale)))
            scaled_frame = cv.resize(frame, new_size, interpolation=cv.INTER_AREA)
            if self.verbose and not self._input_shape_logged:
                self._log(f"Input frame {frame_width}x{frame_height}, scaled to {scaled_frame.shape[1]}x{scaled_frame.shape[0]}")
        rgb_frame = cv.cvtColor(scaled_frame, cv.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        if self.verbose and not self._input_shape_logged:
            self._log("Running MediaPipe Hands inference...")
            self._input_shape_logged = True

        start_time = time.time()
        results = self.hands.process(rgb_frame)
        inference_time = time.time() - start_time
        hand_count = len(results.multi_hand_landmarks) if results and results.multi_hand_landmarks else 0
        self._maybe_log(hand_count, inference_time)

        if not hand_count:
            if self.verbose:
                self._log("No hands detected in this frame.")
            return None, None, frame

        handedness = []
        index_pos = None
        movement_status = "moving"

        for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            handedness.append(self._get_handedness_label(results, h_idx))

            ratio_index = self._finger_ratio(hand_landmarks, (5, 6, 7, 8))
            ratio_middle = self._finger_ratio(hand_landmarks, (9, 10, 11, 12))
            ratio_ring = self._finger_ratio(hand_landmarks, (13, 14, 15, 16))
            ratio_little = self._finger_ratio(hand_landmarks, (17, 18, 19, 20))

            position = self._map_landmark_to_model(hand_landmarks, frame_width, frame_height, H)
            if position is None:
                continue

            if index_pos is None:
                index_pos = position
                if self.verbose:
                    self._log(f"Hand {h_idx} ({handedness[-1]}): ratios idx={ratio_index:.2f}, mid={ratio_middle:.2f}, ring={ratio_ring:.2f}, little={ratio_little:.2f}")
                    self._log(f"Hand {h_idx} projected position: ({position[0]:.1f}, {position[1]:.1f})")

            if (ratio_index > 0.7) and (ratio_middle < 0.95) and (ratio_ring < 0.95) and (ratio_little < 0.95):
                if movement_status != "pointing" or (len(handedness) > 1 and handedness[-1] == handedness[0]):
                    index_pos = position
                    movement_status = "pointing"
                else:
                    index_pos = np.append(index_pos, position)
                    movement_status = "too_many"
            elif movement_status != "pointing":
                movement_status = "moving"

            if self.draw_landmarks:
                self._draw_simple_overlay(frame, hand_landmarks)

        if self.verbose and movement_status != self._last_status:
            self._log(f"Status: {movement_status}")
            self._last_status = movement_status

        return index_pos, movement_status, frame

    def _finger_ratio(self, hand_landmarks, indices):
        for offset, landmark_idx in enumerate(indices):
            lm = hand_landmarks.landmark[landmark_idx]
            self._coors_buffer[offset, 0] = lm.x
            self._coors_buffer[offset, 1] = lm.y
            self._coors_buffer[offset, 2] = lm.z
        return self.ratio(self._coors_buffer)

    def _map_landmark_to_model(self, hand_landmarks, frame_width, frame_height, H):
        fingertip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        pixel_x = fingertip.x * frame_width
        pixel_y = fingertip.y * frame_height
        homogeneous = np.array([pixel_x, pixel_y, 1.0], dtype=float)
        mapped = H @ homogeneous
        if mapped[2] == 0:
            if self.verbose and not self._projection_warning_logged:
                self._log("Homography projection returned zero w component; check calibration.")
                self._projection_warning_logged = True
            return None
        return np.array([mapped[0] / mapped[2], mapped[1] / mapped[2], 0.0], dtype=float)

    def _draw_simple_overlay(self, frame, hand_landmarks):
        if not self.draw_landmarks:
            return
        h, w = frame.shape[:2]
        idx_tip = self._landmark_to_pixel(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_TIP, w, h)
        idx_mcp = self._landmark_to_pixel(hand_landmarks, self.mp_hands.HandLandmark.INDEX_FINGER_MCP, w, h)
        wrist = self._landmark_to_pixel(hand_landmarks, self.mp_hands.HandLandmark.WRIST, w, h)
        cv.line(frame, wrist, idx_mcp, (0, 128, 255), 2)
        cv.line(frame, idx_mcp, idx_tip, (0, 255, 0), 2)
        cv.circle(frame, idx_tip, 6, (0, 200, 255), -1)

    def _landmark_to_pixel(self, hand_landmarks, enum_value, width, height):
        landmark = hand_landmarks.landmark[enum_value]
        return int(landmark.x * width), int(landmark.y * height)

    def _get_handedness_label(self, results, index):
        if results.multi_handedness and len(results.multi_handedness) > index:
            classification = results.multi_handedness[index].classification
            if classification:
                return classification[0].label
        return "Unknown"

    def _maybe_log(self, hand_count, inference_time):
        if not self.verbose:
            return
        now = time.time()
        if hand_count != self._last_hand_count or (now - self._last_log_time) >= 1.0:
            self._last_log_time = now
            self._last_hand_count = hand_count
            self._log(f"hands={hand_count} inference={inference_time * 1000:.1f}ms")

    def _log(self, message):
        if self.verbose:
            print(f"{self._log_prefix} {message}")


    def ratio(self, coors):  # ratio is 1 if points are collinear, lower otherwise (minimum is 0)
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])

        return d / (a + b + c)

class InteractionPolicyMP:
    def __init__(self, model):
        self.model = model
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)
        self.ZONE_FILTER_SIZE = 10
        self.Z_THRESHOLD = 2.0
        self.zone_filter = -1 * np.ones(self.ZONE_FILTER_SIZE, dtype=int)
        self.zone_filter_cnt = 0

    # Sergio: we are currently returning the zone id also when the ring buffer is not full. Is this the desired behavior?
    # the impact is clearly minor, but conceptually I am not convinced that this is the right behavior.
    # Sergio (2): I have a concern about this function, I will discuss it in an email.
    def push_gesture(self, position):
        zone_color = self.get_zone(position, self.image_map_color, self.model['pixels_per_cm'])
        self.zone_filter[self.zone_filter_cnt] = self.get_dict_idx_from_color(zone_color)
        self.zone_filter_cnt = (self.zone_filter_cnt + 1) % self.ZONE_FILTER_SIZE
        zone = stats.mode(self.zone_filter).mode
        if isinstance(zone, np.ndarray):
            zone = zone[0]
        if np.abs(position[2]) < self.Z_THRESHOLD:
            return zone
        else:
            return -1


class SIFTModelDetectorMP:
    def __init__(self, model):
        self.model = model
        # Load the template image
        img_object = cv.imread(
            model["template_image"], cv.IMREAD_GRAYSCALE
        )
        # print(img_object)
        # new_image = cv.resize(img_object, (640, 480))
        # cv.imshow('image', new_image)
        # cv.waitKey(0)

        # Detect SIFT keypoints
        self.detector = cv.SIFT_create()
        self.keypoints_obj, self.descriptors_obj = self.detector.detectAndCompute(
            img_object, mask=None
        )
        
        self.requires_homography = True
        self.H = None
        self.MIN_INLIER_COUNT = 15

    def detect(self, frame):
        # If we have already computed the coordinate transform then simply return it
        if not self.requires_homography:
            return True, self.H, None
        keypoints_scene, descriptors_scene = self.detector.detectAndCompute(frame, None)
        # print(keypoints_scene, descriptors_scene)
        matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
        knn_matches = matcher.knnMatch(self.descriptors_obj, descriptors_scene, 2)

        # Only keep uniquely good matches
        RATIO_THRESH = 0.75
        good_matches = []
        for m, n in knn_matches:
            if m.distance < RATIO_THRESH * n.distance:
                good_matches.append(m)
        print("There were {} good matches".format(len(good_matches)))
        # -- Localize the object
        if len(good_matches) < 4:
            return False, None, None
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        self.scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            obj[i, 0] = self.keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_obj[good_matches[i].queryIdx].pt[1]
            self.scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            self.scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]
        # Compute homography and find inliers
        H, self.mask_out = cv.findHomography(
            self.scene, obj, cv.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )
        total = sum([int(i) for i in self.mask_out])
        obj_in = np.empty((total,2),dtype=np.float32)
        scene_in = np.empty((total,2),dtype=np.float32)
        index = 0
        for i in range(len(self.mask_out)):
            if self.mask_out[i]:
                obj_in[index,:] = obj[i,:]
                scene_in[index,:] = self.scene[i,:]
                index += 1
        scene_out = np.squeeze(cv.perspectiveTransform(scene_in.reshape(-1,1,2), H))
        biggest_distance = 0
        sum_distance = 0
        for i in range(len(scene_out)):
            dist = cv.norm(obj_in[i,:],scene_out[i,:],cv.NORM_L2)
            sum_distance += dist
            if dist > biggest_distance:
                biggest_distance = dist
        ave_dist = sum_distance/total
        print(f'Inlier count: {total}. Biggest distance: {biggest_distance}. Average distance: {ave_dist}.')
        if total > self.MIN_INLIER_COUNT:
            self.H = H
            self.requires_homography = False
            return True, H, None
        elif self.H is not None:
            return True, self.H, None
        else:
            return False, None, None
