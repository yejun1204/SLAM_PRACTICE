"""
Tracking - Camera pose estimation for each frame
"""

import cv2
import numpy as np
import threading
from enum import Enum


class TrackingState(Enum):
    """Tracking state machine"""
    NOT_INITIALIZED = 0
    OK = 1
    LOST = 2


class Tracking:
    """
    Tracking thread - Estimates camera pose for each frame

    Responsibilities:
    - Feature extraction
    - Initial pose estimation (using motion model or reference keyframe)
    - Track local map (match with local MapPoints)
    - Decide when to insert new KeyFrame
    """

    def __init__(self, slam_map, K, dist_coeffs, orb_params=None):
        """
        Initialize Tracking

        Args:
            slam_map: Map object
            K: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            orb_params: Dictionary of ORB parameters (optional)
        """
        self.map = slam_map
        self.K = K.copy()
        self.dist_coeffs = dist_coeffs.copy()

        # Tracking state
        self.state = TrackingState.NOT_INITIALIZED

        # Current and last frames
        self.current_frame = None
        self.last_frame = None
        self.reference_keyframe = None

        # Current pose estimate (T_cw: world to camera)
        self.current_pose = np.eye(4)

        # Velocity model (for constant velocity motion model)
        self.velocity = np.eye(4)
        self.last_pose = np.eye(4)

        # ORB detector
        if orb_params is None:
            orb_params = {'nfeatures': 2000}
        self.orb = cv2.ORB_create(**orb_params)

        # Feature matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # KeyFrame decision thresholds
        self.min_frames = 0  # Minimum frames since last KeyFrame
        self.max_frames = 30  # Maximum frames since last KeyFrame
        self.frames_since_last_kf = 0

        # Tracking quality thresholds
        self.min_tracked_points = 50  # Minimum tracked MapPoints for OK state
        self.min_inliers = 30  # Minimum inliers for pose estimation

        # Thread safety
        self.lock = threading.RLock()

    def track_frame(self, image, timestamp, frame_id):
        """
        Track a new frame

        Args:
            image: Grayscale image
            timestamp: Frame timestamp
            frame_id: Frame ID

        Returns:
            success: Boolean indicating tracking success
            pose: Estimated pose (T_cw) if successful, None otherwise
        """
        with self.lock:
            # Extract features
            keypoints, descriptors = self.orb.detectAndCompute(image, None)

            if descriptors is None or len(keypoints) < 10:
                print(f"[Tracking] Frame {frame_id}: Too few features")
                return False, None

            # Create current frame data
            self.current_frame = {
                'image': image,
                'timestamp': timestamp,
                'frame_id': frame_id,
                'keypoints': keypoints,
                'descriptors': descriptors,
                'mappoints': [None] * len(keypoints),  # Matched MapPoints
                'pose': None
            }

            # State machine
            if self.state == TrackingState.NOT_INITIALIZED:
                # Wait for initialization
                return False, None

            elif self.state == TrackingState.OK:
                # Track with motion model
                success = self._track_with_motion_model()

                if not success:
                    # Fall back to reference KeyFrame
                    success = self._track_reference_keyframe()

                if success:
                    # Refine by tracking local map
                    success = self._track_local_map()

                if success:
                    self.current_frame['pose'] = self.current_pose.copy()
                    self._update_motion_model()
                    self.frames_since_last_kf += 1
                    self.last_frame = self.current_frame
                    return True, self.current_pose.copy()
                else:
                    self.state = TrackingState.LOST
                    print(f"[Tracking] Frame {frame_id}: Tracking lost")
                    return False, None

            elif self.state == TrackingState.LOST:
                # Try to relocalize
                success = self._relocalize()

                if success:
                    self.state = TrackingState.OK
                    self.current_frame['pose'] = self.current_pose.copy()
                    self.last_frame = self.current_frame
                    return True, self.current_pose.copy()
                else:
                    return False, None

    def _track_with_motion_model(self):
        """
        Track using constant velocity motion model

        Returns:
            success: Boolean
        """
        if self.last_frame is None:
            return False

        # Predict current pose using velocity
        predicted_pose = self.velocity @ self.last_pose

        # Match features between last frame and current frame
        matches = self.matcher.knnMatch(
            self.last_frame['descriptors'],
            self.current_frame['descriptors'],
            k=2
        )

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 20:
            return False

        # Get 3D-2D correspondences
        pts_3d = []
        pts_2d = []
        matched_indices = []  # current frame의 keypoint 인덱스 추적

        for m in good_matches:
            last_idx = m.queryIdx
            curr_idx = m.trainIdx

            # Get MapPoint from last frame
            mappoint = self.last_frame['mappoints'][last_idx]
            if mappoint is not None and not mappoint.is_bad:
                pts_3d.append(mappoint.get_position().ravel())
                pts_2d.append(self.current_frame['keypoints'][curr_idx].pt)
                matched_indices.append(curr_idx)

                # Store matched MapPoint
                self.current_frame['mappoints'][curr_idx] = mappoint

        if len(pts_3d) < 10:
            return False

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Use predicted pose as initial guess
        rvec, _ = cv2.Rodrigues(predicted_pose[:3, :3])
        tvec = predicted_pose[:3, 3:4]

        # Solve PnP with RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, self.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            iterationsCount=100,
            reprojectionError=2.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            return False

        # Update pose
        R, _ = cv2.Rodrigues(rvec)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3:4] = tvec

        # Clear MapPoints that are outliers
        inlier_set = set(inliers.ravel())
        for i, curr_idx in enumerate(matched_indices):
            if i not in inlier_set:
                self.current_frame['mappoints'][curr_idx] = None

        return True

    def _track_reference_keyframe(self):
        """
        Track by matching with reference KeyFrame

        Returns:
            success: Boolean
        """
        if self.reference_keyframe is None:
            return False

        # Match with reference KeyFrame
        matches = self.matcher.knnMatch(
            self.reference_keyframe['descriptors'],
            self.current_frame['descriptors'],
            k=2
        )

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 30:
            return False

        # Get 3D-2D correspondences
        pts_3d = []
        pts_2d = []
        matched_indices = []  # current frame의 keypoint 인덱스 추적

        for m in good_matches:
            ref_idx = m.queryIdx
            curr_idx = m.trainIdx

            # Get MapPoint from reference KeyFrame
            mappoint = self.reference_keyframe['mappoints'][ref_idx]
            if mappoint is not None and not mappoint.is_bad:
                pts_3d.append(mappoint.get_position().ravel())
                pts_2d.append(self.current_frame['keypoints'][curr_idx].pt)
                matched_indices.append(curr_idx)

                # Store matched MapPoint
                self.current_frame['mappoints'][curr_idx] = mappoint

        if len(pts_3d) < 15:
            return False

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Solve PnP
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            pts_3d, pts_2d, self.K, self.dist_coeffs,
            iterationsCount=100,
            reprojectionError=3.0,
            confidence=0.99,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success or inliers is None or len(inliers) < self.min_inliers:
            return False

        # Update pose
        R, _ = cv2.Rodrigues(rvec)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3:4] = tvec

        # Clear outliers
        inlier_set = set(inliers.ravel())
        for i, curr_idx in enumerate(matched_indices):
            if i not in inlier_set:
                self.current_frame['mappoints'][curr_idx] = None

        return True

    def _track_local_map(self):
        """
        Track local map to refine pose and find more matches

        Returns:
            success: Boolean
        """
        # Get local MapPoints (visible in current and nearby KeyFrames)
        local_mappoints = self._get_local_mappoints()

        if len(local_mappoints) < 20:
            return True  # Accept current tracking

        # Project local MapPoints and search for matches
        new_matches = 0
        already_matched = set(id(mp) for mp in self.current_frame['mappoints'] if mp is not None)

        for mappoint in local_mappoints:
            if mappoint.is_bad:
                continue

            # Skip if already matched
            if id(mappoint) in already_matched:
                mappoint.increase_visible()
                mappoint.increase_found()
                continue

            # Project to current frame
            pt_3d = mappoint.get_position().ravel()
            pt_2d, is_visible = self._project_point(pt_3d, self.current_pose)

            if not is_visible:
                continue

            mappoint.increase_visible()

            # Search in radius for match
            idx = self._search_in_radius(
                pt_2d, mappoint.descriptor,
                self.current_frame['keypoints'],
                self.current_frame['descriptors'],
                radius=15.0
            )

            if idx is not None:
                self.current_frame['mappoints'][idx] = mappoint
                mappoint.increase_found()
                new_matches += 1

        # Optimize pose with all matches
        pts_3d = []
        pts_2d = []

        for i, mappoint in enumerate(self.current_frame['mappoints']):
            if mappoint is not None and not mappoint.is_bad:
                pts_3d.append(mappoint.get_position().ravel())
                pts_2d.append(self.current_frame['keypoints'][i].pt)

        if len(pts_3d) < self.min_tracked_points:
            return False

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Refine pose
        rvec, _ = cv2.Rodrigues(self.current_pose[:3, :3])
        tvec = self.current_pose[:3, 3:4]

        success, rvec, tvec = cv2.solvePnP(
            pts_3d, pts_2d, self.K, self.dist_coeffs,
            rvec, tvec,
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            R, _ = cv2.Rodrigues(rvec)
            self.current_pose[:3, :3] = R
            self.current_pose[:3, 3:4] = tvec

        return len(pts_3d) >= self.min_tracked_points

    def _relocalize(self):
        """
        Try to relocalize when tracking is lost

        Returns:
            success: Boolean
        """
        # For now, use reference KeyFrame (can be improved with place recognition)
        return self._track_reference_keyframe()

    def _get_local_mappoints(self):
        """
        Get local MapPoints visible in current and nearby KeyFrames

        Returns:
            List of MapPoints
        """
        # Simplified: get all MapPoints from map
        # TODO: Filter to local area only
        return self.map.get_valid_mappoints()

    def _project_point(self, pt_3d, pose):
        """
        Project 3D point to image plane

        Args:
            pt_3d: 3D point (3,)
            pose: Camera pose T_cw (4x4)

        Returns:
            pt_2d: Projected 2D point (2,)
            is_visible: Boolean indicating if point is visible
        """
        # Transform to camera frame
        pt_cam = pose[:3, :3] @ pt_3d + pose[:3, 3]

        # Check if in front of camera
        if pt_cam[2] <= 0:
            return None, False

        # Project to image
        pt_2d = self.K @ pt_cam
        pt_2d = pt_2d[:2] / pt_2d[2]

        # Check if in image bounds
        h, w = self.current_frame['image'].shape[:2]
        if pt_2d[0] < 0 or pt_2d[0] >= w or pt_2d[1] < 0 or pt_2d[1] >= h:
            return None, False

        return pt_2d, True

    def _search_in_radius(self, pt_2d, descriptor, keypoints, descriptors, radius):
        """
        Search for matching feature in radius around projected point

        Args:
            pt_2d: Projected 2D point (2,)
            descriptor: MapPoint descriptor
            keypoints: Frame keypoints
            descriptors: Frame descriptors
            radius: Search radius in pixels

        Returns:
            idx: Index of best match, or None if no match found
        """
        best_dist = 256  # Max Hamming distance for ORB
        best_idx = None

        for i, kp in enumerate(keypoints):
            # Check if in radius
            dist = np.linalg.norm(np.array(kp.pt) - pt_2d)
            if dist > radius:
                continue

            # Compute descriptor distance
            desc_dist = cv2.norm(descriptor, descriptors[i], cv2.NORM_HAMMING)

            if desc_dist < best_dist:
                best_dist = desc_dist
                best_idx = i

        # Threshold for good match
        if best_dist < 50:  # ORB threshold
            return best_idx

        return None

    def _update_motion_model(self):
        """Update velocity for motion model"""
        if self.last_frame is not None and self.last_frame['pose'] is not None:
            self.last_pose = self.last_frame['pose'].copy()
            # velocity = T_curr @ T_last^{-1}
            T_last_inv = np.linalg.inv(self.last_pose)
            self.velocity = self.current_pose @ T_last_inv # T_cc-1
        else:
            self.velocity = np.eye(4)

    def need_new_keyframe(self):
        """
        Decide if a new KeyFrame should be inserted

        Returns:
            Boolean indicating if new KeyFrame is needed
        """
        with self.lock:
            if self.state != TrackingState.OK:
                return False

            # Check minimum frames
            if self.frames_since_last_kf < self.min_frames:
                return False

            # Force KeyFrame if max frames exceeded
            if self.frames_since_last_kf >= self.max_frames:
                return True

            # Count tracked MapPoints
            n_tracked = sum(1 for mp in self.current_frame['mappoints'] if mp is not None)

            # Need KeyFrame if tracking quality is low
            if n_tracked < self.min_tracked_points * 1.5:
                return True

            return False

    def create_keyframe(self):
        """
        Create KeyFrame from current frame

        Returns:
            KeyFrame data dictionary
        """
        with self.lock:
            if self.current_frame is None:
                return None

            # Reset frame counter
            self.frames_since_last_kf = 0

            # Return KeyFrame data
            kf_data = self.current_frame.copy()
            kf_data['pose'] = self.current_pose.copy()

            return kf_data

    def set_initialized(self, reference_kf, current_pose):
        """
        Set tracking to initialized state

        Args:
            reference_kf: Reference KeyFrame data
            current_pose: Initial pose (T_cw)
        """
        with self.lock:
            self.state = TrackingState.OK
            self.reference_keyframe = reference_kf
            self.current_pose = current_pose.copy()
            self.last_pose = current_pose.copy()
            self.velocity = np.eye(4)
            self.frames_since_last_kf = 0
            print("[Tracking] Initialized!")

    def update_reference_keyframe(self, keyframe):
        """
        Update reference KeyFrame used for tracking fallback.

        Args:
            keyframe: KeyFrame object (newly created by LocalMapping)
        """
        with self.lock:
            self.reference_keyframe = {
                'frame_id': keyframe.frame_id,
                'timestamp': keyframe.timestamp,
                'keypoints': keyframe.keypoints,
                'descriptors': keyframe.descriptors,
                'image': keyframe.image,
                'pose': keyframe.get_pose(),
                'mappoints': keyframe.get_mappoints()
            }

    def get_state(self):
        """Get current tracking state"""
        with self.lock:
            return self.state

    def get_tracked_mappoints_count(self):
        """Get number of tracked MapPoints in current frame"""
        with self.lock:
            if self.current_frame is None:
                return 0
            return sum(1 for mp in self.current_frame['mappoints'] if mp is not None)
