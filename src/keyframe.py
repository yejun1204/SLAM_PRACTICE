"""
KeyFrame class - Represents a keyframe in the map
"""

import numpy as np
import cv2
import threading


class KeyFrame:
    """
    KeyFrame - A special frame selected for mapping
    Contains image features, pose, and connections to MapPoints
    """

    # Class variable for ID generation
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, frame_id, timestamp, pose, K, dist_coeffs,
                 keypoints, descriptors, image=None):
        """
        Initialize a KeyFrame

        Args:
            frame_id: Original frame ID from the dataset
            timestamp: Timestamp
            pose: Camera pose T_cw (4x4) - world to camera transformation
            K: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            keypoints: List of cv2.KeyPoint
            descriptors: ORB descriptors (Nx32 uint8 array)
            image: Original grayscale image (optional, for visualization)
        """
        # Generate unique KeyFrame ID
        with KeyFrame._id_lock:
            self.id = KeyFrame._next_id
            KeyFrame._next_id += 1

        self.frame_id = frame_id
        self.timestamp = timestamp

        # Camera pose: T_cw (world to camera transformation)
        # T_cw = [R_cw | t_cw] where:
        #   R_cw: rotation from world to camera frame
        #   t_cw: world origin position in camera frame
        self.T_cw = np.array(pose, dtype=np.float64)

        # Camera parameters
        self.K = K.copy()
        self.dist_coeffs = dist_coeffs.copy()

        # Extract camera center (camera position in world coordinates)
        # Since p_cam = R_cw @ p_world + t_cw, and for camera center p_cam = 0:
        # 0 = R_cw @ C + t_cw  =>  C = -R_cw^T @ t_cw
        R_cw = self.T_cw[:3, :3]
        t_cw = self.T_cw[:3, 3:4]
        self.camera_center = -R_cw.T @ t_cw

        # Features
        self.keypoints = keypoints  # List of cv2.KeyPoint
        self.descriptors = descriptors  # Nx32 uint8 array
        self.n_keypoints = len(keypoints)

        # MapPoints associated with each keypoint
        # mappoints[i] is the MapPoint corresponding to keypoints[i], or None
        self.mappoints = [None] * self.n_keypoints

        # Image (optional, for visualization)
        self.image = image

        # Connected KeyFrames with weights (covisibility graph)
        # {keyframe: weight} where weight = number of shared MapPoints
        self.connected_keyframes = {}

        # Bad flag
        self.is_bad = False

        # Thread safety
        self.lock = threading.Lock()

        # Undistort keypoints once
        self.keypoints_undistorted = self._undistort_keypoints()

    def _undistort_keypoints(self):
        """
        Undistort all keypoints

        Returns:
            Array of undistorted keypoint coordinates (Nx2)
        """
        if len(self.keypoints) == 0:
            return np.array([])

        # Extract keypoint coordinates
        pts = np.array([kp.pt for kp in self.keypoints], dtype=np.float32)

        # Undistort
        pts_undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
            None,
            self.K
        ).reshape(-1, 2)

        return pts_undist

    def get_pose(self):
        """Get camera pose T_cw (thread-safe)"""
        with self.lock:
            return self.T_cw.copy()

    def set_pose(self, pose):
        """Set camera pose T_cw (thread-safe)"""
        with self.lock:
            self.T_cw = np.array(pose, dtype=np.float64)

            # Update camera center
            R_cw = self.T_cw[:3, :3]
            t_cw = self.T_cw[:3, 3:4]
            self.camera_center = -R_cw.T @ t_cw

    def get_camera_center(self):
        """Get camera center in world coordinates (thread-safe)"""
        with self.lock:
            return self.camera_center.copy()

    def get_rotation(self):
        """Get rotation matrix R_cw"""
        with self.lock:
            return self.T_cw[:3, :3].copy()

    def get_translation(self):
        """Get translation vector t_cw"""
        with self.lock:
            return self.T_cw[:3, 3:4].copy()

    def add_mappoint(self, mappoint, keypoint_idx):
        """
        Associate a MapPoint with a keypoint

        Args:
            mappoint: MapPoint object
            keypoint_idx: Index of the keypoint
        """
        with self.lock:
            if keypoint_idx < self.n_keypoints:
                self.mappoints[keypoint_idx] = mappoint

    def remove_mappoint(self, keypoint_idx):
        """Remove MapPoint association from a keypoint"""
        with self.lock:
            if keypoint_idx < self.n_keypoints:
                self.mappoints[keypoint_idx] = None

    def get_mappoints(self):
        """Get all MapPoints (thread-safe copy)"""
        with self.lock:
            return self.mappoints.copy()

    def get_valid_mappoints(self):
        """
        Get list of valid (non-None, non-bad) MapPoints

        Returns:
            List of MapPoint objects
        """
        with self.lock:
            valid_mps = []
            for mp in self.mappoints:
                if mp is not None and not mp.is_bad_point():
                    valid_mps.append(mp)
            return valid_mps

    def num_mappoints(self):
        """Get number of valid MapPoints"""
        return len(self.get_valid_mappoints())

    def update_connections(self, min_shared_points=15):
        """
        Update covisibility graph connections

        Args:
            min_shared_points: Minimum number of shared MapPoints to form a connection
        """
        with self.lock:
            # Count shared MapPoints with other KeyFrames
            shared_counts = {}

            for mp in self.mappoints:
                if mp is None or mp.is_bad_point():
                    continue

                observations = mp.get_observations()
                for kf in observations.keys():
                    if kf == self:
                        continue

                    if kf not in shared_counts:
                        shared_counts[kf] = 0
                    shared_counts[kf] += 1

            # Keep only connections with enough shared points
            self.connected_keyframes = {
                kf: count for kf, count in shared_counts.items()
                if count >= min_shared_points
            }

    def get_connected_keyframes(self):
        """Get connected KeyFrames sorted by weight (descending)"""
        with self.lock:
            # Sort by weight (number of shared MapPoints)
            sorted_kfs = sorted(
                self.connected_keyframes.items(),
                key=lambda x: x[1],
                reverse=True
            )
            return [kf for kf, weight in sorted_kfs]

    def get_best_covisible_keyframes(self, n=10):
        """Get N best covisible KeyFrames"""
        connected = self.get_connected_keyframes()
        return connected[:n]

    def set_bad_flag(self):
        """Mark this KeyFrame as bad"""
        with self.lock:
            self.is_bad = True

            # Remove MapPoint associations
            for i in range(self.n_keypoints):
                if self.mappoints[i] is not None:
                    self.mappoints[i].remove_observation(self)
                    self.mappoints[i] = None

    def is_bad_keyframe(self):
        """Check if this KeyFrame is bad"""
        with self.lock:
            return self.is_bad

    def compute_bow_vector(self):
        """
        Compute Bag-of-Words vector (placeholder for DBoW2 integration)
        TODO: Implement with DBoW vocabulary
        """
        pass

    def __repr__(self):
        """String representation"""
        pos = self.camera_center.ravel()
        n_mp = self.num_mappoints()
        n_conn = len(self.connected_keyframes)
        return (f"KeyFrame(id={self.id}, frame_id={self.frame_id}, "
                f"pos=[{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}], "
                f"mappoints={n_mp}/{self.n_keypoints}, connections={n_conn})")
