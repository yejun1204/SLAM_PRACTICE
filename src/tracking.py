"""
Tracking - Camera pose estimation for each frame
"""

import cv2
import numpy as np
import threading
import gtsam
from enum import Enum


def grid_distribute(keypoints, descriptors, img_shape, n_keep,
                    grid_rows=8, grid_cols=12):
    """
    Distribute keypoints uniformly across image grid.
    In each cell, keep keypoints with highest response score.
    """
    h, w = img_shape[:2]
    cell_h = h / grid_rows
    cell_w = w / grid_cols
    n_per_cell = max(1, n_keep // (grid_rows * grid_cols))

    selected = []
    for r in range(grid_rows):
        for c in range(grid_cols):
            x0, x1 = c * cell_w, (c + 1) * cell_w
            y0, y1 = r * cell_h, (r + 1) * cell_h
            cell_kps = [(i, kp) for i, kp in enumerate(keypoints)
                        if x0 <= kp.pt[0] < x1 and y0 <= kp.pt[1] < y1]
            cell_kps.sort(key=lambda x: x[1].response, reverse=True)
            selected.extend(cell_kps[:n_per_cell])

    if len(selected) == 0:
        return keypoints, descriptors

    indices = [i for i, _ in selected]
    kps_out = [keypoints[i] for i in indices]
    desc_out = descriptors[indices]
    return kps_out, desc_out


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

        # Last reference KF matches for visualization (inlier DMatch list)
        self.last_ref_inlier_matches = []

        # Current pose estimate (T_cw: world to camera)
        self.current_pose = np.eye(4)

        # Velocity model (for constant velocity motion model)
        self.velocity = np.eye(4)
        self.last_pose = np.eye(4)

        # ORB detector (dense: detect more, then grid-distribute)
        if orb_params is None:
            orb_params = {'nfeatures': 2000}
        self.n_features = orb_params.get('nfeatures', 2000)
        dense_params = dict(orb_params)
        dense_params['nfeatures'] = self.n_features * 3
        self.orb = cv2.ORB_create(**dense_params)
        self.grid_rows = 8
        self.grid_cols = 12

        # Feature matching
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # KeyFrame decision thresholds
        self.min_frames = 3   # Minimum frames since last KeyFrame
        self.max_frames = 30  # Maximum frames since last KeyFrame
        self.frames_since_last_kf = 0

        # Tracking quality thresholds
        self.min_tracked_points = 50  # Minimum tracked MapPoints for OK state
        self.min_inliers = 30  # Minimum inliers for pose estimation
        self.ref_kf_tracked_ratio = 0.9  # KF needed if tracked < ref_kf * ratio

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
            # Extract features with grid-based distribution
            keypoints, descriptors = self.orb.detectAndCompute(image, None)
            if descriptors is not None and len(keypoints) > self.n_features:
                keypoints, descriptors = self._grid_distribute(
                    keypoints, descriptors, image.shape, self.n_features,
                    self.grid_rows, self.grid_cols)

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
                    self.current_frame['mappoints'] = [None] * len(self.current_frame['keypoints'])
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
                    # Refine relocalized pose with local map
                    self._track_local_map()
                    self.state = TrackingState.OK
                    self.current_frame['pose'] = self.current_pose.copy()
                    self._update_motion_model()
                    self.last_frame = self.current_frame
                    return True, self.current_pose.copy()
                else:
                    return False, None

    def _track_with_motion_model(self):
        """
        Track using constant velocity motion model (ORB-SLAM3 style).
        Projects last frame's map points into current frame using predicted pose,
        then searches for matches in a small radius around each projected position.

        Returns:
            success: Boolean
        """
        if self.last_frame is None:
            return False

        # Reject unreliable velocity (large translation jump)
        velocity_translation = np.linalg.norm(self.velocity[:3, 3])
        if velocity_translation > 1.0:
            return False

        # Predict current pose using velocity
        predicted_pose = self.velocity @ self.last_pose

        pts_3d = []
        pts_2d = []
        matched_indices = []
        matched_mask = list(self.current_frame['mappoints'])

        for last_idx, mappoint in enumerate(self.last_frame['mappoints']):
            if mappoint is None or mappoint.is_bad:
                continue

            # Project map point to current frame using predicted pose
            pt_3d = mappoint.get_position().ravel()
            pt_2d_proj, is_visible = self._project_point(pt_3d, predicted_pose)
            if not is_visible:
                continue

            # Search in radius around projected position
            descriptor = self.last_frame['descriptors'][last_idx]
            curr_idx = self._search_in_radius(
                pt_2d_proj, descriptor,
                self.current_frame['keypoints'],
                self.current_frame['descriptors'],
                radius=20.0,
                matched_mask=matched_mask
            )

            if curr_idx is not None:
                pts_3d.append(pt_3d)
                pts_2d.append(self.current_frame['keypoints'][curr_idx].pt)
                matched_indices.append(curr_idx)
                matched_mask[curr_idx] = mappoint
                self.current_frame['mappoints'][curr_idx] = mappoint

        if len(pts_3d) < 10:
            return False

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        rvec, _ = cv2.Rodrigues(predicted_pose[:3, :3])
        tvec = predicted_pose[:3, 3:4]

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

        R, _ = cv2.Rodrigues(rvec)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3:4] = tvec

        # Clear MapPoints that are outliers
        inlier_set = set(inliers.ravel())
        for i, curr_idx in enumerate(matched_indices):
            if i not in inlier_set:
                self.current_frame['mappoints'][curr_idx] = None

        inlier_pts_3d = pts_3d[inliers.ravel()]
        inlier_pts_2d = pts_2d[inliers.ravel()]
        self.current_pose = self._optimize_pose_gtsam(inlier_pts_3d, inlier_pts_2d, self.current_pose)

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
                mappoint.increase_visible()
                mappoint.increase_found()

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

        R, _ = cv2.Rodrigues(rvec)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3:4] = tvec

        # Clear outliers, save inlier matches for visualization
        inlier_set = set(inliers.ravel())
        inlier_matches = []
        inlier_pts_3d, inlier_pts_2d = [], []
        for i, (m, curr_idx) in enumerate(zip(good_matches, matched_indices)):
            if i not in inlier_set:
                self.current_frame['mappoints'][curr_idx] = None
            else:
                inlier_matches.append(m)
                mp = self.current_frame['mappoints'][curr_idx]
                if mp is not None:
                    inlier_pts_3d.append(mp.get_position().ravel())
                    inlier_pts_2d.append(self.current_frame['keypoints'][curr_idx].pt)
        self.last_ref_inlier_matches = inlier_matches

        if len(inlier_pts_3d) >= self.min_inliers:
            self.current_pose = self._optimize_pose_gtsam(
                np.array(inlier_pts_3d, dtype=np.float32),
                np.array(inlier_pts_2d, dtype=np.float32),
                self.current_pose
            )

        return True

    def get_ref_match_viz_data(self):
        """Return data needed for cv2.drawMatches visualization"""
        with self.lock:
            if self.reference_keyframe is None or self.current_frame is None:
                return None
            return {
                'ref_image': self.reference_keyframe['image'],
                'ref_keypoints': self.reference_keyframe['keypoints'],
                'curr_image': self.current_frame['image'],
                'curr_keypoints': self.current_frame['keypoints'],
                'matches': list(self.last_ref_inlier_matches)
            }

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

            # Search in radius for match (skip already-matched keypoints)
            idx = self._search_in_radius(
                pt_2d, mappoint.descriptor,
                self.current_frame['keypoints'],
                self.current_frame['descriptors'],
                radius=20.0,
                matched_mask=self.current_frame['mappoints']
            )

            if idx is not None:
                self.current_frame['mappoints'][idx] = mappoint
                already_matched.add(id(mappoint))
                mappoint.increase_found()
                new_matches += 1

        # Optimize pose with all matches
        pts_3d = []
        pts_2d = []
        kp_indices = []

        for i, mappoint in enumerate(self.current_frame['mappoints']):
            if mappoint is not None and not mappoint.is_bad:
                pts_3d.append(mappoint.get_position().ravel())
                pts_2d.append(self.current_frame['keypoints'][i].pt)
                kp_indices.append(i)

        if len(pts_3d) < self.min_inliers:
            return False

        pts_3d = np.array(pts_3d, dtype=np.float32)
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Refine pose with RANSAC to reject outlier MapPoints
        rvec, _ = cv2.Rodrigues(self.current_pose[:3, :3])
        tvec = self.current_pose[:3, 3:4]

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

        R, _ = cv2.Rodrigues(rvec)
        self.current_pose[:3, :3] = R
        self.current_pose[:3, 3:4] = tvec

        # Remove outlier MapPoints from current frame
        inlier_set = set(inliers.ravel())
        for list_idx, kp_idx in enumerate(kp_indices):
            if list_idx not in inlier_set:
                self.current_frame['mappoints'][kp_idx] = None

        inlier_pts_3d = pts_3d[inliers.ravel()]
        inlier_pts_2d = pts_2d[inliers.ravel()]
        self.current_pose = self._optimize_pose_gtsam(inlier_pts_3d, inlier_pts_2d, self.current_pose)

        return len(inliers) >= self.min_inliers

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
        if self.reference_keyframe is None:
            return self.map.get_valid_mappoints()

        ref_kf = self.reference_keyframe
        if isinstance(ref_kf, dict):
            return self.map.get_valid_mappoints()

        local_kfs = [ref_kf] + ref_kf.get_best_covisible_keyframes(n=20)

        seen = set()
        local_mps = []
        for kf in local_kfs:
            for mp in kf.get_valid_mappoints():
                if id(mp) not in seen:
                    seen.add(id(mp))
                    local_mps.append(mp)

        return local_mps

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

    def _grid_distribute(self, keypoints, descriptors, img_shape, n_keep,
                         grid_rows, grid_cols):
        return grid_distribute(keypoints, descriptors, img_shape, n_keep,
                               grid_rows, grid_cols)

    def _search_in_radius(self, pt_2d, descriptor, keypoints, descriptors, radius,
                          matched_mask=None):
        """
        Search for matching feature in radius around projected point

        Args:
            pt_2d: Projected 2D point (2,)
            descriptor: MapPoint descriptor
            keypoints: Frame keypoints
            descriptors: Frame descriptors
            radius: Search radius in pixels
            matched_mask: List of MapPoint-or-None per keypoint; skip occupied slots

        Returns:
            idx: Index of best match, or None if no match found
        """
        best_dist = 256  # Max Hamming distance for ORB
        best_idx = None

        for i, kp in enumerate(keypoints):
            # Skip already-matched keypoints
            if matched_mask is not None and matched_mask[i] is not None:
                continue

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
        if best_dist < 50:  # ORB TH_LOW
            return best_idx

        return None

    def _optimize_pose_gtsam(self, pts_3d, pts_2d, initial_pose):
        """
        Pose-only optimization using gtsam (map points fixed).
        Two rounds: 1st with Huber robust kernel, 2nd with inliers only (chi2 < threshold).
        """
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        cal = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-6)
        pose_key = gtsam.symbol('x', 0)

        def to_gtsam_pose(T_cw):
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            return gtsam.Pose3(gtsam.Rot3(R_cw.T), gtsam.Point3(-R_cw.T @ t_cw))

        def to_T_cw(gtsam_pose):
            R_wc = gtsam_pose.rotation().matrix()
            t_wc = gtsam_pose.translation()
            T = np.eye(4)
            T[:3, :3] = R_wc.T
            T[:3, 3] = -R_wc.T @ t_wc
            return T

        def run_opt(pts3, pts2, pose_init, use_robust):
            graph = gtsam.NonlinearFactorGraph()
            initial = gtsam.Values()
            initial.insert(pose_key, to_gtsam_pose(pose_init))

            if use_robust:
                huber = gtsam.noiseModel.mEstimator.Huber.Create(2.0)
                obs_noise = gtsam.noiseModel.Robust.Create(
                    huber, gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
                )
            else:
                obs_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)

            for i, (pt3d, pt2d) in enumerate(zip(pts3, pts2)):
                lkey = gtsam.symbol('l', i)
                pt = gtsam.Point3(float(pt3d[0]), float(pt3d[1]), float(pt3d[2]))
                initial.insert(lkey, pt)
                graph.add(gtsam.PriorFactorPoint3(lkey, pt, point_noise))
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(float(pt2d[0]), float(pt2d[1])),
                    obs_noise, pose_key, lkey, cal
                ))

            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(10)
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
            return graph, initial, result

        try:
            # 1st round: robust kernel (Huber) — downweights outliers
            graph1, initial1, result1 = run_opt(pts_3d, pts_2d, initial_pose, use_robust=True)

            if graph1.error(result1) > graph1.error(initial1) * 1.5:
                return initial_pose.copy()

            pose_after_r1 = to_T_cw(result1.atPose3(pose_key))

            # Compute reprojection errors to find inliers (chi2 threshold: 5.991 = 2dof, 95%)
            R = pose_after_r1[:3, :3]
            t = pose_after_r1[:3, 3]
            inlier_mask = []
            for pt3d, pt2d in zip(pts_3d, pts_2d):
                pt_cam = R @ pt3d + t
                if pt_cam[2] <= 0:
                    inlier_mask.append(False)
                    continue
                proj = (self.K @ pt_cam)
                proj_2d = proj[:2] / proj[2]
                err = float(np.sum((proj_2d - pt2d) ** 2))
                inlier_mask.append(err < 5.991)

            inlier_mask = np.array(inlier_mask)
            if inlier_mask.sum() < self.min_inliers:
                return pose_after_r1

            # 2nd round: inliers only, no robust kernel
            pts_3d_in = pts_3d[inlier_mask]
            pts_2d_in = pts_2d[inlier_mask]
            graph2, initial2, result2 = run_opt(pts_3d_in, pts_2d_in, pose_after_r1, use_robust=False)

            if graph2.error(result2) > graph2.error(initial2) * 1.5:
                return pose_after_r1

            return to_T_cw(result2.atPose3(pose_key))

        except Exception:
            return initial_pose.copy()

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

            # Absolute minimum — tracking barely surviving
            if n_tracked < self.min_tracked_points:
                return True

            # Compare against reference KF's MapPoint count
            if self.reference_keyframe is not None:
                if isinstance(self.reference_keyframe, dict):
                    n_ref = sum(1 for mp in self.reference_keyframe.get('mappoints', []) if mp is not None)
                else:
                    n_ref = sum(1 for mp in self.reference_keyframe.get_mappoints() if mp is not None)
                if n_ref > 0 and n_tracked < n_ref * self.ref_kf_tracked_ratio:
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
