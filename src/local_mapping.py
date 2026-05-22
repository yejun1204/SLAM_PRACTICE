"""
Local Mapping module for monocular SLAM

Processes new KeyFrames inserted by Tracking:
- Creates KeyFrame objects and adds them to the Map
- Triangulates new MapPoints from covisible KeyFrame pairs
- Culls bad MapPoints
- Performs Local Bundle Adjustment
"""

import cv2
import numpy as np
import threading
import gtsam

from src.keyframe import KeyFrame
from src.map_point import MapPoint


def triangulate_points(K, pose1, pose2, pts1, pts2):
    """
    Triangulate 3D points from two camera views.

    Args:
        K: Camera intrinsic matrix (3x3)
        pose1: T_cw for camera 1 (4x4)
        pose2: T_cw for camera 2 (4x4)
        pts1: 2D points in image 1 (Nx2)
        pts2: 2D points in image 2 (Nx2)

    Returns:
        points_3d: (N, 3) world-frame 3D points
        valid: boolean mask for points with positive depth in both cameras
    """
    P1 = K @ pose1[:3, :]  # 3x4
    P2 = K @ pose2[:3, :]  # 3x4

    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = (points_4d[:3, :] / points_4d[3, :]).T  # (N, 3)

    # Positive depth check in both cameras
    pts_cam1 = (pose1[:3, :3] @ points_3d.T + pose1[:3, 3:4]).T
    pts_cam2 = (pose2[:3, :3] @ points_3d.T + pose2[:3, 3:4]).T

    valid = (pts_cam1[:, 2] > 0) & (pts_cam2[:, 2] > 0)

    return points_3d, valid


class LocalMapping:
    def __init__(self, slam_map, K, dist_coeffs):
        """
        Args:
            slam_map: Map object
            K: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
        """
        self.map = slam_map
        self.K = K.copy()
        self.dist_coeffs = dist_coeffs.copy()

        self.current_keyframe = None
        self.recent_added_mappoints = []

        # Feature matcher
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def process_new_keyframe(self, kf_data):
        """
        Process a new KeyFrame from Tracking.

        Creates a KeyFrame object, wires up existing MapPoint observations,
        updates the covisibility graph, and adds it to the Map.

        Args:
            kf_data: dict from Tracking.create_keyframe() with keys:
                frame_id, timestamp, image, keypoints, descriptors,
                mappoints (list of MapPoint or None), pose (T_cw 4x4)

        Returns:
            KeyFrame object
        """
        # 1. Create KeyFrame object
        kf = KeyFrame(
            frame_id=kf_data['frame_id'],
            timestamp=kf_data['timestamp'],
            pose=kf_data['pose'],
            K=self.K,
            dist_coeffs=self.dist_coeffs,
            keypoints=kf_data['keypoints'],
            descriptors=kf_data['descriptors'],
            image=kf_data.get('image')
        )

        # 2. Associate existing MapPoints (bidirectional linking)
        for i, mp in enumerate(kf_data['mappoints']):
            if mp is not None and not mp.is_bad:
                kf.add_mappoint(mp, i)
                mp.add_observation(kf, i)
                mp.update_descriptor()

        # 3. Update covisibility graph
        kf.update_connections()

        # 4. Add to map
        self.map.add_keyframe(kf)

        # 5. Store as current
        self.current_keyframe = kf

        return kf

    def create_new_mappoints(self):
        """
        Triangulate new MapPoints from unmatched features in covisible KF pairs.

        For each covisible KeyFrame:
        1. Check baseline is sufficient
        2. Match unmatched features between the two KFs
        3. Triangulate and validate (depth, reprojection error, parallax)
        4. Create MapPoint with bidirectional observations

        Returns:
            Number of new MapPoints created
        """
        kf_curr = self.current_keyframe
        pose_curr = kf_curr.get_pose()
        center_curr = kf_curr.get_camera_center().ravel()

        # Get covisible KeyFrames (limit to top 5 to avoid too many low-quality points)
        covisible_kfs = kf_curr.get_best_covisible_keyframes(n=5)

        # Fall back to all KFs if not enough covisible
        if len(covisible_kfs) < 2:
            covisible_kfs = [kf for kf in self.map.get_all_keyframes()
                             if kf != kf_curr][-5:]

        n_new = 0

        for kf_neighbor in covisible_kfs:
            if kf_neighbor.is_bad:
                continue

            pose_neighbor = kf_neighbor.get_pose()
            center_neighbor = kf_neighbor.get_camera_center().ravel()

            # Check baseline / median depth ratio
            baseline = np.linalg.norm(center_curr - center_neighbor)
            median_depth = self._compute_median_depth(kf_neighbor)
            if median_depth <= 0:
                continue
            if baseline / median_depth < 0.01:
                continue

            # Find unmatched feature indices in both KFs
            curr_mappoints = kf_curr.get_mappoints()
            curr_unmatched = [i for i, mp in enumerate(curr_mappoints) if mp is None]

            neigh_mappoints = kf_neighbor.get_mappoints()
            neigh_unmatched = [i for i, mp in enumerate(neigh_mappoints) if mp is None]

            if len(curr_unmatched) < 10 or len(neigh_unmatched) < 10:
                continue

            # Compute Fundamental matrix (curr → neighbor)
            F = self._compute_fundamental_matrix(pose_curr, pose_neighbor)

            # Epipolar-constrained matching
            kps_curr = [kf_curr.keypoints[i] for i in curr_unmatched]
            kps_neigh = [kf_neighbor.keypoints[i] for i in neigh_unmatched]
            desc_curr = kf_curr.descriptors[curr_unmatched]
            desc_neigh = kf_neighbor.descriptors[neigh_unmatched]

            good_matches = self._match_with_epipolar(
                F, kps_curr, desc_curr, kps_neigh, desc_neigh
            )

            if len(good_matches) < 5:
                continue

            # Get pixel coordinates
            pts_curr = np.array(
                [kf_curr.keypoints[curr_unmatched[m.queryIdx]].pt for m in good_matches],
                dtype=np.float32)
            pts_neigh = np.array(
                [kf_neighbor.keypoints[neigh_unmatched[m.trainIdx]].pt for m in good_matches],
                dtype=np.float32)

            # Triangulate
            points_3d, valid = triangulate_points(
                self.K, pose_curr, pose_neighbor, pts_curr, pts_neigh)

            # Filter and create MapPoints
            for j, m in enumerate(good_matches):
                if not valid[j]:
                    continue

                pt_3d = points_3d[j]

                # Reprojection error check
                err_curr = self._reprojection_error(pt_3d, pose_curr, pts_curr[j])
                err_neigh = self._reprojection_error(pt_3d, pose_neighbor, pts_neigh[j])
                if err_curr > 5.991 or err_neigh > 5.991:
                    continue

                # Parallax check
                ray_curr = pt_3d - center_curr
                ray_neigh = pt_3d - center_neighbor
                norm_curr = np.linalg.norm(ray_curr)
                norm_neigh = np.linalg.norm(ray_neigh)
                if norm_curr < 1e-6 or norm_neigh < 1e-6:
                    continue
                cos_parallax = (ray_curr @ ray_neigh) / (norm_curr * norm_neigh)
                if cos_parallax > 0.9994:  # < ~2.0 degrees
                    continue

                # Depth ratio check: reject points too far relative to baseline
                # If depth >> baseline, triangulation is unreliable
                depth_curr = (pose_curr[:3, :3] @ pt_3d + pose_curr[:3, 3])[2]
                if depth_curr <= 0:
                    continue
                if depth_curr > baseline * 500:
                    continue

                # Create MapPoint
                idx_in_curr = curr_unmatched[m.queryIdx]
                idx_in_neigh = neigh_unmatched[m.trainIdx]

                mp = MapPoint(
                    position=pt_3d,
                    ref_keyframe=kf_curr,
                    descriptor=kf_curr.descriptors[idx_in_curr]
                )

                mp.add_observation(kf_curr, idx_in_curr)
                mp.add_observation(kf_neighbor, idx_in_neigh)
                mp.update_descriptor()
                # Credit the two creation observations so found_ratio starts at 1.0
                mp.increase_visible(2)
                mp.increase_found(2)

                kf_curr.add_mappoint(mp, idx_in_curr)
                kf_neighbor.add_mappoint(mp, idx_in_neigh)

                self.map.add_mappoint(mp)
                self.recent_added_mappoints.append(mp)

                n_new += 1

        # Update covisibility graph
        kf_curr.update_connections()
        for kf_neighbor in covisible_kfs:
            if not kf_neighbor.is_bad:
                kf_neighbor.update_connections()

        return n_new

    def _compute_fundamental_matrix(self, pose1, pose2):
        """Compute fundamental matrix F mapping points from camera1 to epipolar lines in camera2."""
        R1, t1 = pose1[:3, :3], pose1[:3, 3]
        R2, t2 = pose2[:3, :3], pose2[:3, 3]

        # Relative pose: camera1 → camera2
        R_21 = R2 @ R1.T
        t_21 = t2 - R_21 @ t1

        # Skew-symmetric matrix of t_21
        tx = np.array([[0, -t_21[2], t_21[1]],
                       [t_21[2], 0, -t_21[0]],
                       [-t_21[1], t_21[0], 0]])

        K_inv = np.linalg.inv(self.K)
        F = K_inv.T @ tx @ R_21 @ K_inv
        return F

    def _match_with_epipolar(self, F, kps1, desc1, kps2, desc2,
                              epi_threshold=3.0, desc_threshold=50):
        """
        Match features using epipolar constraint.
        For each feature in kps1, compute epipolar line in image2,
        then search for best descriptor match among features near that line.
        """
        import cv2

        matches = []
        for i, (kp1, d1) in enumerate(zip(kps1, desc1)):
            p1 = np.array([kp1.pt[0], kp1.pt[1], 1.0])
            line = F @ p1  # epipolar line in image2: ax + by + c = 0
            a, b, c = line
            denom = np.sqrt(a * a + b * b)
            if denom < 1e-6:
                continue

            best_dist = desc_threshold
            best_j = -1

            for j, (kp2, d2) in enumerate(zip(kps2, desc2)):
                # Epipolar distance
                epi_dist = abs(a * kp2.pt[0] + b * kp2.pt[1] + c) / denom
                if epi_dist > epi_threshold:
                    continue

                # Descriptor distance
                dist = cv2.norm(d1, d2, cv2.NORM_HAMMING)
                if dist < best_dist:
                    best_dist = dist
                    best_j = j

            if best_j >= 0:
                matches.append(cv2.DMatch(i, best_j, best_dist))

        return matches

    def _compute_median_depth(self, keyframe):
        """Compute median depth of MapPoints observed from a KeyFrame."""
        pose = keyframe.get_pose()
        R = pose[:3, :3]
        t = pose[:3, 3]

        depths = []
        for mp in keyframe.get_valid_mappoints():
            pt_world = mp.get_position().ravel()
            pt_cam = R @ pt_world + t
            if pt_cam[2] > 0:
                depths.append(pt_cam[2])

        if len(depths) == 0:
            return -1.0
        return float(np.median(depths))

    def _reprojection_error(self, pt_3d, pose, pt_2d_observed):
        """Compute squared reprojection error."""
        pt_cam = pose[:3, :3] @ pt_3d + pose[:3, 3]
        if pt_cam[2] <= 0:
            return float('inf')
        pt_proj = self.K @ pt_cam
        pt_proj_2d = pt_proj[:2] / pt_proj[2]
        diff = pt_proj_2d - pt_2d_observed
        return float(diff @ diff)

    def mappoint_culling(self):
        """
        Cull recently added MapPoints that don't meet quality thresholds.

        Criteria (ORB-SLAM style):
        - found_ratio < 0.25 -> bad
        - Created >= 2 KFs ago and observations <= 2 -> bad
        - Created >= 3 KFs ago -> survived, remove from tracking list

        Returns:
            Number of MapPoints culled
        """
        if self.current_keyframe is None:
            return 0

        current_kf_id = self.current_keyframe.id
        n_culled = 0
        surviving = []

        for mp in self.recent_added_mappoints:
            if mp.is_bad:
                continue

            if mp.get_found_ratio() < 0.25:
                mp.set_bad_flag()
                n_culled += 1
                continue

            kf_age = current_kf_id - mp.ref_keyframe.id

            if kf_age >= 3 and mp.num_observations() <= 2:
                mp.set_bad_flag()
                n_culled += 1
                continue

            if kf_age >= 4:
                # Survived the probation period
                continue

            surviving.append(mp)

        self.recent_added_mappoints = surviving
        self.map.cull_bad_mappoints()

        return n_culled

    def local_bundle_adjustment(self, n_local_kfs=10):
        """
        Optimize local KeyFrame poses and MapPoint positions jointly using gtsam.
        """
        kf_curr = self.current_keyframe
        if kf_curr is None:
            return {'skipped': True}

        # 1. Collect local KFs
        local_kfs = [kf_curr]
        for kf in kf_curr.get_best_covisible_keyframes(n=n_local_kfs):
            if not kf.is_bad:
                local_kfs.append(kf)
        local_kf_set = set(local_kfs)

        # 2. Collect local MPs
        local_mps = set()
        for kf in local_kfs:
            for mp in kf.get_valid_mappoints():
                local_mps.add(mp)
        local_mps = list(local_mps)

        if len(local_mps) < 10:
            return {'skipped': True}

        # 3. Fixed KFs
        first_kf = self.map.get_reference_keyframe()
        fixed_kfs = set()
        for mp in local_mps:
            for kf_obs in mp.get_observations().keys():
                if kf_obs not in local_kf_set and not kf_obs.is_bad:
                    fixed_kfs.add(kf_obs)
        if first_kf in local_kf_set:
            fixed_kfs.add(first_kf)
        fixed_kfs = list(fixed_kfs)

        opt_kfs = [kf for kf in local_kfs if kf not in fixed_kfs]
        if len(opt_kfs) == 0:
            return {'skipped': True}

        # Gauge freedom: fix at least one KF if no fixed KFs
        if len(fixed_kfs) == 0:
            fixed_kfs = [opt_kfs[0]]
            opt_kfs = opt_kfs[1:]
            if len(opt_kfs) == 0:
                return {'skipped': True}

        # 4. gtsam setup
        fx, fy = float(self.K[0, 0]), float(self.K[1, 1])
        cx, cy = float(self.K[0, 2]), float(self.K[1, 2])
        cal = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)

        huber = gtsam.noiseModel.mEstimator.Huber.Create(2.0)
        noise = gtsam.noiseModel.Robust.Create(
            huber, gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        )
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.ones(6) * 1e-6)

        graph = gtsam.NonlinearFactorGraph()
        initial = gtsam.Values()

        def pose_key(kf):
            return gtsam.symbol('x', kf.id)

        def point_key(i):
            return gtsam.symbol('l', i)

        def to_gtsam_pose(T_cw):
            R_cw = T_cw[:3, :3]
            t_cw = T_cw[:3, 3]
            R_wc = R_cw.T
            t_wc = -R_cw.T @ t_cw
            return gtsam.Pose3(gtsam.Rot3(R_wc), gtsam.Point3(t_wc))

        # 5. Insert pose variables
        all_kfs = opt_kfs + list(fixed_kfs)
        for kf in all_kfs:
            initial.insert(pose_key(kf), to_gtsam_pose(kf.get_pose()))

        # Fix KFs with strong prior
        for kf in fixed_kfs:
            graph.add(gtsam.PriorFactorPose3(
                pose_key(kf), initial.atPose3(pose_key(kf)), prior_noise
            ))

        # 6. Insert point variables and projection factors
        fixed_kf_set = set(fixed_kfs)
        n_factors = 0
        mp_idx_map = {}

        for i, mp in enumerate(local_mps):
            mp_idx_map[mp] = i
            initial.insert(point_key(i), gtsam.Point3(mp.get_position().ravel().astype(float)))

            for kf_obs, kp_idx in mp.get_observations().items():
                if kf_obs.is_bad:
                    continue
                if kf_obs not in local_kf_set and kf_obs not in fixed_kf_set:
                    continue
                pt = kf_obs.keypoints[kp_idx].pt
                graph.add(gtsam.GenericProjectionFactorCal3_S2(
                    gtsam.Point2(float(pt[0]), float(pt[1])),
                    noise, pose_key(kf_obs), point_key(i), cal
                ))
                n_factors += 1

        if n_factors < 10:
            return {'skipped': True}

        # 7. Optimize
        cost_before = graph.error(initial)
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(20)
            result = gtsam.LevenbergMarquardtOptimizer(graph, initial, params).optimize()
        except Exception as e:
            return {'skipped': True, 'reason': str(e)}

        cost_after = graph.error(result)
        if cost_after > cost_before * 1.5:
            return {'skipped': True, 'reason': 'BA diverged'}

        # 8. Apply results
        for kf in opt_kfs:
            pose = result.atPose3(pose_key(kf))
            R_wc = pose.rotation().matrix()
            t_wc = pose.translation()
            R_cw = R_wc.T
            t_cw = -R_wc.T @ t_wc
            new_pose = np.eye(4)
            new_pose[:3, :3] = R_cw
            new_pose[:3, 3] = t_cw
            kf.set_pose(new_pose)

        for mp in local_mps:
            pt = result.atPoint3(point_key(mp_idx_map[mp]))
            mp.set_position(np.array(pt))

        return {
            'skipped': False,
            'n_opt_kfs': len(opt_kfs),
            'n_fixed_kfs': len(fixed_kfs),
            'n_local_mps': len(local_mps),
            'n_observations': n_factors,
            'cost_before': cost_before,
            'cost_after': cost_after,
        }
