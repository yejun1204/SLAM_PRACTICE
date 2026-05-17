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
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

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

            # Extract descriptors for unmatched features
            desc_curr = kf_curr.descriptors[curr_unmatched]
            desc_neigh = kf_neighbor.descriptors[neigh_unmatched]

            # Match with Lowe's ratio test
            matches_raw = self.matcher.knnMatch(desc_curr, desc_neigh, k=2)
            good_matches = []
            for match_pair in matches_raw:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)

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

            if kf_age >= 2 and mp.num_observations() <= 2:
                mp.set_bad_flag()
                n_culled += 1
                continue

            if kf_age >= 3:
                # Survived the probation period
                continue

            surviving.append(mp)

        self.recent_added_mappoints = surviving
        self.map.cull_bad_mappoints()

        return n_culled

    def local_bundle_adjustment(self, n_local_kfs=10):
        """
        Optimize local KeyFrame poses and MapPoint positions jointly.

        Uses scipy.optimize.least_squares with Huber loss to minimize
        reprojection errors in a local window around the current KeyFrame.

        Args:
            n_local_kfs: Max number of covisible KFs to optimize

        Returns:
            dict with optimization statistics
        """
        kf_curr = self.current_keyframe
        if kf_curr is None:
            return {'skipped': True}

        # 1. Collect LOCAL KeyFrames (to optimize)
        local_kfs = [kf_curr]
        for kf in kf_curr.get_best_covisible_keyframes(n=n_local_kfs):
            if not kf.is_bad:
                local_kfs.append(kf)
        local_kf_set = set(local_kfs)

        # 2. Collect LOCAL MapPoints (observed by local KFs)
        local_mps = set()
        for kf in local_kfs:
            for mp in kf.get_valid_mappoints():
                local_mps.add(mp)
        local_mps = list(local_mps)

        if len(local_mps) < 10:
            return {'skipped': True, 'n_local_kfs': len(local_kfs), 'n_local_mps': len(local_mps)}

        # 3. Collect FIXED KeyFrames (observe local MPs but not in local set)
        # Also fix the origin KF to prevent gauge freedom
        first_kf = self.map.get_reference_keyframe()
        fixed_kfs = set()

        for mp in local_mps:
            for kf_obs in mp.get_observations().keys():
                if kf_obs not in local_kf_set and not kf_obs.is_bad:
                    fixed_kfs.add(kf_obs)

        # Origin KF is always fixed
        if first_kf in local_kf_set:
            fixed_kfs.add(first_kf)

        fixed_kfs = list(fixed_kfs)

        # KFs to optimize = local KFs minus fixed
        opt_kfs = [kf for kf in local_kfs if kf not in fixed_kfs]

        if len(opt_kfs) == 0:
            return {'skipped': True, 'n_local_kfs': len(local_kfs), 'n_local_mps': len(local_mps)}

        # 4. Build parameter vector: [kf0_rvec, kf0_tvec, ..., mp0_xyz, ...]
        kf_param_index = {}
        mp_param_index = {}

        idx = 0
        for kf in opt_kfs:
            kf_param_index[kf] = idx
            idx += 6

        for mp in local_mps:
            mp_param_index[mp] = idx
            idx += 3

        n_params = idx

        # 5. Initial parameter vector
        x0 = np.zeros(n_params)
        for kf in opt_kfs:
            pose = kf.get_pose()
            rvec, _ = cv2.Rodrigues(pose[:3, :3])
            i = kf_param_index[kf]
            x0[i:i+3] = rvec.ravel()
            x0[i+3:i+6] = pose[:3, 3].ravel()

        for mp in local_mps:
            i = mp_param_index[mp]
            x0[i:i+3] = mp.get_position().ravel()

        # 6. Build observation list
        fixed_kf_poses = {kf: kf.get_pose() for kf in fixed_kfs}
        observations = []

        for mp in local_mps:
            for kf_obs, kp_idx in mp.get_observations().items():
                if kf_obs in kf_param_index or kf_obs in fixed_kf_poses:
                    pt_2d = np.array(kf_obs.keypoints[kp_idx].pt, dtype=np.float64)
                    observations.append((kf_obs, mp, pt_2d))

        if len(observations) < 10:
            return {'skipped': True, 'n_local_kfs': len(opt_kfs), 'n_local_mps': len(local_mps)}

        K = self.K

        # 7. Residual function
        def compute_residuals(x):
            residuals = np.zeros(len(observations) * 2)

            for obs_idx, (kf_obs, mp, pt_2d_obs) in enumerate(observations):
                # KF pose
                if kf_obs in kf_param_index:
                    i = kf_param_index[kf_obs]
                    R, _ = cv2.Rodrigues(x[i:i+3])
                    t = x[i+3:i+6]
                else:
                    pose = fixed_kf_poses[kf_obs]
                    R = pose[:3, :3]
                    t = pose[:3, 3]

                # MP position
                mp_i = mp_param_index[mp]
                pt_3d = x[mp_i:mp_i+3]

                # Project
                pt_cam = R @ pt_3d + t
                if pt_cam[2] <= 1e-6:
                    residuals[obs_idx*2] = 100.0
                    residuals[obs_idx*2+1] = 100.0
                    continue

                pt_proj = K @ pt_cam
                pt_proj_2d = pt_proj[:2] / pt_proj[2]

                residuals[obs_idx*2] = pt_proj_2d[0] - pt_2d_obs[0]
                residuals[obs_idx*2+1] = pt_proj_2d[1] - pt_2d_obs[1]

            return residuals

        # 8. Build sparsity matrix for Jacobian
        sparsity = lil_matrix((2 * len(observations), n_params), dtype=int)
        for obs_idx, (kf_obs, mp, _) in enumerate(observations):
            if kf_obs in kf_param_index:
                kf_i = kf_param_index[kf_obs]
                sparsity[obs_idx*2:obs_idx*2+2, kf_i:kf_i+6] = 1
            mp_i = mp_param_index[mp]
            sparsity[obs_idx*2:obs_idx*2+2, mp_i:mp_i+3] = 1

        # 9. Optimize
        result = least_squares(
            compute_residuals,
            x0,
            jac_sparsity=sparsity,
            method='trf',
            loss='huber',
            f_scale=1.0,
            max_nfev=50,
            verbose=0
        )

        # 10. Apply optimized values
        x_opt = result.x

        for kf in opt_kfs:
            i = kf_param_index[kf]
            R, _ = cv2.Rodrigues(x_opt[i:i+3])
            new_pose = np.eye(4)
            new_pose[:3, :3] = R
            new_pose[:3, 3] = x_opt[i+3:i+6]
            kf.set_pose(new_pose)

        for mp in local_mps:
            i = mp_param_index[mp]
            mp.set_position(x_opt[i:i+3])

        return {
            'skipped': False,
            'n_opt_kfs': len(opt_kfs),
            'n_fixed_kfs': len(fixed_kfs),
            'n_local_mps': len(local_mps),
            'n_observations': len(observations),
            'cost': float(result.cost),
            'success': bool(result.success),
            'n_evaluations': result.nfev
        }
