"""
Initializer - Two-frame initialization for monocular SLAM
Uses Homography and Fundamental matrix models to select best initialization
"""

import cv2
import numpy as np


class Initializer:
    """
    Monocular initialization with automatic model selection (H vs F)
    Based on ORB-SLAM3 initialization strategy
    """

    def __init__(self, ref_frame_data, K, dist_coeffs, sigma=1.0):
        """
        Initialize with reference frame

        Args:
            ref_frame_data: Dictionary containing:
                - 'keypoints': List of cv2.KeyPoint
                - 'descriptors': ORB descriptors (Nx32 uint8)
                - 'image': Grayscale image
                - 'frame_id': Frame ID
                - 'timestamp': Timestamp
            K: Camera intrinsic matrix (3x3)
            dist_coeffs: Distortion coefficients
            sigma: Standard deviation for RANSAC (default: 1.0 pixel)
        """
        self.ref_frame = ref_frame_data
        self.K = K.copy()
        self.dist_coeffs = dist_coeffs.copy()
        self.sigma = sigma

        # For model selection
        self.min_parallax = 1.0  # Minimum parallax in degrees
        self.min_triangulated = 50  # Minimum triangulated points

    def initialize(self, curr_frame_data, ratio_thresh=0.75):
        """
        Attempt initialization with current frame

        Args:
            curr_frame_data: Dictionary containing current frame data (same format as ref_frame)
            ratio_thresh: Lowe's ratio test threshold for feature matching

        Returns:
            success: Boolean indicating if initialization succeeded
            result: Dictionary containing initialization results if successful:
                - 'R': Rotation matrix (3x3) from ref to curr
                - 't': Translation vector (3x1) from ref to curr (unit norm)
                - 'points_3d': Triangulated 3D points in ref camera frame (Nx3)
                - 'matches': List of cv2.DMatch for good matches
                - 'mask': Inlier mask for matches
                - 'model': Selected model ('H' or 'F')
        """
        # Step 1: Match features between reference and current frame
        matches = self._match_features(
            self.ref_frame['descriptors'],
            curr_frame_data['descriptors'],
            ratio_thresh
        )

        print(f"  Feature matches: {len(matches)}")

        if len(matches) < 100:
            print("  Not enough matches for initialization")
            return False, None

        # Step 2: Extract matched keypoint coordinates
        ref_pts = np.float32([self.ref_frame['keypoints'][m.queryIdx].pt for m in matches])
        curr_pts = np.float32([curr_frame_data['keypoints'][m.trainIdx].pt for m in matches])

        # Step 3: Undistort keypoints
        ref_pts_undist = cv2.undistortPoints(
            ref_pts.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
            None,
            self.K
        ).reshape(-1, 2)

        curr_pts_undist = cv2.undistortPoints(
            curr_pts.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
            None,
            self.K
        ).reshape(-1, 2)

        # Step 4: Compute both Homography and Fundamental matrix
        H, mask_H, score_H = self._find_homography(ref_pts_undist, curr_pts_undist)
        F, mask_F, score_F = self._find_fundamental(ref_pts_undist, curr_pts_undist)

        print(f"  Homography score: {score_H:.2f}, inliers: {np.sum(mask_H)}")
        print(f"  Fundamental score: {score_F:.2f}, inliers: {np.sum(mask_F)}")

        # Step 5: Select best model (H vs F)
        ratio_H_F = score_H / (score_H + score_F)
        print(f"  H/(H+F) ratio: {ratio_H_F:.3f}")

        # Use Homography if scene is planar (ratio > 0.45)
        # Otherwise use Fundamental matrix
        use_homography = ratio_H_F > 0.45

        if use_homography:
            print("  Selected model: Homography (planar scene)")
            success, R, t, points_3d, mask = self._reconstruct_from_H(
                H, ref_pts_undist, curr_pts_undist, mask_H
            )
            model = 'H'
        else:
            print("  Selected model: Fundamental (general scene)")
            success, R, t, points_3d, mask = self._reconstruct_from_F(
                F, ref_pts_undist, curr_pts_undist, mask_F
            )
            model = 'F'

        if not success:
            print("  Reconstruction failed")
            return False, None

        # Check parallax and number of triangulated points
        parallax = self._compute_parallax(ref_pts_undist[mask.ravel().astype(bool)],
                                          curr_pts_undist[mask.ravel().astype(bool)])

        n_good = len(points_3d)

        # Compute mean reprojection error for successful points
        if n_good > 0:
            good_pts1 = ref_pts_undist[mask.ravel().astype(bool)]
            good_pts2 = curr_pts_undist[mask.ravel().astype(bool)]
            t_used = t.ravel()

            errors = []
            for i, pt in enumerate(points_3d):
                pt_proj1 = self.K @ pt
                pt_proj1 = pt_proj1[:2] / pt_proj1[2]
                err1 = np.sqrt(np.sum((pt_proj1 - good_pts1[i]) ** 2))

                pt_proj2 = self.K @ (R @ pt + t_used)
                pt_proj2 = pt_proj2[:2] / pt_proj2[2]
                err2 = np.sqrt(np.sum((pt_proj2 - good_pts2[i]) ** 2))

                errors.append((err1 + err2) / 2)
            mean_err = float(np.mean(errors))
        else:
            mean_err = float('inf')

        print(f"  Parallax: {parallax:.2f} deg, triangulated points: {n_good}, mean reproj error: {mean_err:.2f} px")

        if parallax < self.min_parallax:
            print(f"  Insufficient parallax (< {self.min_parallax} deg)")
            return False, None

        if n_good < self.min_triangulated:
            print(f"  Too few triangulated points (< {self.min_triangulated})")
            return False, None

        # Success!
        print(f"  Initialization successful!")

        result = {
            'R': R,
            't': t,
            'points_3d': points_3d,
            'matches': matches,
            'mask': mask,
            'model': model,
            'parallax': parallax,
            'n_points': n_good
        }

        return True, result

    def _match_features(self, desc1, desc2, ratio_thresh):
        """
        Match features using BF matcher with Lowe's ratio test

        Args:
            desc1, desc2: ORB descriptors
            ratio_thresh: Ratio test threshold

        Returns:
            List of cv2.DMatch
        """
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(desc1, desc2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        return good_matches

    def _find_homography(self, pts1, pts2):
        """
        Compute Homography using RANSAC

        Args:
            pts1, pts2: Matched point coordinates (Nx2)

        Returns:
            H: Homography matrix (3x3)
            mask: Inlier mask
            score: Score based on inliers (higher is better)
        """
        H, mask = cv2.findHomography(
            pts1, pts2,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.sigma,
            confidence=0.999
        )

        if H is None:
            return None, np.zeros(len(pts1), dtype=bool), 0.0

        # Compute score (symmetric transfer error)
        score = self._compute_homography_score(H, pts1, pts2, mask)

        return H, mask.ravel().astype(bool), score

    def _find_fundamental(self, pts1, pts2):
        """
        Compute Fundamental matrix using RANSAC

        Args:
            pts1, pts2: Matched point coordinates (Nx2)

        Returns:
            F: Fundamental matrix (3x3)
            mask: Inlier mask
            score: Score based on inliers
        """
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.sigma,
            confidence=0.999
        )

        if F is None:
            return None, np.zeros(len(pts1), dtype=bool), 0.0

        # Compute score (symmetric epipolar distance)
        score = self._compute_fundamental_score(F, pts1, pts2, mask)

        return F, mask.ravel().astype(bool), score

    def _compute_homography_score(self, H, pts1, pts2, mask):
        """
        Compute Homography score using symmetric transfer error

        Args:
            H: Homography matrix (3x3)
            pts1, pts2: Point correspondences (Nx2)
            mask: Inlier mask

        Returns:
            Score (sum of inlier weights)
        """
        # Convert to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

        # Forward transfer: pts1 -> pts2
        pts2_proj = (H @ pts1_h.T).T
        pts2_proj = pts2_proj[:, :2] / pts2_proj[:, 2:3]
        error_forward = np.sum((pts2 - pts2_proj) ** 2, axis=1)

        # Backward transfer: pts2 -> pts1
        H_inv = np.linalg.inv(H)
        pts1_proj = (H_inv @ pts2_h.T).T
        pts1_proj = pts1_proj[:, :2] / pts1_proj[:, 2:3]
        error_backward = np.sum((pts1 - pts1_proj) ** 2, axis=1)

        # Symmetric error
        error = error_forward + error_backward

        # Compute score (robust to outliers using Huber-like weighting)
        threshold = 5.991 * self.sigma ** 2  # Chi-square threshold
        score = 0.0
        for i in range(len(error)):
            if mask[i]:
                if error[i] < threshold:
                    score += (threshold - error[i])

        return score

    def _compute_fundamental_score(self, F, pts1, pts2, mask):
        """
        Compute Fundamental matrix score using symmetric epipolar distance

        Args:
            F: Fundamental matrix (3x3)
            pts1, pts2: Point correspondences (Nx2)
            mask: Inlier mask

        Returns:
            Score (sum of inlier weights)
        """
        # Convert to homogeneous coordinates
        pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
        pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

        # Epipolar lines: l2 = F @ pts1, l1 = F^T @ pts2
        lines2 = (F @ pts1_h.T).T  # (N, 3)
        lines1 = (F.T @ pts2_h.T).T  # (N, 3)

        # Symmetric epipolar distance
        # d(pt2, l2) = |pt2^T @ l2| / sqrt(l2[0]^2 + l2[1]^2)
        dist2 = np.abs(np.sum(pts2_h * lines2, axis=1)) / np.sqrt(lines2[:, 0]**2 + lines2[:, 1]**2)
        dist1 = np.abs(np.sum(pts1_h * lines1, axis=1)) / np.sqrt(lines1[:, 0]**2 + lines1[:, 1]**2)

        error = dist1 + dist2

        # Compute score
        threshold = 3.841 * self.sigma  # Chi-square threshold
        score = 0.0
        for i in range(len(error)):
            if mask[i]:
                if error[i] < threshold:
                    score += (threshold - error[i])

        return score

    def _reconstruct_from_H(self, H, pts1, pts2, mask):
        """
        Reconstruct 3D structure from Homography

        Args:
            H: Homography matrix (3x3)
            pts1, pts2: Point correspondences (Nx2)
            mask: Inlier mask

        Returns:
            success: Boolean
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            points_3d: Triangulated 3D points (Mx3)
            good_mask: Mask for successfully triangulated points
        """
        # Decompose Homography into R, t, n (normal)
        # Returns multiple solutions
        num_solutions, Rs, ts, normals = cv2.decomposeHomographyMat(H, self.K)

        if num_solutions == 0:
            return False, None, None, None, None

        # Test all solutions and pick the one with most points in front
        best_solution = None
        max_good = 0

        inlier_pts1 = pts1[mask]
        inlier_pts2 = pts2[mask]

        for i in range(num_solutions):
            R = Rs[i]
            t = ts[i]

            # Triangulate points
            points_3d, valid_mask = self._triangulate(
                np.eye(3), np.zeros(3), R, t.ravel(), inlier_pts1, inlier_pts2
            )

            n_good = np.sum(valid_mask)

            if n_good > max_good:
                max_good = n_good
                best_solution = (R, t, points_3d[valid_mask], valid_mask)

        if best_solution is None or max_good < self.min_triangulated:
            print(f"  Reconstruction failed: n_good={max_good}, min_required={self.min_triangulated}")
            return False, None, None, None, None

        R, t, points_3d, good_mask = best_solution

        # Update mask to include only successfully triangulated points
        full_mask = np.zeros(len(mask), dtype=bool)
        full_mask[mask] = good_mask

        return True, R, t, points_3d, full_mask

    def _reconstruct_from_F(self, F, pts1, pts2, mask):
        """
        Reconstruct 3D structure from Fundamental matrix

        Args:
            F: Fundamental matrix (3x3)
            pts1, pts2: Point correspondences (Nx2)
            mask: Inlier mask

        Returns:
            success: Boolean
            R: Rotation matrix (3x3)
            t: Translation vector (3x1)
            points_3d: Triangulated 3D points (Mx3)
            good_mask: Mask for successfully triangulated points
        """
        # Compute Essential matrix: E = K^T @ F @ K
        E = self.K.T @ F @ self.K

        # Recover pose from Essential matrix
        inlier_pts1 = pts1[mask]
        inlier_pts2 = pts2[mask]

        _, R, t, pose_mask = cv2.recoverPose(E, inlier_pts1, inlier_pts2, self.K)

        # Triangulate points
        points_3d, valid_mask = self._triangulate(
            np.eye(3), np.zeros(3), R, t.ravel(), inlier_pts1, inlier_pts2
        )

        n_good = np.sum(valid_mask)

        if n_good < self.min_triangulated:
            print(f"  Reconstruction failed: n_good={n_good}, min_required={self.min_triangulated}")
            return False, None, None, None, None

        # Update mask
        full_mask = np.zeros(len(mask), dtype=bool)
        full_mask[mask] = valid_mask

        return True, R, t, points_3d[valid_mask], full_mask

    def _triangulate(self, R1, t1, R2, t2, pts1, pts2):
        """
        Triangulate 3D points from two views

        Args:
            R1, t1: Camera 1 pose - T^1_w = [R^1_w | t^1_w] (world to cam1)
                    where t^1_w is world origin in cam1 frame
            R2, t2: Camera 2 pose - T^2_w = [R^2_w | t^2_w] (world to cam2)
                    where t^2_w is world origin in cam2 frame
            pts1, pts2: 2D point correspondences (Nx2)

        Returns:
            points_3d: Triangulated 3D points (Nx3) in world (cam1) frame
            valid_mask: Boolean mask for valid points (positive depth)
        """
        # Construct projection matrices
        # P1 = K * T^1_w: world -> image1
        P1 = self.K @ np.hstack([R1, t1.reshape(-1, 1)])
        # P2 = K * T^2_w: world -> image2
        P2 = self.K @ np.hstack([R2, t2.reshape(-1, 1)])

        # Triangulate
        points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert from homogeneous to 3D
        points_3d = points_4d[:3, :] / points_4d[3, :]
        points_3d = points_3d.T  # (N, 3)

        # Check for positive depth in both cameras
        # Camera 1 좌표계로 변환
        points_3d_cam1 = (R1 @ points_3d.T + t1.reshape(-1, 1)).T
        depths1 = points_3d_cam1[:, 2]

        # Camera 2 좌표계로 변환
        points_3d_cam2 = (R2 @ points_3d.T + t2.reshape(-1, 1)).T
        depths2 = points_3d_cam2[:, 2]

        valid_mask = (depths1 > 0) & (depths2 > 0)

        # Reprojection error check
        for i in range(len(points_3d)):
            if not valid_mask[i]:
                continue

            # Camera 1 reprojection
            pt_proj1 = self.K @ points_3d_cam1[i]
            pt_proj1 = pt_proj1[:2] / pt_proj1[2]
            err1 = np.sum((pt_proj1 - pts1[i]) ** 2)

            # Camera 2 reprojection
            pt_proj2 = self.K @ points_3d_cam2[i]
            pt_proj2 = pt_proj2[:2] / pt_proj2[2]
            err2 = np.sum((pt_proj2 - pts2[i]) ** 2)

            if err1 > 5.991 or err2 > 5.991:
                valid_mask[i] = False

        return points_3d, valid_mask

    def _compute_parallax(self, pts1, pts2):
        """
        Compute median parallax angle between two views

        Args:
            pts1, pts2: Matched point coordinates (Nx2)

        Returns:
            Median parallax in degrees
        """
        if len(pts1) == 0:
            return 0.0

        # Convert to normalized coordinates
        pts1_norm = cv2.undistortPoints(
            pts1.reshape(-1, 1, 2),
            self.K,
            np.zeros(5),
            None,
            np.eye(3)
        ).reshape(-1, 2)

        pts2_norm = cv2.undistortPoints(
            pts2.reshape(-1, 1, 2),
            self.K,
            np.zeros(5),
            None,
            np.eye(3)
        ).reshape(-1, 2)

        # Convert to 3D rays (assuming at unit depth)
        rays1 = np.hstack([pts1_norm, np.ones((len(pts1_norm), 1))])
        rays2 = np.hstack([pts2_norm, np.ones((len(pts2_norm), 1))])

        # Normalize rays
        rays1 = rays1 / np.linalg.norm(rays1, axis=1, keepdims=True)
        rays2 = rays2 / np.linalg.norm(rays2, axis=1, keepdims=True)

        # Compute angles between rays
        cos_parallax = np.sum(rays1 * rays2, axis=1)
        cos_parallax = np.clip(cos_parallax, -1.0, 1.0)
        parallax_angles = np.arccos(cos_parallax) * 180.0 / np.pi

        # Return median parallax
        return np.median(parallax_angles)
