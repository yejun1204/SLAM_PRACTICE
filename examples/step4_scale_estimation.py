"""
Step 4: Scale Estimation with Triangulation
Uses 3 consecutive frames to estimate relative scale via triangulation and median depth ratio
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.euroc_loader import EuRoCLoader


def match_features(desc1, desc2, ratio_thresh=0.75):
    """Match features using BF matcher with ratio test"""
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(desc1, desc2, k=2)

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


def visualize_pose_estimation(img1, img2, kp1, kp2, matches, mask, R, t, scale=1.0):
    """
    Visualize inlier/outlier matches and pose information
    """
    # Separate inliers and outliers
    matches_inliers = []
    matches_outliers = []

    for i, m in enumerate(matches):
        if mask[i]:
            matches_inliers.append(m)
        else:
            matches_outliers.append(m)

    # Draw matches (green: inliers, red: outliers)
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_inliers, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add outliers in red
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches_outliers, img_matches,
        matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
    )

    # Add text information
    inlier_count = np.sum(mask)
    total_count = len(matches)
    inlier_ratio = inlier_count / total_count if total_count > 0 else 0

    cv2.putText(img_matches, f"Inliers: {inlier_count}/{total_count} ({inlier_ratio*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show rotation and translation
    rotation_angles = cv2.Rodrigues(R)[0].ravel() * 180 / np.pi
    cv2.putText(img_matches, f"Rotation (deg): [{rotation_angles[0]:.2f}, {rotation_angles[1]:.2f}, {rotation_angles[2]:.2f}]",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    scaled_t = t * scale
    cv2.putText(img_matches, f"Translation: [{scaled_t[0,0]:.3f}, {scaled_t[1,0]:.3f}, {scaled_t[2,0]:.3f}] (scale: {scale:.3f})",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Pose Estimation - Green: Inliers, Red: Outliers", img_matches)


def estimate_pose(kp1, kp2, matches, K, dist_coeffs):
    """
    Estimate relative camera pose using Essential matrix

    Args:
        kp1, kp2: Keypoints from two frames
        matches: Feature matches
        K: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients

    Returns:
        R: Rotation matrix (3x3)
        t: Translation vector (3x1)
        inliers: Inlier mask from RANSAC
        pts1, pts2: Matched point coordinates (undistorted)
    """
    # Extract matched keypoint coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

    # Undistort keypoint coordinates
    pts1_undist = cv2.undistortPoints(
        pts1.reshape(-1, 1, 2),
        K,
        dist_coeffs,
        None,
        K
    ).reshape(-1, 2)

    pts2_undist = cv2.undistortPoints(
        pts2.reshape(-1, 1, 2),
        K,
        dist_coeffs,
        None,
        K
    ).reshape(-1, 2)

    # Compute Essential matrix using RANSAC
    E, mask = cv2.findEssentialMat(
        pts1_undist, pts2_undist, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    # Recover pose (R, t) from Essential matrix
    # Returns: T^2_1 = [R^2_1 | t^2_2->1]
    # where R^2_1: rotation from cam1 to cam2
    #       t^2_2->1: position of cam1's origin in cam2's frame
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_undist, pts2_undist, K, mask=mask)

    return R, t, mask_pose, pts1_undist, pts2_undist


def triangulate_points(R1, t1, R2, t2, pts1, pts2, K):
    """
    Triangulate 3D points from two views

    Args:
        R1, t1: Camera 1 pose - T^1_w = [R^1_w | t^1_w->o] (world to cam1)
        R2, t2: Camera 2 pose - T^2_w = [R^2_w | t^2_w->o] (world to cam2)
        pts1, pts2: 2D point correspondences (Nx2)
        K: Camera intrinsic matrix

    Returns:
        points_3d: Triangulated 3D points (Nx3) in world coordinate frame
        valid_mask: Boolean mask for valid points (positive depth)
    """
    # Construct projection matrices
    # P1 = K * T^1_w: world -> image1
    P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
    # P2 = K * T^2_w: world -> image2
    P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])

    # Triangulate
    # cv2.triangulatePoints expects points in shape (2, N)
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T  # (N, 3)

    # Check for positive depth (valid points)
    # Points should be in front of both cameras
    depths1 = points_3d[:, 2]

    # Transform points to camera 2 frame
    points_3d_cam2 = (R2 @ points_3d.T + t2.reshape(-1, 1)).T
    depths2 = points_3d_cam2[:, 2]

    valid_mask = (depths1 > 0) & (depths2 > 0)

    return points_3d, valid_mask


def estimate_scale_from_translation_ratio(t_prev_norm, points_3d_prev, points_3d_curr, common_indices):
    """
    Estimate relative scale using translation distance ratio

    The idea: If the camera moved twice as far in the current frame pair compared to
    the previous frame pair, the scale should be 2.0.

    We estimate this by comparing the median distances of common 3D points.

    Args:
        t_prev_norm: Norm of translation vector from previous frame pair
        points_3d_prev: 3D points from previous frame pair (Nx3)
        points_3d_curr: 3D points from current frame pair (Mx3)
        common_indices: Tuple of (indices_prev, indices_curr) for common points

    Returns:
        scale_ratio: Ratio of translation distances (curr/prev)
        num_common: Number of common points used
    """
    idx_prev, idx_curr = common_indices

    if len(idx_prev) < 5:
        return 1.0, 0

    # Get common 3D points
    pts_prev = points_3d_prev[idx_prev]
    pts_curr = points_3d_curr[idx_curr]

    # Compute distances from camera origin
    dists_prev = np.linalg.norm(pts_prev, axis=1)
    dists_curr = np.linalg.norm(pts_curr, axis=1)

    # Filter out points too close to camera
    valid = (dists_prev > 0.5) & (dists_curr > 0.5)

    if np.sum(valid) < 5:
        return 1.0, 0

    # Compute distance ratios
    ratios = dists_prev[valid] / dists_curr[valid]

    # Use median for robustness
    scale_ratio = np.median(ratios)

    return scale_ratio, np.sum(valid)


def find_common_matches(matches_12, matches_23):
    """
    Find common feature tracks across 3 frames

    Args:
        matches_12: Matches between frame 1 and 2 (list of DMatch)
        matches_23: Matches between frame 2 and 3 (list of DMatch)

    Returns:
        common_12_indices: Indices in matches_12 for common tracks
        common_23_indices: Indices in matches_23 for common tracks
    """
    # Build mapping: frame2_idx -> frame1_idx
    frame2_to_frame1 = {}
    for i, m in enumerate(matches_12):
        frame2_to_frame1[m.trainIdx] = (i, m.queryIdx)

    common_12_indices = []
    common_23_indices = []

    for j, m in enumerate(matches_23):
        if m.queryIdx in frame2_to_frame1:
            idx_12, frame1_idx = frame2_to_frame1[m.queryIdx]
            common_12_indices.append(idx_12)
            common_23_indices.append(j)

    return common_12_indices, common_23_indices


def plot_trajectory(trajectory, trajectory_poses, current_idx):
    """
    Plot camera trajectory in X-Z plane (top view) and Z-Y plane (side view)
    Also draws camera coordinate axes at current position

    Args:
        trajectory: List of 3D positions
        trajectory_poses: List of 4x4 pose matrices
        current_idx: Current frame index
    """
    if len(trajectory) < 2:
        return

    trajectory = np.array(trajectory)

    # Create canvas with two views side by side
    canvas = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    margin = 50
    plot_width = 600 - 2 * margin
    plot_height = 600 - 2 * margin - 50

    # ===== Left: X-Z plane (Top View) =====
    cv2.putText(canvas, f"X-Z Plane (Top View) - Frame: {current_idx}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Calculate scaling for X-Z view
    x_coords = trajectory[:, 0]
    z_coords = trajectory[:, 2]

    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_range = x_max - x_min if x_max - x_min > 0.01 else 1.0
    z_range = z_max - z_min if z_max - z_min > 0.01 else 1.0

    scale_xz = min(plot_width / x_range, plot_height / z_range) * 0.8

    def to_canvas_xz(x, z):
        """Convert X-Z coords to canvas coords (up is +Z)"""
        canvas_x = int(margin + (x - x_min) * scale_xz)
        canvas_y = int(plot_height + margin + 50 - (z - z_min) * scale_xz)  # Flip Y
        return canvas_x, canvas_y

    # Draw X-Z trajectory
    for i in range(len(trajectory) - 1):
        pt1 = to_canvas_xz(trajectory[i, 0], trajectory[i, 2])
        pt2 = to_canvas_xz(trajectory[i + 1, 0], trajectory[i + 1, 2])
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 2)

    # Draw start (green) and current (red)
    start_pt = to_canvas_xz(trajectory[0, 0], trajectory[0, 2])
    cv2.circle(canvas, start_pt, 6, (0, 255, 0), -1)

    current_pt = to_canvas_xz(trajectory[-1, 0], trajectory[-1, 2])
    cv2.circle(canvas, current_pt, 6, (0, 0, 255), -1)

    # Draw camera axes at current position (X-Z plane: X and Z axes only)
    if len(trajectory_poses) > 0:
        current_pose = trajectory_poses[-1]
        R = current_pose[:3, :3]
        t = current_pose[:3, 3]

        # Adaptive axis length: 10% of the current trajectory range
        axis_length = max(x_range, z_range) * 0.1

        # X axis (red) - project onto X-Z plane
        x_axis_end = t + R[:, 0] * axis_length  # First column of R
        pt_origin = to_canvas_xz(t[0], t[2])
        pt_x_end = to_canvas_xz(x_axis_end[0], x_axis_end[2])
        cv2.arrowedLine(canvas, pt_origin, pt_x_end, (0, 0, 255), 2, tipLength=0.3)

        # Z axis (blue) - project onto X-Z plane
        z_axis_end = t + R[:, 2] * axis_length  # Third column of R
        pt_z_end = to_canvas_xz(z_axis_end[0], z_axis_end[2])
        cv2.arrowedLine(canvas, pt_origin, pt_z_end, (255, 0, 0), 2, tipLength=0.3)

    # ===== Right: Z-Y plane (Side View) =====
    cv2.putText(canvas, "Z-Y Plane (Side View)",
                (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Calculate scaling for Z-Y view
    y_coords = trajectory[:, 1]

    z_min2, z_max2 = z_coords.min(), z_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    z_range2 = z_max2 - z_min2 if z_max2 - z_min2 > 0.01 else 1.0
    y_range = y_max - y_min if y_max - y_min > 0.01 else 1.0

    scale_zy = min(plot_width / z_range2, plot_height / y_range) * 0.8

    def to_canvas_zy(z, y):
        """Convert Z-Y coords to canvas coords (right is +Z, down is +Y)"""
        canvas_x = int(600 + margin + (z - z_min2) * scale_zy)
        canvas_y = int(margin + 50 + (y - y_min) * scale_zy)  # No flip, down is +Y
        return canvas_x, canvas_y

    # Draw Z-Y trajectory
    for i in range(len(trajectory) - 1):
        pt1 = to_canvas_zy(trajectory[i, 2], trajectory[i, 1])
        pt2 = to_canvas_zy(trajectory[i + 1, 2], trajectory[i + 1, 1])
        cv2.line(canvas, pt1, pt2, (255, 0, 0), 2)

    # Draw start (green) and current (red)
    start_pt_zy = to_canvas_zy(trajectory[0, 2], trajectory[0, 1])
    cv2.circle(canvas, start_pt_zy, 6, (0, 255, 0), -1)

    current_pt_zy = to_canvas_zy(trajectory[-1, 2], trajectory[-1, 1])
    cv2.circle(canvas, current_pt_zy, 6, (0, 0, 255), -1)

    # Draw camera axes at current position (Z-Y plane: Z and Y axes only)
    if len(trajectory_poses) > 0:
        current_pose = trajectory_poses[-1]
        R = current_pose[:3, :3]
        t = current_pose[:3, 3]

        # Adaptive axis length: 10% of the current trajectory range
        axis_length_zy = max(z_range2, y_range) * 0.1

        # Z axis (blue) - project onto Z-Y plane
        z_axis_end = t + R[:, 2] * axis_length_zy  # Third column of R
        pt_origin_zy = to_canvas_zy(t[2], t[1])
        pt_z_end_zy = to_canvas_zy(z_axis_end[2], z_axis_end[1])
        cv2.arrowedLine(canvas, pt_origin_zy, pt_z_end_zy, (255, 0, 0), 2, tipLength=0.3)

        # Y axis (green) - project onto Z-Y plane
        y_axis_end = t + R[:, 1] * axis_length_zy  # Second column of R
        pt_y_end_zy = to_canvas_zy(y_axis_end[2], y_axis_end[1])
        cv2.arrowedLine(canvas, pt_origin_zy, pt_y_end_zy, (0, 255, 0), 2, tipLength=0.3)

    # Add coordinate info at bottom
    cv2.putText(canvas, f"Position: X={trajectory[-1, 0]:.3f}, Y={trajectory[-1, 1]:.3f}, Z={trajectory[-1, 2]:.3f}",
                (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display
    cv2.imshow("Camera Trajectory", canvas)


def process_frame_pair(frame_buffer, matches_buffer, points_3d_buffer,
                       loader, current_pose, trajectory, trajectory_poses, accumulated_scale):
    """
    Process a pair of consecutive frames for pose estimation and scale calculation

    Args:
        accumulated_scale: Previously accumulated scale value

    Returns:
        updated_pose: Updated camera pose after processing this frame pair
        should_quit: Whether user requested to quit
        new_accumulated_scale: Updated accumulated scale
        success: Whether processing succeeded (True means add to buffers)
        result_data: Dictionary containing frame pair data if successful
    """
    # ===== Step 2: Get consecutive frame pair =====
    img1, kp1, desc1, idx1 = frame_buffer[-2]
    img2, kp2, desc2, idx2 = frame_buffer[-1]

    print(f"\n" + "=" * 60)
    print(f"Frame pair: {idx1} <-> {idx2}")

    # ===== Step 3: Match features between frames =====
    matches = match_features(desc1, desc2, ratio_thresh=0.75)
    print(f"  Matches found: {len(matches)}")

    if len(matches) < 8:
        print("  Not enough matches! Skipping frame...")
        return current_pose, False, accumulated_scale, False, None

    # ===== Step 4: Estimate relative pose (R, t) =====
    # T^cam2_cam1= [R|t]
    R, t, mask, pts1_undist, pts2_undist = estimate_pose(
        kp1, kp2, matches, loader.K, loader.dist_coeffs
    )

    inlier_count = np.sum(mask)
    print(f"  Inliers: {inlier_count}/{len(matches)} ({inlier_count/len(matches)*100:.1f}%)")

    if inlier_count < 20:
        print("  Not enough inliers after RANSAC! Skipping frame...")
        return current_pose, False, accumulated_scale, False, None

    # ===== Step 5: Triangulate 3D points from inliers =====
    inlier_matches = [m for i, m in enumerate(matches) if mask[i]]
    pts1_inliers = pts1_undist[mask.ravel().astype(bool)]
    pts2_inliers = pts2_undist[mask.ravel().astype(bool)]

    # Triangulate in normalized camera coordinates (camera1 = world origin)
    # T^1_w = [I | 0]: camera1 is the world frame
    # T^2_w = T^2_1 = [R^2_1 | t^2_2->1]: camera2 relative to camera1(=world)
    R1 = np.eye(3)
    t1 = np.zeros(3)
    points_3d, valid_mask = triangulate_points(R1, t1, R, t, pts1_inliers, pts2_inliers, loader.K)

    valid_count = np.sum(valid_mask)
    print(f"  Valid 3D points: {valid_count}/{len(points_3d)}")

    # ===== Step 6: Estimate scale (requires 3 frames) =====
    scale_ratio = 1.0  # Default: no scale correction
    t_norm = np.linalg.norm(t)

    # We need at least 1 previous frame pair in buffer to estimate scale
    # matches_buffer[-1] and inlier_matches form two consecutive pairs
    if len(matches_buffer) >= 1 and len(points_3d_buffer) >= 1:
        # Find common feature tracks across 3 frames
        # matches_buffer[-1]: matches between frame(n-1) and frame(n) - the PREVIOUS pair
        # inlier_matches: matches between frame(n) and frame(n+1) - the CURRENT pair
        common_12_idx, common_23_idx = find_common_matches(matches_buffer[-1], inlier_matches)

        if len(common_12_idx) >= 5:
            # Extract 3D points and transformations from previous pair
            points_3d_12, R_21, t_21 = points_3d_buffer[-1]  # Frame(n-1)-n pair, in frame(n-1) coords
            # Note: t_21 has unit norm, points_3d_12 are triangulated with this normalized translation

            # Current pair: points_3d (frame n to n+1), in frame(n) coords
            # Note: t has unit norm, points_3d are triangulated with this normalized translation

            # Transform previous 3D points from frame(n-1) to frame(n) coordinates
            # R_21, t_21 form T^n_(n-1) (transformation from frame(n-1) to frame n)
            # To transform points: p_n = R_21 @ p_(n-1) + t_21
            points_3d_12_common = points_3d_12[common_12_idx]
            points_3d_12_in_frame_n = (R_21 @ points_3d_12_common.T).T + t_21.ravel()

            # Current pair's 3D points (already in frame n)
            points_3d_23_common = points_3d[valid_mask][common_23_idx]

            # Both point sets now represent THE SAME physical points observed from frame n
            # The depth ratio gives us the scale correction needed
            depths_12_in_n = points_3d_12_in_frame_n[:, 2]  # Z coordinate = depth
            depths_23_in_n = points_3d_23_common[:, 2]

            # Filter valid depths
            valid_depths = (np.abs(depths_12_in_n) > 0.1) & (np.abs(depths_23_in_n) > 0.1)

            if np.sum(valid_depths) >= 5:
                # Scale ratio: how much should we scale the current translation?
                # If depths_12 (from previous scaled translation) > depths_23 (unit translation),
                # we need to scale up the current translation
                depth_ratios = depths_12_in_n[valid_depths] / depths_23_in_n[valid_depths]
                scale_ratio = np.median(depth_ratios)
                print(f"  Scale ratio: {scale_ratio:.6f} from {np.sum(valid_depths)} common points")
            else:
                print(f"  Not enough valid depths, using scale_ratio=1.0")
        else:
            print(f"  Not enough common points ({len(common_12_idx)}), using scale_ratio=1.0")
    else:
        # First frame pair, no previous pair to compare with
        print(f"  First frame pair, using scale_ratio=1.0")

    # ===== Step 7: Apply scale and update camera pose =====
    # Accumulate scale: new_scale = accumulated_scale * scale_ratio
    new_accumulated_scale = accumulated_scale * scale_ratio

    # Apply accumulated scale to current translation
    scaled_t = t * new_accumulated_scale
    print(f"  Translation norm: {t_norm:.3f}, scale ratio: {scale_ratio:.3f}, accumulated scale: {new_accumulated_scale:.3f}")
    print(f"  Scaled translation: {scaled_t.ravel()}")

    # Build transformation matrix T^2_1 = [R^2_1 | t^2_2->1]
    # This represents camera2's pose relative to camera1
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3:4] = scaled_t

    # Update world pose:
    # current_pose = T^w_cam (world to current camera)
    # We need T^w_cam2 = T^w_cam1 @ T^1_2 = T^w_cam1 @ inv(T^2_1)
    current_pose = current_pose @ np.linalg.inv(T)

    # Store trajectory
    trajectory.append(current_pose[:3, 3].copy())
    trajectory_poses.append(current_pose.copy())

    # ===== Step 8: Visualization =====
    # Show feature matches with inliers/outliers
    visualize_pose_estimation(img1, img2, kp1, kp2, matches, mask, R, t, new_accumulated_scale)

    # Show camera trajectory in 2D projections
    plot_trajectory(trajectory, trajectory_poses, idx2)

    # Store current frame's data for next scale estimation
    # inlier_matches: matches between current frame pair
    # (points_3d, R, t): 3D points triangulated in first camera's frame
    result_data = {
        'inlier_matches': inlier_matches,
        'points_3d': points_3d[valid_mask],
        'R': R.copy(),
        't': t.copy()
    }

    # Wait for key press
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("\nQuitting...")
        return current_pose, True, new_accumulated_scale, True, result_data

    return current_pose, False, new_accumulated_scale, True, result_data


def main():
    dataset_path = 'data/MH_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Step 4: Scale Estimation with Triangulation")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    print(f"\nCamera matrix K:")
    print(loader.K)

    # Create ORB detector
    n_features = 2000
    orb = cv2.ORB_create(nfeatures=n_features)

    print(f"\nORB Configuration:")
    print(f"  Max features: {n_features}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  'q'   - Quit")
    print("=" * 60)

    # ===== Trajectory tracking =====
    trajectory = []
    trajectory_poses = []
    current_pose = np.eye(4)  # World coordinate system
    trajectory.append(current_pose[:3, 3].copy())
    trajectory_poses.append(current_pose.copy())

    # ===== Frame buffer for scale estimation =====
    # Only successfully matched frames are stored in these buffers
    prev_frame = None         # Previous successfully matched frame (img, kp, desc, idx)
    matches_buffer = []       # Store inlier matches between consecutive frame pairs
    points_3d_buffer = []     # Store (3D points, R, t) for consecutive frame pairs
    accumulated_scale = 1.0   # Accumulated scale factor

    for img, timestamp, idx in loader:
        if img is None:
            continue

        # ===== Step 1: Extract ORB features =====
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None:
            continue

        current_frame = (img, keypoints, descriptors, idx)

        # Need a previous frame for pose estimation
        if prev_frame is None:
            prev_frame = current_frame
            continue

        # Build temporary frame buffer for processing
        frame_buffer = [prev_frame, current_frame]

        # Process frame pair for pose estimation and scale calculation
        current_pose, should_quit, accumulated_scale, success, result_data = process_frame_pair(
            frame_buffer, matches_buffer, points_3d_buffer,
            loader, current_pose, trajectory, trajectory_poses, accumulated_scale
        )

        # Only update prev_frame and buffers if processing was successful
        if success:
            prev_frame = current_frame  # Move to next frame only on success
            matches_buffer.append(result_data['inlier_matches'])
            points_3d_buffer.append((result_data['points_3d'], result_data['R'], result_data['t']))

            # Keep only last 1 pair for scale estimation (we compare with current pair)
            if len(matches_buffer) > 1:
                matches_buffer.pop(0)
            if len(points_3d_buffer) > 1:
                points_3d_buffer.pop(0)
        # If failed, keep prev_frame unchanged and try again with next frame

        if should_quit:
            break

    # Print trajectory summary
    if len(trajectory) > 0:
        print("\n" + "=" * 60)
        print("Camera Trajectory Summary:")
        print(f"  Total frames processed: {len(trajectory)}")
        trajectory_array = np.array(trajectory)
        print(f"  Final position: {trajectory_array[-1]}")
        total_distance = np.sum(np.linalg.norm(np.diff(trajectory_array, axis=0), axis=1))
        print(f"  Total distance traveled: {total_distance:.3f} units")

        # Keep trajectory window open
        print("\nPress any key to close trajectory window...")
        cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
