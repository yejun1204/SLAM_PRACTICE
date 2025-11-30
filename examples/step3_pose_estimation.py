"""
Step 3: Camera Pose Estimation
Estimates relative camera pose (R, t) from matched features using Essential matrix
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

    # Undistort keypoint coordinates (ORB-SLAM3 방식)
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

    # Compute Essential matrix using RANSAC (with undistorted points)
    E, mask = cv2.findEssentialMat(
        pts1_undist, pts2_undist, K,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=2.0
    )

    # Recover pose (R, t) from Essential matrix
    _, R, t, mask_pose = cv2.recoverPose(E, pts1_undist, pts2_undist, K, mask=mask)

    return R, t, mask_pose, pts1_undist, pts2_undist


def visualize_pose_estimation(img1, img2, kp1, kp2, matches, mask, R, t):
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
    cv2.putText(img_matches, f"Translation: [{t[0,0]:.3f}, {t[1,0]:.3f}, {t[2,0]:.3f}]",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow("Pose Estimation - Green: Inliers, Red: Outliers", img_matches)


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


def main():
    dataset_path = 'data/MH_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Step 3: Camera Pose Estimation Demo")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    print(f"\nCamera matrix K:")
    print(loader.K)

    # Create ORB detector
    n_features = 2000  # More features for better pose estimation
    orb = cv2.ORB_create(nfeatures=n_features)

    print(f"\nORB Configuration:")
    print(f"  Max features: {n_features}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  SPACE - Next frame pair")
    print("  'q'   - Quit")
    print("=" * 60)

    # Track camera trajectory
    trajectory = []
    trajectory_poses = []  # Store full 4x4 poses for drawing axes
    current_pose = np.eye(4)  # 4x4 transformation matrix
    trajectory.append(current_pose[:3, 3].copy())
    trajectory_poses.append(current_pose.copy())

    prev_img = None
    prev_kp = None
    prev_desc = None
    prev_idx = None

    for img, timestamp, idx in loader:
        if img is None:
            continue

        # Extract features
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if prev_img is not None and descriptors is not None and prev_desc is not None:
            print(f"\n" + "=" * 60)
            print(f"Frame pair: {prev_idx} <-> {idx}")

            # Match features
            matches = match_features(prev_desc, descriptors, ratio_thresh=0.75)
            print(f"  Matches found: {len(matches)}")

            if len(matches) >= 8:  # Minimum for Essential matrix
                # Estimate pose (with distortion correction)
                R, t, mask, pts1, pts2 = estimate_pose(
                    prev_kp, keypoints, matches, loader.K, loader.dist_coeffs
                )

                inlier_count = np.sum(mask)
                print(f"  Inliers: {inlier_count}/{len(matches)} ({inlier_count/len(matches)*100:.1f}%)")
                print(f"  Rotation (Rodrigues): {cv2.Rodrigues(R)[0].ravel()}")
                print(f"  Translation: {t.ravel()}")

                if inlier_count < 20:
                    print("  Not enough inliers after RANSAC!")
                    continue
                # Update camera pose (accumulate transformations)
                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3:4] = t
                current_pose = current_pose @ np.linalg.inv(T)

                # Store trajectory
                trajectory.append(current_pose[:3, 3].copy())
                trajectory_poses.append(current_pose.copy())

                # Visualize matches
                visualize_pose_estimation(
                    prev_img, img, prev_kp, keypoints, matches, mask, R, t
                )

                # Plot trajectory in real-time
                plot_trajectory(trajectory, trajectory_poses, idx)
            else:
                print("  Not enough matches for pose estimation!")

            # Wait for key
            key = cv2.waitKey(1)  # Short delay for real-time display
            if key == ord('q'):
                print("\nQuitting...")
                break

        # Update previous frame
        prev_img = img.copy()
        prev_kp = keypoints
        prev_desc = descriptors.copy() if descriptors is not None else None
        prev_idx = idx

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
