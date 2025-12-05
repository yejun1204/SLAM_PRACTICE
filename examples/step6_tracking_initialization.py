"""
Step 6: Tracking - Initialization
Monocular SLAM initialization using Homography vs Fundamental matrix model selection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.euroc_loader import EuRoCLoader
from src.initializer import Initializer
from src.map_point import MapPoint
from src.keyframe import KeyFrame
from src.map import Map


def visualize_initialization(ref_img, curr_img, ref_kp, curr_kp, matches, mask, model):
    """
    Visualize initialization matches

    Args:
        ref_img, curr_img: Reference and current images
        ref_kp, curr_kp: Keypoints
        matches: List of cv2.DMatch
        mask: Inlier mask
        model: Selected model ('H' or 'F')
    """
    # Separate inliers and outliers
    matches_inliers = []
    matches_outliers = []

    for i, m in enumerate(matches):
        if mask[i]:
            matches_inliers.append(m)
        else:
            matches_outliers.append(m)

    # Draw inliers (green) and outliers (red)
    img_matches = cv2.drawMatches(
        ref_img, ref_kp, curr_img, curr_kp, matches_inliers, None,
        matchColor=(0, 255, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    img_matches = cv2.drawMatches(
        ref_img, ref_kp, curr_img, curr_kp, matches_outliers, img_matches,
        matchColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG
    )

    # Add text information
    inlier_count = np.sum(mask)
    total_count = len(matches)
    inlier_ratio = inlier_count / total_count if total_count > 0 else 0

    cv2.putText(img_matches, f"Model: {model} | Inliers: {inlier_count}/{total_count} ({inlier_ratio*100:.1f}%)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    cv2.imshow("Initialization Matches - Green: Inliers, Red: Outliers", img_matches)


def visualize_map_3d(slam_map, window_name="Initial Map"):
    """
    Visualize initial 3D map

    Args:
        slam_map: Map object
        window_name: Window name
    """
    mappoints = slam_map.get_valid_mappoints()
    keyframes = slam_map.get_all_keyframes()

    if len(mappoints) == 0:
        print("No MapPoints to visualize")
        return

    # Create canvas
    canvas = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    margin = 50
    plot_width = 600 - 2 * margin
    plot_height = 600 - 2 * margin - 50

    # Extract 3D positions
    mp_positions = np.array([mp.get_position().ravel() for mp in mappoints])
    kf_positions = np.array([kf.get_camera_center().ravel() for kf in keyframes])

    all_positions = np.vstack([mp_positions, kf_positions])

    # ===== Left: X-Z plane (Top View) =====
    cv2.putText(canvas, "X-Z Plane (Top View)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    x_coords = all_positions[:, 0]
    z_coords = all_positions[:, 2]

    x_min, x_max = x_coords.min(), x_coords.max()
    z_min, z_max = z_coords.min(), z_coords.max()

    x_range = x_max - x_min if x_max - x_min > 0.01 else 1.0
    z_range = z_max - z_min if z_max - z_min > 0.01 else 1.0

    scale_xz = min(plot_width / x_range, plot_height / z_range) * 0.8

    def to_canvas_xz(x, z):
        canvas_x = int(margin + (x - x_min) * scale_xz)
        canvas_y = int(plot_height + margin + 50 - (z - z_min) * scale_xz)
        return canvas_x, canvas_y

    # Draw MapPoints (blue dots)
    for mp_pos in mp_positions:
        pt = to_canvas_xz(mp_pos[0], mp_pos[2])
        cv2.circle(canvas, pt, 2, (255, 0, 0), -1)

    # Draw KeyFrames (red squares)
    for i, kf_pos in enumerate(kf_positions):
        pt = to_canvas_xz(kf_pos[0], kf_pos[2])
        cv2.rectangle(canvas, (pt[0]-5, pt[1]-5), (pt[0]+5, pt[1]+5), (0, 0, 255), -1)
        cv2.putText(canvas, f"KF{i}", (pt[0]+8, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw baseline
    if len(kf_positions) == 2:
        pt1 = to_canvas_xz(kf_positions[0, 0], kf_positions[0, 2])
        pt2 = to_canvas_xz(kf_positions[1, 0], kf_positions[1, 2])
        cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

    # ===== Right: Z-Y plane (Side View) =====
    cv2.putText(canvas, "Z-Y Plane (Side View)",
                (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    y_coords = all_positions[:, 1]

    z_min2, z_max2 = z_coords.min(), z_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    z_range2 = z_max2 - z_min2 if z_max2 - z_min2 > 0.01 else 1.0
    y_range = y_max - y_min if y_max - y_min > 0.01 else 1.0

    scale_zy = min(plot_width / z_range2, plot_height / y_range) * 0.8

    def to_canvas_zy(z, y):
        canvas_x = int(600 + margin + (z - z_min2) * scale_zy)
        canvas_y = int(margin + 50 + (y - y_min) * scale_zy)
        return canvas_x, canvas_y

    # Draw MapPoints
    for mp_pos in mp_positions:
        pt = to_canvas_zy(mp_pos[2], mp_pos[1])
        cv2.circle(canvas, pt, 2, (255, 0, 0), -1)

    # Draw KeyFrames
    for i, kf_pos in enumerate(kf_positions):
        pt = to_canvas_zy(kf_pos[2], kf_pos[1])
        cv2.rectangle(canvas, (pt[0]-5, pt[1]-5), (pt[0]+5, pt[1]+5), (0, 0, 255), -1)
        cv2.putText(canvas, f"KF{i}", (pt[0]+8, pt[1]+5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw baseline
    if len(kf_positions) == 2:
        pt1 = to_canvas_zy(kf_positions[0, 2], kf_positions[0, 1])
        pt2 = to_canvas_zy(kf_positions[1, 2], kf_positions[1, 1])
        cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

    # Add statistics
    stats = slam_map.get_statistics()
    cv2.putText(canvas, f"KeyFrames: {stats['valid_keyframes']}",
                (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"MapPoints: {stats['valid_mappoints']}",
                (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, canvas)


def main():
    dataset_path = 'data/MH_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Step 6: Tracking - Initialization")
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
    print("Attempting initialization...")
    print("=" * 60)

    # State machine
    state = "NOT_INITIALIZED"
    ref_frame = None
    slam_map = Map()

    frame_skip = 0  # Skip frames for better baseline

    for img, timestamp, idx in loader:
        if img is None:
            continue

        # Extract ORB features
        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None:
            continue

        current_frame = {
            'image': img,
            'timestamp': timestamp,
            'frame_id': idx,
            'keypoints': keypoints,
            'descriptors': descriptors
        }

        if state == "NOT_INITIALIZED":
            # Set first frame as reference
            if ref_frame is None:
                ref_frame = current_frame
                print(f"\n[Frame {idx}] Set as reference frame")
                print(f"  Features: {len(keypoints)}")

                # Show reference frame
                img_ref_display = cv2.drawKeypoints(
                    img, keypoints, None, color=(0, 255, 0),
                    flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT
                )
                cv2.imshow("Reference Frame", img_ref_display)
                cv2.waitKey(1)
                continue

            # Skip frames for better baseline
            frame_skip += 1
            if frame_skip < 3:  # Try initialization every 3 frames
                continue
            frame_skip = 0

            print(f"\n[Frame {idx}] Attempting initialization...")
            print(f"  Features: {len(keypoints)}")

            # Create initializer
            initializer = Initializer(ref_frame, loader.K, loader.dist_coeffs)

            # Attempt initialization
            success, result = initializer.initialize(current_frame, ratio_thresh=0.75)

            if success:
                print("\n" + "=" * 60)
                print("INITIALIZATION SUCCESSFUL!")
                print("=" * 60)

                # Create initial KeyFrames
                # Reference KeyFrame at origin (T_cw = I, so T_wc = I)
                kf_ref = KeyFrame(
                    frame_id=ref_frame['frame_id'],
                    timestamp=ref_frame['timestamp'],
                    pose=np.eye(4),  # T_cw: world to camera (identity)
                    K=loader.K,
                    dist_coeffs=loader.dist_coeffs,
                    keypoints=ref_frame['keypoints'],
                    descriptors=ref_frame['descriptors'],
                    image=ref_frame['image']
                )

                # Current KeyFrame with relative pose
                # Initializer returns R, t as cam1 to cam2 (T_21)
                # So T_wc2 = T_w1 @ T_12 = I @ T_12 = T_12
                # But KeyFrame needs T_cw = T_wc^-1
                T_wc = np.eye(4)
                T_wc[:3, :3] = result['R']
                T_wc[:3, 3:4] = result['t']
                T_cw = np.linalg.inv(T_wc)

                kf_curr = KeyFrame(
                    frame_id=current_frame['frame_id'],
                    timestamp=current_frame['timestamp'],
                    pose=T_cw,  # T_cw: world to camera
                    K=loader.K,
                    dist_coeffs=loader.dist_coeffs,
                    keypoints=current_frame['keypoints'],
                    descriptors=current_frame['descriptors'],
                    image=current_frame['image']
                )

                # Add KeyFrames to map
                slam_map.add_keyframe(kf_ref)
                slam_map.add_keyframe(kf_curr)

                print(f"\nCreated KeyFrames:")
                print(f"  {kf_ref}")
                print(f"  {kf_curr}")

                # Create MapPoints
                inlier_matches = [m for i, m in enumerate(result['matches']) if result['mask'][i]]

                for i, (point_3d, match) in enumerate(zip(result['points_3d'], inlier_matches)):
                    # Create MapPoint
                    mp = MapPoint(
                        position=point_3d,
                        ref_keyframe=kf_ref,
                        descriptor=ref_frame['descriptors'][match.queryIdx]
                    )

                    # Add observations
                    mp.add_observation(kf_ref, match.queryIdx)
                    mp.add_observation(kf_curr, match.trainIdx)

                    # Associate with KeyFrames
                    kf_ref.add_mappoint(mp, match.queryIdx)
                    kf_curr.add_mappoint(mp, match.trainIdx)

                    # Add to map
                    slam_map.add_mappoint(mp)

                # Update KeyFrame connections
                kf_ref.update_connections()
                kf_curr.update_connections()

                # Print statistics
                stats = slam_map.get_statistics()
                print(f"\nInitial Map Statistics:")
                print(f"  KeyFrames: {stats['valid_keyframes']}")
                print(f"  MapPoints: {stats['valid_mappoints']}")
                print(f"  Parallax: {result['parallax']:.2f} degrees")
                print(f"  Model: {result['model']}")

                # Visualize
                visualize_initialization(
                    ref_frame['image'], current_frame['image'],
                    ref_frame['keypoints'], current_frame['keypoints'],
                    result['matches'], result['mask'], result['model']
                )

                visualize_map_3d(slam_map, "Initial Map")

                print("\n" + "=" * 60)
                print("Press any key to exit...")
                cv2.waitKey(0)

                state = "INITIALIZED"
                break

            else:
                print("  Initialization failed, trying next frame...")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if state != "INITIALIZED":
        print("\nFailed to initialize!")

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
