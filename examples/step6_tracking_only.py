"""
Step 6: Tracking Only (without Local Mapping)
Initialize with two frames, then track subsequent frames
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
from src.tracking import Tracking, TrackingState


def visualize_tracking(img, keypoints, mappoints, state, n_tracked, pose):
    """
    Visualize tracking result on image

    Args:
        img: Image
        keypoints: List of cv2.KeyPoint
        mappoints: List of MapPoint (or None)
        state: TrackingState
        n_tracked: Number of tracked MapPoints
        pose: Camera pose (T_cw)
    """
    # Draw keypoints
    # Green: tracked MapPoints, Red: untracked features
    img_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for i, kp in enumerate(keypoints):
        pt = tuple(map(int, kp.pt))
        if mappoints[i] is not None:
            cv2.circle(img_vis, pt, 3, (0, 255, 0), -1)  # Green for tracked
        else:
            cv2.circle(img_vis, pt, 2, (0, 0, 255), 1)   # Red for untracked

    # Add tracking info
    state_color = {
        TrackingState.NOT_INITIALIZED: (128, 128, 128),
        TrackingState.OK: (0, 255, 0),
        TrackingState.LOST: (0, 0, 255)
    }

    color = state_color.get(state, (255, 255, 255))
    cv2.putText(img_vis, f"State: {state.name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(img_vis, f"Tracked: {n_tracked}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show camera position
    if pose is not None:
        # Get camera center in world frame: C = -R^T @ t
        R = pose[:3, :3]
        t = pose[:3, 3]
        camera_center = -R.T @ t

        pos_text = f"Pos: [{camera_center[0]:.2f}, {camera_center[1]:.2f}, {camera_center[2]:.2f}]"
        cv2.putText(img_vis, pos_text, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    return img_vis


def visualize_trajectory_3d(slam_map, trajectory, window_name="Trajectory"):
    """
    Visualize 3D trajectory and map

    Args:
        slam_map: Map object
        trajectory: List of camera positions (world frame)
        window_name: Window name
    """
    mappoints = slam_map.get_valid_mappoints()
    keyframes = slam_map.get_all_keyframes()

    if len(mappoints) == 0 and len(trajectory) == 0:
        return

    # Create canvas
    canvas = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    margin = 50
    plot_width = 600 - 2 * margin
    plot_height = 600 - 2 * margin - 50

    # Collect all positions
    positions = []

    # MapPoints
    if len(mappoints) > 0:
        mp_positions = np.array([mp.get_position().ravel() for mp in mappoints])
        positions.append(mp_positions)

    # KeyFrames
    if len(keyframes) > 0:
        kf_positions = np.array([kf.get_camera_center().ravel() for kf in keyframes])
        positions.append(kf_positions)

    # Trajectory
    if len(trajectory) > 0:
        traj_positions = np.array(trajectory)
        positions.append(traj_positions)

    if len(positions) == 0:
        return

    all_positions = np.vstack(positions)

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
    if len(mappoints) > 0:
        for mp_pos in mp_positions:
            pt = to_canvas_xz(mp_pos[0], mp_pos[2])
            if 0 <= pt[0] < 600 and 0 <= pt[1] < 600:
                cv2.circle(canvas, pt, 1, (255, 0, 0), -1)

    # Draw KeyFrames (red squares)
    if len(keyframes) > 0:
        for i, kf_pos in enumerate(kf_positions):
            pt = to_canvas_xz(kf_pos[0], kf_pos[2])
            if 0 <= pt[0] < 600 and 0 <= pt[1] < 600:
                cv2.rectangle(canvas, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), (0, 0, 255), -1)

    # Draw trajectory (green line)
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            pt1 = to_canvas_xz(trajectory[i][0], trajectory[i][2])
            pt2 = to_canvas_xz(trajectory[i+1][0], trajectory[i+1][2])
            if (0 <= pt1[0] < 600 and 0 <= pt1[1] < 600 and
                0 <= pt2[0] < 600 and 0 <= pt2[1] < 600):
                cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

    # Current position (large green circle)
    if len(trajectory) > 0:
        pt = to_canvas_xz(trajectory[-1][0], trajectory[-1][2])
        if 0 <= pt[0] < 600 and 0 <= pt[1] < 600:
            cv2.circle(canvas, pt, 5, (0, 255, 0), -1)

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
    if len(mappoints) > 0:
        for mp_pos in mp_positions:
            pt = to_canvas_zy(mp_pos[2], mp_pos[1])
            if 600 <= pt[0] < 1200 and 0 <= pt[1] < 600:
                cv2.circle(canvas, pt, 1, (255, 0, 0), -1)

    # Draw KeyFrames
    if len(keyframes) > 0:
        for i, kf_pos in enumerate(kf_positions):
            pt = to_canvas_zy(kf_pos[2], kf_pos[1])
            if 600 <= pt[0] < 1200 and 0 <= pt[1] < 600:
                cv2.rectangle(canvas, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), (0, 0, 255), -1)

    # Draw trajectory
    if len(trajectory) > 1:
        for i in range(len(trajectory) - 1):
            pt1 = to_canvas_zy(trajectory[i][2], trajectory[i][1])
            pt2 = to_canvas_zy(trajectory[i+1][2], trajectory[i+1][1])
            if (600 <= pt1[0] < 1200 and 0 <= pt1[1] < 600 and
                600 <= pt2[0] < 1200 and 0 <= pt2[1] < 600):
                cv2.line(canvas, pt1, pt2, (0, 200, 0), 2)

    # Current position
    if len(trajectory) > 0:
        pt = to_canvas_zy(trajectory[-1][2], trajectory[-1][1])
        if 600 <= pt[0] < 1200 and 0 <= pt[1] < 600:
            cv2.circle(canvas, pt, 5, (0, 255, 0), -1)

    # Add statistics
    stats = slam_map.get_statistics()
    cv2.putText(canvas, f"KeyFrames: {stats['valid_keyframes']}",
                (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"MapPoints: {stats['valid_mappoints']}",
                (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"Trajectory: {len(trajectory)} poses",
                (600 + 10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow(window_name, canvas)


def main():
    dataset_path = 'data/V1_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Step 6: Tracking Only (without Local Mapping)")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    print(f"\nCamera matrix K:")
    print(loader.K)

    # Create components
    slam_map = Map()
    n_features = 2000
    orb = cv2.ORB_create(nfeatures=n_features)

    # Create Tracking
    tracking = Tracking(slam_map, loader.K, loader.dist_coeffs, {'nfeatures': n_features})

    print(f"\nORB Configuration:")
    print(f"  Max features: {n_features}")

    print("\n" + "=" * 60)
    print("Phase 1: Initialization")
    print("=" * 60)

    # State machine
    state = "NOT_INITIALIZED"
    ref_frame = None
    initializer = None
    frame_skip = 0
    trajectory = []  # Camera positions in world frame

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
            # Initialization phase
            if ref_frame is None:
                ref_frame = current_frame
                # Create initializer once with reference frame
                initializer = Initializer(ref_frame, loader.K, loader.dist_coeffs)
                print(f"\n[Frame {idx}] Set as reference frame")
                print(f"  Features: {len(keypoints)}")
                continue

            # Skip frames for better baseline
            frame_skip += 1
            if frame_skip < 10:
                continue
            frame_skip = 0

            print(f"\n[Frame {idx}] Attempting initialization...")

            # Attempt initialization
            success, result = initializer.initialize(current_frame, ratio_thresh=0.75)

            if success:
                print("\n" + "=" * 60)
                print("INITIALIZATION SUCCESSFUL!")
                print("=" * 60)

                # Create initial KeyFrames
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

                T_c2w = np.eye(4)
                T_c2w[:3, :3] = result['R']
                T_c2w[:3, 3:4] = result['t']

                kf_curr = KeyFrame(
                    frame_id=current_frame['frame_id'],
                    timestamp=current_frame['timestamp'],
                    pose=T_c2w,
                    K=loader.K,
                    dist_coeffs=loader.dist_coeffs,
                    keypoints=current_frame['keypoints'],
                    descriptors=current_frame['descriptors'],
                    image=current_frame['image']
                )

                # Add KeyFrames to map
                slam_map.add_keyframe(kf_ref)
                slam_map.add_keyframe(kf_curr)

                # Create MapPoints
                inlier_matches = [m for i, m in enumerate(result['matches']) if result['mask'][i]]

                # Prepare mappoints list for KeyFrames
                ref_mappoints = [None] * len(ref_frame['keypoints'])
                curr_mappoints = [None] * len(current_frame['keypoints'])

                for i, (point_3d, match) in enumerate(zip(result['points_3d'], inlier_matches)):
                    mp = MapPoint(
                        position=point_3d,
                        ref_keyframe=kf_ref,
                        descriptor=ref_frame['descriptors'][match.queryIdx]
                    )

                    mp.add_observation(kf_ref, match.queryIdx)
                    mp.add_observation(kf_curr, match.trainIdx)

                    kf_ref.add_mappoint(mp, match.queryIdx)
                    kf_curr.add_mappoint(mp, match.trainIdx)

                    ref_mappoints[match.queryIdx] = mp
                    curr_mappoints[match.trainIdx] = mp

                    slam_map.add_mappoint(mp)

                # Update KeyFrame connections
                kf_ref.update_connections()
                kf_curr.update_connections()

                # Prepare reference KeyFrame data for tracking
                ref_kf_data = {
                    'frame_id': kf_ref.frame_id,
                    'timestamp': kf_ref.timestamp,
                    'keypoints': kf_ref.keypoints,
                    'descriptors': kf_ref.descriptors,
                    'image': kf_ref.image,
                    'pose': kf_ref.T_cw.copy(),
                    'mappoints': ref_mappoints
                }

                # Initialize tracking
                tracking.set_initialized(ref_kf_data, T_c2w.copy())

                # Add initial trajectory points
                trajectory.append(kf_ref.get_camera_center().ravel())
                trajectory.append(kf_curr.get_camera_center().ravel())

                stats = slam_map.get_statistics()
                print(f"\nInitial Map Statistics:")
                print(f"  KeyFrames: {stats['valid_keyframes']}")
                print(f"  MapPoints: {stats['valid_mappoints']}")
                print(f"  Parallax: {result['parallax']:.2f} degrees")
                print(f"  Model: {result['model']}")

                print("\n" + "=" * 60)
                print("Phase 2: Tracking")
                print("=" * 60)

                state = "TRACKING"

        elif state == "TRACKING":
            # Track frame
            success, pose = tracking.track_frame(img, timestamp, idx)

            if success:
                # Get camera center in world frame
                R = pose[:3, :3]
                t = pose[:3, 3]
                camera_center = -R.T @ t
                trajectory.append(camera_center)

                n_tracked = tracking.get_tracked_mappoints_count()
                print(f"[Frame {idx}] Tracked: {n_tracked} points | "
                      f"Pos: [{camera_center[0]:.2f}, {camera_center[1]:.2f}, {camera_center[2]:.2f}]")

                # Visualize
                img_vis = visualize_tracking(
                    img, keypoints, tracking.current_frame['mappoints'],
                    tracking.get_state(), n_tracked, pose
                )
                cv2.imshow("Tracking", img_vis)

                visualize_trajectory_3d(slam_map, trajectory, "Trajectory")

            else:
                print(f"[Frame {idx}] Tracking failed!")

                # Visualize lost state
                img_vis = visualize_tracking(
                    img, keypoints, [None] * len(keypoints),
                    tracking.get_state(), 0, None
                )
                cv2.imshow("Tracking", img_vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("\n" + "=" * 60)
    print(f"Tracking finished!")
    print(f"  Total trajectory points: {len(trajectory)}")
    print("=" * 60)

    print("\nPress any key to exit...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
