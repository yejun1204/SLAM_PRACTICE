"""
Step 5: Map Data Structure - MapPoint, KeyFrame, Map
Basic data structures for SLAM
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.euroc_loader import EuRoCLoader
from src.map_point import MapPoint
from src.keyframe import KeyFrame
from src.map import Map


def visualize_map(slam_map, window_name="Map Visualization"):
    """
    Visualize the 3D map in 2D projections (X-Z and Z-Y planes)

    Args:
        slam_map: Map object
        window_name: Window name for display
    """
    mappoints = slam_map.get_valid_mappoints()
    keyframes = slam_map.get_all_keyframes()

    if len(mappoints) == 0 and len(keyframes) == 0:
        print("Empty map, nothing to visualize")
        return

    # Create canvas
    canvas = np.ones((600, 1200, 3), dtype=np.uint8) * 255

    margin = 50
    plot_width = 600 - 2 * margin
    plot_height = 600 - 2 * margin - 50

    # ===== Extract 3D positions =====
    mp_positions = np.array([mp.get_position().ravel() for mp in mappoints])
    kf_positions = np.array([kf.get_camera_center().ravel() for kf in keyframes])

    all_positions = np.vstack([mp_positions, kf_positions]) if len(mp_positions) > 0 and len(kf_positions) > 0 else \
                    mp_positions if len(mp_positions) > 0 else kf_positions

    if len(all_positions) == 0:
        return

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

    # Draw KeyFrame positions (red squares)
    for kf_pos in kf_positions:
        pt = to_canvas_xz(kf_pos[0], kf_pos[2])
        cv2.rectangle(canvas, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), (0, 0, 255), -1)

    # Draw trajectory connections
    if len(kf_positions) > 1:
        for i in range(len(kf_positions) - 1):
            pt1 = to_canvas_xz(kf_positions[i, 0], kf_positions[i, 2])
            pt2 = to_canvas_xz(kf_positions[i+1, 0], kf_positions[i+1, 2])
            cv2.line(canvas, pt1, pt2, (0, 200, 0), 1)

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
    for kf_pos in kf_positions:
        pt = to_canvas_zy(kf_pos[2], kf_pos[1])
        cv2.rectangle(canvas, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), (0, 0, 255), -1)

    # Draw trajectory
    if len(kf_positions) > 1:
        for i in range(len(kf_positions) - 1):
            pt1 = to_canvas_zy(kf_positions[i, 2], kf_positions[i, 1])
            pt2 = to_canvas_zy(kf_positions[i+1, 2], kf_positions[i+1, 1])
            cv2.line(canvas, pt1, pt2, (0, 200, 0), 1)

    # Add statistics
    stats = slam_map.get_statistics()
    cv2.putText(canvas, f"KeyFrames: {stats['valid_keyframes']}",
                (10, 560), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(canvas, f"MapPoints: {stats['valid_mappoints']}",
                (10, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Display
    cv2.imshow(window_name, canvas)


def triangulate_points(R1, t1, R2, t2, pts1, pts2, K):
    """
    Triangulate 3D points from two views

    Args:
        R1, t1: Camera 1 pose (world to cam1)
        R2, t2: Camera 2 pose (world to cam2)
        pts1, pts2: 2D point correspondences (Nx2)
        K: Camera intrinsic matrix

    Returns:
        points_3d: Triangulated 3D points (Nx3) in world coordinates
        valid_mask: Boolean mask for valid points
    """
    # Construct projection matrices
    P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])

    # Triangulate
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T) # world coordinates

    # Convert from homogeneous to 3D
    points_3d = points_4d[:3, :] / points_4d[3, :]
    points_3d = points_3d.T  # (N, 3)

    # Check for positive depth
    # Camera 1 좌표계로 변환
    points_3d_cam1 = (R1 @ points_3d.T + t1.reshape(-1, 1)).T
    depths1 = points_3d_cam1[:, 2]

    # Camera 2 좌표계로 변환
    points_3d_cam2 = (R2 @ points_3d.T + t2.reshape(-1, 1)).T
    depths2 = points_3d_cam2[:, 2]

    valid_mask = (depths1 > 0) & (depths2 > 0)

    return points_3d, valid_mask


def main():
    dataset_path = 'data/MH_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    print("=" * 60)
    print("Step 5: Map Data Structure Test")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    print(f"\nCamera matrix K:")
    print(loader.K)

    # Create ORB detector
    n_features = 2000
    orb = cv2.ORB_create(nfeatures=n_features)

    # Create SLAM Map
    slam_map = Map()

    print("\n" + "=" * 60)
    print("Creating map from first few frames...")
    print("=" * 60)

    # Extract features from first 5 frames
    frames_data = []
    for img, timestamp, idx in loader:
        if img is None:
            continue

        keypoints, descriptors = orb.detectAndCompute(img, None)
        if descriptors is None:
            continue

        frames_data.append({
            'img': img,
            'timestamp': timestamp,
            'idx': idx,
            'keypoints': keypoints,
            'descriptors': descriptors
        })

        if len(frames_data) >= 5:
            break

    # Create KeyFrames and MapPoints from frame pairs
    current_pose = np.eye(4)  # Start at origin

    for i in range(len(frames_data) - 1):
        frame1 = frames_data[i]
        frame2 = frames_data[i + 1]

        print(f"\n--- Processing frame pair {i} <-> {i+1} ---")

        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(frame1['descriptors'], frame2['descriptors'], k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        print(f"  Matches: {len(good_matches)}")

        if len(good_matches) < 20:
            print("  Not enough matches, skipping...")
            continue

        # Estimate pose
        pts1 = np.float32([frame1['keypoints'][m.queryIdx].pt for m in good_matches])
        pts2 = np.float32([frame2['keypoints'][m.trainIdx].pt for m in good_matches])

        pts1_undist = cv2.undistortPoints(pts1.reshape(-1, 1, 2), loader.K, loader.dist_coeffs, None, loader.K).reshape(-1, 2)
        pts2_undist = cv2.undistortPoints(pts2.reshape(-1, 1, 2), loader.K, loader.dist_coeffs, None, loader.K).reshape(-1, 2)

        E, mask = cv2.findEssentialMat(pts1_undist, pts2_undist, loader.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        _, R, t, mask_pose = cv2.recoverPose(E, pts1_undist, pts2_undist, loader.K, mask=mask)

        inlier_count = np.sum(mask_pose)
        print(f"  Inliers: {inlier_count}/{len(good_matches)}")

        if inlier_count < 20:
            print("  Not enough inliers, skipping...")
            continue

        # Create first KeyFrame (if this is the first pair)
        if i == 0:
            kf1 = KeyFrame(
                frame_id=frame1['idx'],
                timestamp=frame1['timestamp'],
                pose=np.linalg.inv(current_pose),
                K=loader.K,
                dist_coeffs=loader.dist_coeffs,
                keypoints=frame1['keypoints'],
                descriptors=frame1['descriptors'],
                image=frame1['img']
            )
            slam_map.add_keyframe(kf1)
            print(f"  Created KeyFrame: {kf1}")

        # Update pose for second frame
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3, 3:4] = t * 1.0  # Scale factor for visualization
        current_pose = current_pose @ np.linalg.inv(T_rel)

        # Create second KeyFrame
        kf2 = KeyFrame(
            frame_id=frame2['idx'],
            timestamp=frame2['timestamp'],
            pose=np.linalg.inv(current_pose), #T_cw 대입해야 함 
            K=loader.K,
            dist_coeffs=loader.dist_coeffs,
            keypoints=frame2['keypoints'],
            descriptors=frame2['descriptors'],
            image=frame2['img']
        )
        slam_map.add_keyframe(kf2)
        print(f"  Created KeyFrame: {kf2}")

        # Triangulate MapPoints
        inlier_matches = [m for j, m in enumerate(good_matches) if mask_pose[j]]
        pts1_inliers = pts1_undist[mask_pose.ravel().astype(bool)]
        pts2_inliers = pts2_undist[mask_pose.ravel().astype(bool)]

        # Get KeyFrame poses
        if i == 0:
            kf1_pose = slam_map.get_all_keyframes()[0]
        else:
            kf1_pose = slam_map.get_all_keyframes()[-2]
        kf2_pose = slam_map.get_all_keyframes()[-1]

        R1 = kf1_pose.get_rotation()
        t1 = kf1_pose.get_translation().ravel()
        R2 = kf2_pose.get_rotation()
        t2 = kf2_pose.get_translation().ravel()

        points_3d, valid_mask = triangulate_points(R1, t1, R2, t2, pts1_inliers, pts2_inliers, loader.K)

        valid_count = np.sum(valid_mask)
        print(f"  Valid 3D points: {valid_count}/{len(points_3d)}")

        # Create MapPoints and add observations
        # NOTE: 현재는 데이터 구조 테스트용으로 중복 체크 없이 무조건 새로 생성합니다.
        # 실제 SLAM에서는 다음과 같이 중복 체크가 필요합니다:
        #
        # existing_mp = kf1_pose.mappoints[m.queryIdx]
        # if existing_mp is not None:
        #     # 이미 존재하는 MapPoint에 새로운 관측만 추가
        #     existing_mp.add_observation(kf2_pose, m.trainIdx)
        #     kf2_pose.add_mappoint(existing_mp, m.trainIdx)
        # else:
        #     # 새로운 MapPoint 생성
        #
        # 현재 방식의 문제:
        # Frame 0-1: Frame0의 kp[5] + Frame1의 kp[10] → MapPoint_A 생성
        # Frame 1-2: Frame1의 kp[10] + Frame2의 kp[20] → MapPoint_B 생성 (중복!)
        # → Frame1의 kp[10]이 MapPoint_A와 MapPoint_B 두 개와 연결되어 중복됨

        for j, m in enumerate(inlier_matches):
            if not valid_mask[j]:
                continue

            # Create MapPoint
            mp = MapPoint(
                position=points_3d[j],
                ref_keyframe=kf1_pose,
                descriptor=frame1['descriptors'][m.queryIdx]
            )

            # Add observations from both KeyFrames
            mp.add_observation(kf1_pose, m.queryIdx)
            mp.add_observation(kf2_pose, m.trainIdx)

            # Associate MapPoint with KeyFrames
            kf1_pose.add_mappoint(mp, m.queryIdx)
            kf2_pose.add_mappoint(mp, m.trainIdx)

            # Add to map
            slam_map.add_mappoint(mp)

        # Update KeyFrame connections
        kf1_pose.update_connections()
        kf2_pose.update_connections()

    # Print map statistics
    print("\n" + "=" * 60)
    print("Map Statistics:")
    stats = slam_map.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")

    print(f"\nMap: {slam_map}")

    # Test covisibility graph
    print("\n" + "=" * 60)
    print("Covisibility Graph:")
    for kf in slam_map.get_all_keyframes():
        connected = kf.get_connected_keyframes()
        print(f"  {kf.id} connected to: {[k.id for k in connected]}")

    # Visualize map
    print("\n" + "=" * 60)
    print("Press any key to close visualization...")
    visualize_map(slam_map)
    cv2.waitKey(0)

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
