"""
Step 2: Feature Matching between Consecutive Frames
Extracts ORB features from two frames and matches them using descriptor distance
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.euroc_loader import EuRoCLoader


def match_features(desc1, desc2, ratio_thresh=0.75):
    """
    Match features using Brute-Force matcher with ratio test

    Args:
        desc1: Descriptors from first image (N1, 32)
        desc2: Descriptors from second image (N2, 32)
        ratio_thresh: Lowe's ratio test threshold

    Returns:
        good_matches: List of cv2.DMatch objects
    """
    # BFMatcher with Hamming distance (for binary descriptors like ORB)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False) # NORM_HAMMING : XOR

    # KNN match with k=2 (find 2 best matches)
    matches = bf.knnMatch(desc1, desc2, k=2) # 2 best matches for each descriptor1

    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < ratio_thresh * n.distance:
                good_matches.append(m)

    return good_matches


def visualize_matches(img1, kp1, img2, kp2, matches, title="Feature Matches"):
    """
    Visualize feature matches between two images

    Args:
        img1, img2: Grayscale images
        kp1, kp2: Keypoints
        matches: List of cv2.DMatch
        title: Window title
    """
    # Draw matches
    img_matches = cv2.drawMatches(
        img1, kp1, img2, kp2, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Add statistics
    text = f"Matches: {len(matches)}"
    cv2.putText(img_matches, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow(title, img_matches)


def main():
    # Dataset path
    dataset_path = 'data/MH_01_easy'

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please create a symbolic link:")
        print("  ln -sf ~/MH_01_easy data/MH_01_easy")
        return

    print("=" * 60)
    print("Step 2: Feature Matching Demo")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    # Create ORB detector
    n_features = 1000
    orb = cv2.ORB_create(nfeatures=n_features)

    print(f"\nORB Configuration:")
    print(f"  Max features: {n_features}")
    print(f"  Scale levels: {orb.getNLevels()}")
    print(f"  Scale factor: {orb.getScaleFactor()}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  SPACE - Next frame pair")
    print("  'q'   - Quit")
    print("=" * 60)

    # Process consecutive frame pairs
    prev_img = None
    prev_kp = None
    prev_desc = None
    prev_idx = None

    for img, timestamp, idx in loader:
        if img is None:
            continue

        # Extract features from current image
        keypoints, descriptors = orb.detectAndCompute(img, None)

        if prev_img is not None:
            print(f"\n" + "=" * 60)
            print(f"Frame pair: {prev_idx} <-> {idx}")
            print(f"  Frame {prev_idx}: {len(prev_kp)} features")
            print(f"  Frame {idx}: {len(keypoints)} features")

            # Match features
            if prev_desc is not None and descriptors is not None:
                matches = match_features(prev_desc, descriptors, ratio_thresh=0.75)

                print(f"  Matches found: {len(matches)}")

                if len(matches) > 0:
                    # Statistics
                    distances = [m.distance for m in matches]
                    print(f"  Distance: min={min(distances):.1f}, max={max(distances):.1f}, mean={np.mean(distances):.1f}")

                    # Visualize
                    visualize_matches(prev_img, prev_kp, img, keypoints, matches,
                                    "Feature Matches - EuRoC")
                else:
                    print("  No matches found!")

            # Wait for key
            key = cv2.waitKey(0)

            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord(' '):
                pass  # Continue to next pair

        # Update previous frame
        prev_img = img.copy()
        prev_kp = keypoints
        prev_desc = descriptors.copy() if descriptors is not None else None
        prev_idx = idx

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
