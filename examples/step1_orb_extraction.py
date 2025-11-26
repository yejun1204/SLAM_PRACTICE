"""
Step 1: ORB Feature Extraction and Visualization
Loads EuRoC images and extracts ORB features
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import cv2
import numpy as np
from src.euroc_loader import EuRoCLoader


def visualize_features(img, keypoints, title="ORB Features"):
    """
    Visualize ORB keypoints on image

    Args:
        img: Grayscale image
        keypoints: List of cv2.KeyPoint
        title: Window title
    """
    # Draw keypoints
    img_with_kp = cv2.drawKeypoints(
        img, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Add text info
    text = f"Features: {len(keypoints)}"
    cv2.putText(img_with_kp, text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display
    cv2.imshow(title, img_with_kp)


def main():
    # Dataset path
    dataset_path = 'data/MH_01_easy'

    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        print("Please create a symbolic link:")
        print("  ln -sf ~/MH_01_easy data/MH_01_easy")
        return

    print("=" * 60)
    print("Step 1: ORB Feature Extraction Demo")
    print("=" * 60)

    # Load dataset
    loader = EuRoCLoader(dataset_path)

    # Create ORB detector
    # ORB-SLAM3 uses 1000-2000 features per frame
    n_features = 1000
    orb = cv2.ORB_create(nfeatures=n_features)

    print(f"\nORB Configuration:")
    print(f"  Max features: {n_features}")
    print(f"  Scale levels: {orb.getNLevels()}")
    print(f"  Scale factor: {orb.getScaleFactor()}")

    print("\n" + "=" * 60)
    print("Controls:")
    print("  SPACE - Next image")
    print("  'q'   - Quit")
    print("=" * 60)

    # Process images using iterator (for loop)
    for img, timestamp, idx in loader:
        if img is None:
            print(f"Failed to load image {idx}")
            continue

        # Extract ORB features
        keypoints, descriptors = orb.detectAndCompute(img, None) # (img, mask=None)

        print(f"\nImage {idx}/{len(loader)-1}")
        print(f"  Timestamp: {timestamp}")
        print(f"  Features detected: {len(keypoints)}")

        if len(keypoints) > 0:
            # Get feature statistics
            responses = [kp.response for kp in keypoints]
            sizes = [kp.size for kp in keypoints]

            print(f"  Response: min={min(responses):.6f}, max={max(responses):.6f}, mean={np.mean(responses):.6f}")
            print(f"  Size: min={min(sizes):.2f}, max={max(sizes):.2f}, mean={np.mean(sizes):.2f}")

        # Visualize
        visualize_features(img, keypoints, "ORB Features - EuRoC")

        # Wait for key
        key = cv2.waitKey(0)

        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord(' '):
            continue  # Next image (automatic in for loop)

    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == '__main__':
    main()
