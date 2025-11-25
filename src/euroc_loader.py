"""
EuRoC Dataset Loader
Loads images and camera calibration from EuRoC MAV dataset
"""

import os
import cv2
import numpy as np
import yaml


class EuRoCLoader:
    """Load images and calibration from EuRoC dataset"""

    def __init__(self, dataset_path, camera='cam0'):
        """
        Initialize EuRoC dataset loader

        Args:
            dataset_path: Path to dataset (e.g., data/MH_01_easy)
            camera: Camera to use ('cam0' or 'cam1')
        """
        self.dataset_path = dataset_path
        self.camera = camera
        self.mav0_path = os.path.join(dataset_path, 'mav0')
        self.cam_path = os.path.join(self.mav0_path, camera)

        # Load camera calibration
        self.load_calibration()

        # Load image timestamps and filenames
        self.load_image_list()

        self.current_idx = 0

    def load_calibration(self):
        """Load camera calibration from sensor.yaml"""
        calib_file = os.path.join(self.cam_path, 'sensor.yaml')

        with open(calib_file, 'r') as f:
            calib = yaml.safe_load(f)

        # Camera intrinsics: [fu, fv, cu, cv]
        intrinsics = calib['intrinsics']
        self.fx = intrinsics[0]
        self.fy = intrinsics[1]
        self.cx = intrinsics[2]
        self.cy = intrinsics[3]

        # Camera matrix K
        self.K = np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float32)

        # Distortion coefficients
        self.dist_coeffs = np.array(calib['distortion_coefficients'], dtype=np.float32)

        # Resolution
        self.resolution = calib['resolution']  # [width, height]

        print(f"Loaded calibration for {self.camera}:")
        print(f"  Resolution: {self.resolution}")
        print(f"  Intrinsics: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        print(f"  Distortion: {self.dist_coeffs}")

    def load_image_list(self):
        """Load list of image timestamps and filenames"""
        data_csv = os.path.join(self.cam_path, 'data.csv')

        self.timestamps = []
        self.image_files = []

        with open(data_csv, 'r') as f:
            # Skip header
            next(f)

            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Format: timestamp,filename
                parts = line.split(',')
                timestamp = int(parts[0])
                filename = parts[1]

                self.timestamps.append(timestamp)
                self.image_files.append(filename)

        print(f"Loaded {len(self.timestamps)} images from {self.camera}")

    def get_image(self, idx):
        """
        Get image at index

        Args:
            idx: Image index

        Returns:
            img: Grayscale image
            timestamp: Timestamp in nanoseconds
        """
        if idx < 0 or idx >= len(self.image_files):
            return None, None

        img_path = os.path.join(self.cam_path, 'data', self.image_files[idx])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            return None, None

        timestamp = self.timestamps[idx]

        return img, timestamp

    def __len__(self):
        """Return number of images"""
        return len(self.image_files)

    def __iter__(self):
        """Iterator interface"""
        self.current_idx = 0
        return self

    def __next__(self):
        """Get next image"""
        if self.current_idx >= len(self):
            raise StopIteration

        img, timestamp = self.get_image(self.current_idx)
        self.current_idx += 1

        return img, timestamp, self.current_idx - 1


if __name__ == '__main__':
    # Test the loader with distortion comparison
    loader = EuRoCLoader('data/MH_01_easy')

    print(f"\nCamera matrix K:\n{loader.K}")
    print(f"\nDistortion coefficients: {loader.dist_coeffs}")
    print(f"\nTotal images: {len(loader)}")

    print("\nPress SPACE for next image, 'q' to quit")
    print("Left: Original (with distortion) | Right: Undistorted")

    # Iterate through images using for loop
    for img, timestamp, idx in loader:
        if img is None:
            print(f"Failed to load image {idx}")
            continue

        print(f"\nImage {idx}/{len(loader)-1}")
        print(f"  Shape: {img.shape}")
        print(f"  Timestamp: {timestamp}")

        # Undistort the entire image
        img_undistorted = cv2.undistort(img, loader.K, loader.dist_coeffs)

        # Concatenate side by side
        comparison = np.hstack([img, img_undistorted])

        # Add labels
        cv2.putText(comparison, "Original", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Undistorted", (img.shape[1] + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Display
        cv2.imshow('Distortion Comparison', comparison)

        key = cv2.waitKey(0)
        if key == ord('q'):
            print("Quitting...")
            break

    cv2.destroyAllWindows()
