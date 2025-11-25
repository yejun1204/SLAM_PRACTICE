# SLAM_PRACTICE

A simplified monocular visual SLAM implementation based on ORB-SLAM3, using Python with multi-threading.

## Project Structure

```
SLAM_PRACTICE/
├── src/                    # Source code
│   ├── frame.py           # Frame data structure
│   ├── map_point.py       # MapPoint data structure
│   ├── keyframe.py        # KeyFrame data structure
│   ├── map.py             # Map management
│   ├── orb_extractor.py   # ORB feature extraction wrapper
│   ├── tracking.py        # Tracking thread
│   ├── local_mapping.py   # Local mapping thread
│   └── system.py          # Main system
├── examples/               # Example programs
├── config/                 # Configuration files
├── data/                   # Dataset (EuRoC, etc.)
└── reference/              # ORB-SLAM3 reference code
```

## Features

- Monocular camera tracking
- ORB feature extraction
- Multi-threaded architecture (Tracking + Local Mapping)
- Local Bundle Adjustment
- OpenCV-based visualization

## Dependencies

- Python 3.8+
- OpenCV (cv2)
- NumPy
- SciPy
- Matplotlib

## Installation

```bash
pip install opencv-python numpy scipy matplotlib
```

## Dataset

This project uses the EuRoC MAV dataset:
- **Dataset**: MH_01_easy (Machine Hall, easy difficulty)
- **Location**: `~/MH_01_easy/mav0/`
- **Images**: 3682 frames (752x480, 20Hz)
- **Camera**: cam0 (monocular)
- **Download**: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

To link the dataset to the project:
```bash
ln -sf ~/MH_01_easy data/MH_01_easy
```

## Usage

```bash
# Run with EuRoC dataset
python examples/mono_euroc.py --dataset data/MH_01_easy

# With visualization
python examples/mono_euroc.py --dataset data/MH_01_easy --visualize
```

## Reference

This implementation is inspired by ORB-SLAM3:
- Repository: https://github.com/UZ-SLAMLab/ORB_SLAM3
