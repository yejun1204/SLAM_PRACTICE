# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

A monocular visual SLAM implementation in Python, inspired by ORB-SLAM3. Uses EuRoC MAV dataset. Step-by-step tutorial examples build up to a full two-threaded SLAM system.

## Running the Code

```bash
# Install dependencies
pip install opencv-python numpy scipy matplotlib pyyaml

# Run step-by-step tutorials
python examples/step1_orb_extraction.py
python examples/step6_tracking_only.py
python examples/step7_local_mapping.py   # Full SLAM with local mapping
```

Dataset symlinks are pre-configured (`data/MH_01_easy`, `data/V1_01_easy`), pointing to `~/MH_01_easy` and `~/V1_01_easy`.

## Architecture

Two-threaded SLAM pipeline:

```
EuRoCLoader → Tracking → [KeyFrame queue] → LocalMapping
                  ↕                               ↕
                       Map (KeyFrames + MapPoints)
```

**Tracking** (`src/tracking.py`): Per-frame pose estimation. State machine: `NOT_INITIALIZED → OK`. Uses ORB features + BFMatcher (Hamming). Pose strategies: (1) constant velocity motion model, (2) reference KeyFrame fallback. Decides when to insert KeyFrames.

**LocalMapping** (`src/local_mapping.py`): Processes new KeyFrames. Triangulates new MapPoints, culls bad ones, runs Local Bundle Adjustment (LBA) via `scipy.optimize.least_squares` with Huber loss.

**Initializer** (`src/initializer.py`): Two-frame monocular initialization. Scores Homography (planar scenes) vs Fundamental Matrix (general scenes) and picks the best model.

**Map** (`src/map.py`): Thread-safe (RLock) container for all KeyFrames and MapPoints.

**KeyFrame** (`src/keyframe.py`): Stores pose (T_cw), ORB keypoints/descriptors, MapPoint associations, and covisibility graph connections.

**MapPoint** (`src/map_point.py`): 3D world point with observations (KeyFrame → keypoint index), visibility stats, and representative descriptor.

## Coordinate System Convention

**T_cw** = transformation from world to camera frame: `T_cw = [R_cw | t_cw]`

Camera center in world coordinates: `-R_cw.T @ t_cw`

See `docs/coordinate_systems.md` for full details.

## Key Implementation Notes

- ORB: 2000 features/frame, BFMatcher with Hamming distance, Lowe's ratio test at 0.75
- MapPoint culling: bad if found_ratio < 0.25 or too few observations in first 2 KFs
- Triangulation thresholds: min parallax 1.0°, reprojection error ≤ 5.991 px (χ² 2DOF)
- LBA fixes the origin KeyFrame to resolve gauge freedom
- `examples/` scripts are numbered and increase in complexity — step7 is the current complete system
