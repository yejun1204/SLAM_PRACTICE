"""
Map class - Container for MapPoints and KeyFrames
"""

import threading
import numpy as np


class Map:
    """
    Map - Container for all MapPoints and KeyFrames
    Manages the global map structure
    """

    def __init__(self):
        """Initialize empty map"""
        # All MapPoints in the map
        self.mappoints = []

        # All KeyFrames in the map
        self.keyframes = []

        # Reference KeyFrame (first KeyFrame in the map)
        self.reference_keyframe = None

        # Maximum KeyFrame ID (for ordering)
        self.max_keyframe_id = 0

        # Thread safety
        self.lock = threading.Lock()

    def add_mappoint(self, mappoint):
        """
        Add a MapPoint to the map

        Args:
            mappoint: MapPoint object
        """
        with self.lock:
            if mappoint not in self.mappoints:
                self.mappoints.append(mappoint)

    def add_keyframe(self, keyframe):
        """
        Add a KeyFrame to the map

        Args:
            keyframe: KeyFrame object
        """
        with self.lock:
            if keyframe not in self.keyframes:
                self.keyframes.append(keyframe)

                # Update reference KeyFrame (first KeyFrame)
                if self.reference_keyframe is None:
                    self.reference_keyframe = keyframe

                # Update max ID
                if keyframe.id > self.max_keyframe_id:
                    self.max_keyframe_id = keyframe.id

    def remove_mappoint(self, mappoint):
        """
        Remove a MapPoint from the map

        Args:
            mappoint: MapPoint object
        """
        with self.lock:
            if mappoint in self.mappoints:
                self.mappoints.remove(mappoint)

    def remove_keyframe(self, keyframe):
        """
        Remove a KeyFrame from the map

        Args:
            keyframe: KeyFrame object
        """
        with self.lock:
            if keyframe in self.keyframes:
                self.keyframes.remove(keyframe)

                # Update reference KeyFrame if needed
                if keyframe == self.reference_keyframe:
                    if len(self.keyframes) > 0:
                        self.reference_keyframe = self.keyframes[0]
                    else:
                        self.reference_keyframe = None

    def get_all_mappoints(self):
        """Get all MapPoints (thread-safe copy)"""
        with self.lock:
            return self.mappoints.copy()

    def get_all_keyframes(self):
        """Get all KeyFrames (thread-safe copy)"""
        with self.lock:
            return self.keyframes.copy()

    def get_reference_keyframe(self):
        """Get reference KeyFrame"""
        with self.lock:
            return self.reference_keyframe

    def set_reference_keyframe(self, keyframe):
        """Set reference KeyFrame"""
        with self.lock:
            self.reference_keyframe = keyframe

    def num_mappoints(self):
        """Get total number of MapPoints"""
        with self.lock:
            return len(self.mappoints)

    def num_keyframes(self):
        """Get total number of KeyFrames"""
        with self.lock:
            return len(self.keyframes)

    def get_valid_mappoints(self):
        """
        Get all valid (non-bad) MapPoints

        Returns:
            List of valid MapPoint objects
        """
        with self.lock:
            return [mp for mp in self.mappoints if not mp.is_bad_point()]

    def cull_bad_mappoints(self):
        """
        Remove all bad MapPoints from the map
        Should be called periodically by Local Mapping
        """
        with self.lock:
            # Filter out bad MapPoints
            self.mappoints = [mp for mp in self.mappoints if not mp.is_bad_point()]

    def cull_bad_keyframes(self):
        """
        Remove all bad KeyFrames from the map
        Should be called periodically by Local Mapping
        """
        with self.lock:
            # Filter out bad KeyFrames
            self.keyframes = [kf for kf in self.keyframes if not kf.is_bad_keyframe()]

    def clear(self):
        """Clear the entire map"""
        with self.lock:
            self.mappoints.clear()
            self.keyframes.clear()
            self.reference_keyframe = None
            self.max_keyframe_id = 0

    def get_bounding_box(self):
        """
        Get axis-aligned bounding box of all MapPoints

        Returns:
            min_coords: (3,) array of minimum coordinates [x, y, z]
            max_coords: (3,) array of maximum coordinates [x, y, z]
        """
        valid_points = self.get_valid_mappoints()

        if len(valid_points) == 0:
            return np.zeros(3), np.zeros(3)

        positions = np.array([mp.get_position().ravel() for mp in valid_points])

        min_coords = positions.min(axis=0)
        max_coords = positions.max(axis=0)

        return min_coords, max_coords

    def get_statistics(self):
        """
        Get map statistics

        Returns:
            Dictionary with map statistics
        """
        with self.lock:
            total_mp = len(self.mappoints)
            valid_mp = len([mp for mp in self.mappoints if not mp.is_bad_point()])
            total_kf = len(self.keyframes)
            valid_kf = len([kf for kf in self.keyframes if not kf.is_bad_keyframe()])

            # Average observations per MapPoint
            avg_obs = 0.0
            if valid_mp > 0:
                total_obs = sum(mp.num_observations() for mp in self.mappoints if not mp.is_bad_point())
                avg_obs = total_obs / valid_mp

            # Average MapPoints per KeyFrame
            avg_mp_per_kf = 0.0
            if valid_kf > 0:
                total_mp_in_kf = sum(kf.num_mappoints() for kf in self.keyframes if not kf.is_bad_keyframe())
                avg_mp_per_kf = total_mp_in_kf / valid_kf

            return {
                'total_mappoints': total_mp,
                'valid_mappoints': valid_mp,
                'total_keyframes': total_kf,
                'valid_keyframes': valid_kf,
                'avg_observations_per_mappoint': avg_obs,
                'avg_mappoints_per_keyframe': avg_mp_per_kf
            }

    def __repr__(self):
        """String representation"""
        stats = self.get_statistics()
        return (f"Map(KeyFrames={stats['valid_keyframes']}/{stats['total_keyframes']}, "
                f"MapPoints={stats['valid_mappoints']}/{stats['total_mappoints']})")
