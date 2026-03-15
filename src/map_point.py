"""
MapPoint class - Represents a 3D point in the map
"""

import numpy as np
import threading


class MapPoint:
    """
    3D point in the map with observations from multiple keyframes
    """

    # Class variable for ID generation
    _next_id = 0
    _id_lock = threading.Lock()

    def __init__(self, position, ref_keyframe, descriptor=None):
        """
        Initialize a MapPoint

        Args:
            position: 3D position in world coordinates (3,) or (3,1)
            ref_keyframe: Reference KeyFrame that first observed this point
            descriptor: ORB descriptor (optional)
        """
        # Generate unique ID
        with MapPoint._id_lock:
            self.id = MapPoint._next_id
            MapPoint._next_id += 1

        # 3D position in world coordinates
        self.position = np.array(position).reshape(3, 1)

        # Reference KeyFrame (the KeyFrame that created this MapPoint)
        self.ref_keyframe = ref_keyframe

        # ORB descriptor (best descriptor among observations)
        self.descriptor = descriptor

        # Observations: dict of {keyframe: keypoint_idx}
        # Maps KeyFrame objects to the index of the keypoint in that KeyFrame
        self.observations = {}

        # Track viewing statistics
        self.n_visible = 0  # Number of times this point was in frame
        self.n_found = 0    # Number of times this point was matched

        # Bad flag (for culling)
        self.is_bad = False

        # Thread safety (RLock allows reentrant locking from same thread)
        self.lock = threading.RLock()

    def get_position(self):
        """Get 3D position (thread-safe)"""
        with self.lock:
            return self.position.copy()

    def set_position(self, position):
        """Set 3D position (thread-safe)"""
        with self.lock:
            self.position = np.array(position).reshape(3, 1)

    def add_observation(self, keyframe, keypoint_idx):
        """
        Add an observation from a KeyFrame

        Args:
            keyframe: KeyFrame that observes this MapPoint
            keypoint_idx: Index of the keypoint in the KeyFrame
        """
        with self.lock:
            if keyframe not in self.observations:
                self.observations[keyframe] = keypoint_idx

    def remove_observation(self, keyframe):
        """Remove an observation from a KeyFrame"""
        with self.lock:
            if keyframe in self.observations:
                del self.observations[keyframe]

                # If this was the reference keyframe, choose a new one
                if keyframe == self.ref_keyframe and len(self.observations) > 0:
                    self.ref_keyframe = list(self.observations.keys())[0]

    def get_observations(self):
        """Get all observations (thread-safe copy)"""
        with self.lock:
            return self.observations.copy()

    def num_observations(self):
        """Get number of observations"""
        with self.lock:
            return len(self.observations)

    def set_bad_flag(self):
        """Mark this MapPoint as bad (for removal)"""
        with self.lock:
            self.is_bad = True
            # Remove this MapPoint from all observing KeyFrames
            for kf, idx in self.observations.items():
                kf.remove_mappoint(idx)
            self.observations.clear()

    def is_bad_point(self):
        """Check if this MapPoint is bad"""
        with self.lock:
            return self.is_bad

    def update_descriptor(self):
        """
        Update the representative descriptor by computing median of all observed descriptors
        """
        with self.lock:
            if len(self.observations) == 0:
                return

            # Collect all descriptors from observations
            descriptors = []
            for kf, idx in self.observations.items():
                if idx < len(kf.descriptors):
                    descriptors.append(kf.descriptors[idx])

            if len(descriptors) == 0:
                return

            # Compute distances between all pairs
            descriptors = np.array(descriptors)
            n = len(descriptors)

            # Find descriptor with minimum median distance to others
            min_median_dist = float('inf')
            best_desc = None

            for i in range(n):
                distances = []
                for j in range(n):
                    if i != j:
                        # Hamming distance for binary descriptors
                        dist = np.sum(np.unpackbits(descriptors[i] ^ descriptors[j]))
                        distances.append(dist)

                median_dist = np.median(distances)
                if median_dist < min_median_dist:
                    min_median_dist = median_dist
                    best_desc = descriptors[i]

            self.descriptor = best_desc

    def get_found_ratio(self):
        """
        Get the ratio of found/visible
        Used for culling low-quality points
        """
        with self.lock:
            if self.n_visible == 0:
                return 0.0
            return float(self.n_found) / float(self.n_visible)

    def increase_visible(self, n=1):
        """Increase visible counter"""
        with self.lock:
            self.n_visible += n

    def increase_found(self, n=1):
        """Increase found counter"""
        with self.lock:
            self.n_found += n

    def __repr__(self):
        """String representation"""
        pos = self.position.ravel()
        return f"MapPoint(id={self.id}, pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}], obs={self.num_observations()})"
