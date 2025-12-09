# SLAM 좌표계 및 변환 표기법 정리

## 1. 기본 표기법 (핵심 원칙)

### 1.1 변환 행렬 표기

**T_AB**: B 좌표계의 점을 A 좌표계로 변환

```
p_A = T_AB @ p_B
```

**예시:**
- `T_cw`: World 좌표를 Camera 좌표로 변환
  - `p_camera = R_cw @ p_world + t_cw`
- `T_wc`: Camera 좌표를 World 좌표로 변환
  - `p_world = R_wc @ p_camera + t_wc`
- `T_21`: Camera1 좌표를 Camera2 좌표로 변환
  - `p_cam2 = R_21 @ p_cam1 + t_21`

**관계식:**
```
T_wc = inv(T_cw)
T_12 = inv(T_21)
```

### 1.2 Pose 표현

```
T_cw = [R_cw | t_cw]  (4x4 homogeneous matrix)
       [ 0   |   1 ]

where:
  R_cw: 3x3 rotation matrix (world to camera)
  t_cw: 3x1 translation vector
```

**KeyFrame 클래스:**
- 저장하는 pose: `T_cw` (World 좌표를 Camera 좌표로 변환)
- Camera center: `C = -R_cw.T @ t_cw` (world 좌표계에서의 카메라 위치)

---

## 2. OpenCV 함수 반환값

### 2.1 cv2.recoverPose()

```python
retval, R, t, mask = cv2.recoverPose(E, points1, points2, cameraMatrix)
```

**반환값:**
- `R, t`: **T_21** (Camera1 좌표를 Camera2 좌표로 변환)

**수식:**
```
p_cam2 = R @ p_cam1 + t
```

**Initialization 적용 (Camera1 = World):**
```python
# Camera1 = World로 설정
T_c1w = np.eye(4)  # Camera1이 world 좌표계

# recoverPose가 반환한 R, t는 T_21 (Camera1 → Camera2)
# Camera1 = World이므로, T_21 = T_c2w
T_c2w = np.eye(4)
T_c2w[:3, :3] = R
T_c2w[:3, 3:4] = t
```

### 2.2 cv2.triangulatePoints()

```python
points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
```

**입력:**
- `P1 = K @ [R1 | t1]`: World 좌표를 Camera1로 변환하는 투영 행렬 (T_c1w)
- `P2 = K @ [R2 | t2]`: World 좌표를 Camera2로 변환하는 투영 행렬 (T_c2w)

**출력:**
- `points_4d`: World 좌표계의 3D 점 (homogeneous coordinates)

**사용 예시:**
```python
def triangulate(R1, t1, R2, t2, pts1, pts2, K):
    """
    Args:
        R1, t1: Camera1의 T_c1w (world 점을 cam1로 변환)
        R2, t2: Camera2의 T_c2w (world 점을 cam2로 변환)
    Returns:
        points_3d: World 좌표계의 3D 점
    """
    P1 = K @ np.hstack([R1, t1.reshape(-1, 1)])
    P2 = K @ np.hstack([R2, t2.reshape(-1, 1)])
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T
```

---

## 3. 자주 사용하는 패턴

### 3.1 T_cw ↔ T_wc 변환
```python
T_wc = np.linalg.inv(T_cw)
T_cw = np.linalg.inv(T_wc)
```

### 3.2 Camera Center 계산
```python
# T_cw에서 camera center 추출
R_cw = T_cw[:3, :3]
t_cw = T_cw[:3, 3:4]
C = -R_cw.T @ t_cw  # Camera center in world coordinates

# 또는 T_wc에서 직접
C = T_wc[:3, 3:4]
```

### 3.3 Pose 연속 업데이트 (Step5 패턴)
```python
# T_wc 추적 (Camera to World)
current_pose = np.eye(4)  # T_wc_0

for i in range(n_frames - 1):
    # recoverPose: T_21 반환 (Camera_i → Camera_{i+1})
    _, R, t, mask = cv2.recoverPose(E, pts_i, pts_i_plus_1, K)

    T_21 = np.eye(4)
    T_21[:3, :3] = R
    T_21[:3, 3:4] = t

    # T_wc_{i+1} = T_wc_i @ inv(T_21)
    # inv(T_21) = T_12 (Camera_{i+1} → Camera_i 변환)
    T_12 = np.linalg.inv(T_21)
    current_pose = current_pose @ T_12

    # KeyFrame 생성 시 T_cw 필요
    T_cw = np.linalg.inv(current_pose)
    kf = KeyFrame(pose=T_cw, ...)
```

**수식 유도:**
```
목표: T_wc_{i+1} 계산

관계식:
- T_wc_i: World → Camera_i
- T_21: Camera_i → Camera_{i+1}
- T_12 = inv(T_21): Camera_{i+1} → Camera_i

변환 체인:
World → Camera_i → Camera_{i+1}
      (T_wc_i)      (inv(T_12))

따라서:
T_wc_{i+1} = T_wc_i @ T_12 = T_wc_i @ inv(T_21)
```

---

## 4. 변환 체인 규칙


여러 좌표계를 거치는 경우, 변환 행렬을 왼쪽에서 오른쪽으로 곱하면서 중간 인덱스가 소거됩니다:

```
T_AC = T_AB @ T_BC

예시:
- A = World, B = Camera1, C = Camera2
- T_w2 = T_w1 @ T_12
  (World → Cam2) = (World → Cam1) @ (Cam1 → Cam2)
```
