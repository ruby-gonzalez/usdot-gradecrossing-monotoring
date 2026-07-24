"""REFERENCE CODE:https://github.com/AdiGhadge/Video-Stabilization-Using-Lucas-Kanade-Method/blob/main/Video%20Stabilization/Stabilization.py"""

import numpy as np
import cv2
from pathlib import Path
import argparse
def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

SMOOTHING_RADIUS = 50


class ROIStabilizer:
    """Keep an ROI polygon locked onto the scene under camera motion.

    Reference-based, so it is drift-free: every incoming frame is matched
    *directly* against a single reference frame (the one the ROI polygon was
    derived from) with Lucas-Kanade optical flow. The estimated affine is
    applied to the polygon so the ROI follows the rails wherever camera shake
    pushes them. Because the match is always against the fixed reference, shake
    (oscillation about a mean position) snaps the polygon straight back instead
    of accumulating error.

    On a static camera, feature-tracking noise still yields a tiny non-identity
    affine, which would make a fixed ROI wobble. The dead-zone guard collapses
    any sub-threshold motion to identity, so the stabilizer is inert on
    non-shaking footage and only engages under genuine motion.
    """

    def __init__(self, ref_frame, zone, max_corners=200, quality_level=0.01,
                 min_distance=30, block_size=3, min_tracked=12,
                 translation_deadzone=1.0, rotation_deadzone=0.002):
        self.zone = np.asarray(zone, dtype=np.float32)
        self.ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        self.ref_pts = cv2.goodFeaturesToTrack(
            self.ref_gray, maxCorners=max_corners, qualityLevel=quality_level,
            minDistance=min_distance, blockSize=block_size)
        self.min_tracked = min_tracked
        self.translation_deadzone = translation_deadzone
        self.rotation_deadzone = rotation_deadzone
        self.last_zone = self.zone.copy()

    def stabilize(self, frame):
        """Return the ROI polygon warped to match ``frame``'s camera position.

        Falls back to the last good polygon whenever the affine cannot be
        trusted (too few reference features, too few survive the flow, or
        ``estimateAffinePartial2D`` fails).
        """
        if self.ref_pts is None or len(self.ref_pts) < self.min_tracked:
            return self.last_zone

        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.ref_gray, curr_gray, self.ref_pts, None)
        if curr_pts is None or status is None:
            return self.last_zone

        idx = np.where(status.ravel() == 1)[0]
        if len(idx) < self.min_tracked:
            return self.last_zone

        m, _ = cv2.estimateAffinePartial2D(self.ref_pts[idx], curr_pts[idx])
        if m is None:
            return self.last_zone

        # Dead-zone guard: near-identity affine on a static camera is noise.
        dx, dy = m[0, 2], m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        if (abs(dx) < self.translation_deadzone
                and abs(dy) < self.translation_deadzone
                and abs(da) < self.rotation_deadzone):
            self.last_zone = self.zone.copy()
            return self.last_zone

        warped = cv2.transform(self.zone[:, None, :], m).reshape(-1, 2)
        self.last_zone = warped.astype(np.float32)
        return self.last_zone


def stablize(video_path: Path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('motorcycles_stabilized.mp4', fourcc, fps, (2 * w, h))
    # out = cv2.VideoWriter('video_stabilized.mp4', fourcc, fps, (w, h))

    if not out.isOpened():
        print("Error: Could not open video writer with codec 'mp4v'")
        exit()
    else:
        print("Video writer opened successfully with codec 'mp4v'")

    _, prev = cap.read()

    if prev is None:
        print("Error: Could not read the first frame.")
        exit()

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames - 1, 3), np.float32)

    for i in range(n_frames - 1):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        assert prev_pts.shape == curr_pts.shape

        idx = np.where(status == 1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        if m is None:
            m = np.eye(2, 3, dtype=np.float32)
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [dx, dy, da]
        prev_gray = curr_gray

        print("Frame: " + str(i) + "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))



    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames - 1):
        success, frame = cap.read()
        if not success:
            break

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (w, h)) 
        frame_stabilized = fixBorder(frame_stabilized)
        frame_out = cv2.hconcat([frame, frame_stabilized])
        # frame_out = frame_stabilized

        out.write(frame_out)
        if frame_out.shape[1] > 1600 or frame_out.shape[0] > 900: 
            frame_out = cv2.resize(frame_out, (frame_out.shape[1]//2, frame_out.shape[0]//2))
        cv2.imshow("Before and After", frame_out)
        cv2.waitKey(10)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")