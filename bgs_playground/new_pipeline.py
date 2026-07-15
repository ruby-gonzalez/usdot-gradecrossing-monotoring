#!/usr/bin/env python3
"""Extract the rail track-corridor ROI (region of interest) polygon.

ROI-only slice of the counting pipeline: this derives the track-corridor
polygon from a trained rail segmentation model (or loads a manually supplied
polygon JSON), then saves the polygon and a preview overlay. It does NOT do any
object detection, tracking, or enter/exit counting.

Usage:
    # Derive ROI from the rail segmentation model by scanning a video:
    python 
/home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/extract_roi.py   

--video /home/gaelmarquez/usdot-gradecrossing-monotoring/bgs_playground/myData/clip_03.mp4    

--rail-model /home/gaelmarquez/usdot-gradecrossing-monotoring/output/rail_seg_all_rails   

--output-dir /home/gaelmarquez/usdot-gradecrossing-monotoring/output/rail_seg_all_rails/roi4 --annotate-video

Note* Without '--annotate-video' it will only output one frame of the entire video
Note* also works on image inputs (so far .jpg work)

    # Add --annotate-video to render the ROI over every frame.
"""

from __future__ import annotations





import os
import sys

# pybgs is a compiled .so built outside the venv; make it importable before
# anything that needs it (blob_analysis imports pybgs at module load).
_BGS_BUILD = "/home/gaelmarquez/bgslibrary/build_py"
if os.path.isdir(_BGS_BUILD) and _BGS_BUILD not in sys.path:
    sys.path.insert(0, _BGS_BUILD)




import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch
import pybgs as bgs


from deep_sort_realtime.deepsort_tracker import DeepSort
from blob_analysis import detect_blobs

from infer_seg_lines import extract_zone_from_heatmap
from train_rail_seg import RailSegModel

#-----------TRACKER LOGIC-----------BEGINS


def blobs_to_detections(blobs, confidence=1.0, det_class="object"):
    """Convert ``Blob`` objects into DeepSORT detection tuples.

    deep-sort-realtime expects each detection as
    ``([left, top, w, h], confidence, class)``. ``Blob.bbox`` is already
    ``(x, y, w, h)`` in that left-top-width-height convention, so this is a
    straight repackaging. Background-subtraction blobs carry no detector score,
    so a constant ``confidence`` is used for all of them.
    """
    detections = []
    for blob in blobs:
        x, y, w, h = blob.bbox
        detections.append(([int(x), int(y), int(w), int(h)], confidence, det_class))
    return detections


def draw_tracks(frame, tracks):
    """Draw confirmed tracks (box + ID) on a frame, in place.

    Only ``confirmed`` tracks are drawn: DeepSORT holds a track as *tentative*
    for the first few frames (``n_init``) before promoting it, which suppresses
    one-frame noise blobs. Returns the same frame for convenience.
    """
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        left, top, right, bottom = (int(v) for v in track.to_ltrb())

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (left, max(0, top - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    return frame


def build_tracker(max_age=30, n_init=3, max_cosine_distance=0.2,
                  embedder="mobilenet", use_gpu=False):
    """Construct a DeepSort tracker configured for blob tracking.

    Args:
        max_age:    frames a track survives without a matching detection before
                    it is deleted. Larger = more tolerant of missed blobs.
        n_init:     consecutive detections required before a track is confirmed.
        max_cosine_distance: appearance-match gate; smaller is stricter.
        embedder:   appearance model name (default MobileNet, weights ship with
                    the package — no runtime download).
        use_gpu:    run the embedder on CUDA. False = CPU (set up here for the
                    torch-cpu install). ``half`` precision only helps on GPU.
    """
    return DeepSort(
        max_age=max_age,
        n_init=n_init,
        max_cosine_distance=max_cosine_distance,
        embedder=embedder,
        bgr=True,            # OpenCV frames are BGR
        embedder_gpu=use_gpu,
        half=use_gpu,        # fp16 only benefits GPU
    )


def _mask_output_path(output_path: Path):
    """Derive the mask video path from the tracked-video path.

    ``foo/clip_tracked.mp4`` -> ``foo/clip_tracked_mask.mp4``, so the two
    outputs sit side by side and are easy to pair up.
    """
    root, ext = os.path.splitext(output_path)
    return f"{root}_mask{ext}"


def alarm_state():
    return None



#-----------TRACKER LOGIC----------- ENDS



#----------ROI LOGIC---------- BEGINS
def load_zone_json(path: Path) -> np.ndarray:
    """Load a manual ROI polygon from Nx2 points or {'vertices'|'polygon'|'points': ...}."""
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        data = data.get("vertices") or data.get("polygon") or data.get("points")
    zone = np.asarray(data, dtype=np.float32)
    if zone.ndim != 2 or zone.shape[1] != 2 or len(zone) < 3:
        raise ValueError(f"{path} must contain an Nx2 polygon or a dict with vertices/polygon/points")
    return zone



def infer_rail_heatmap(model_dir: Path, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Run the rail segmentation model on one frame → (heatmap, scale_x, scale_y).

    scale_x/scale_y map heatmap pixel coords back to the original frame size.
    """
    config = json.loads((model_dir / "config.json").read_text())
    img_w, img_h = config["img_size"]
    num_classes = int(config.get("num_classes", 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = RailSegModel(
        freeze_backbone=0,
        num_classes=num_classes,
        backbone_name=config.get("backbone", "resnet34"),
    )
    try:
        state = torch.load(model_dir / "best_model.pth", weights_only=True, map_location=device)
    except TypeError:
        state = torch.load(model_dir / "best_model.pth", map_location=device)
    model.load_state_dict(state)
    model.to(device).eval()

    orig_h, orig_w = frame.shape[:2]
    resized = cv2.resize(frame, (img_w, img_h), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0].cpu().numpy()
    heatmap = pred.max(axis=0)
    return heatmap, orig_w / img_w, orig_h / img_h


def detect_zone_from_rail_model(
    model_dir: Path,
    video_path: Path,
    threshold: float,
    scan_step: int,
    scan_limit: int,
) -> tuple[np.ndarray, dict]:
    """Scan video frames until the rail model yields a corridor ROI polygon."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    max_frame = min(total_frames, scan_limit) if scan_limit > 0 and total_frames > 0 else scan_limit
    if max_frame <= 0:
        max_frame = total_frames if total_frames > 0 else scan_step

    frame_index = 0
    while frame_index < max_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            break
        heatmap, sx, sy = infer_rail_heatmap(model_dir, frame)
        polygon = extract_zone_from_heatmap(heatmap, threshold=threshold)
        if polygon is not None:
            polygon = polygon.astype(np.float32)
            polygon[:, 0] *= sx
            polygon[:, 1] *= sy
            mask_values = heatmap[heatmap >= threshold]
            cap.release()
            return polygon, {
                "source": "rail_model",
                "scan_frame": frame_index,
                "threshold": threshold,
                "confidence": float(mean(mask_values.tolist())) if mask_values.size else 0.0,
            }
        frame_index += scan_step

    cap.release()
    raise RuntimeError(f"No rail corridor found in first {max_frame} frames of {video_path}")


def draw_zone_overlay(frame: np.ndarray, zone: np.ndarray) -> np.ndarray:
    """Draw the ROI polygon (translucent fill + outline) on a copy of the frame."""
    out = frame.copy()
    overlay = out.copy()
    cv2.fillPoly(overlay, [zone.astype(np.int32)], (0, 180, 255))
    cv2.addWeighted(overlay, 0.22, out, 0.78, 0, out)
    cv2.polylines(out, [zone.astype(np.int32)], True, (0, 220, 255), 2, cv2.LINE_AA)
    return out


def read_first_frame(video_path: Path) -> tuple[np.ndarray, float, int, int]:
    """Grab the first frame plus (fps, width, height) for previews/annotation."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read a frame from {video_path}")
    return frame, fps, width, height
#----------ROI LOGIC---------- ENDS



def annotate_full_video(video_path: Path, zone: np.ndarray, out_path: Path , fps: float,
                        width: 640, height: 480, progress_every: int = 100,min_area=4800, max_area=10000, mask_out_path=None, max_age=30,n_init=3,display=True) -> None:
    """Render the ROI polygon and the Tracking logic over every frame of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    
    algorithm = bgs.SuBSENSE()
    tracker = build_tracker(max_age=max_age,n_init=n_init, use_gpu=torch.cuda.is_available())


    writer=None
    mask_writer=None
    if out_path:
        if mask_out_path is None:#if theres no specified output path for the mask
            mask_out_path = _mask_output_path(out_path)#put the mask in the same folder as the annotated video
        
        out_dir = os.path.dirname(os.path.abspath(out_path))
        os.makedirs(out_dir,exist_ok=True)
       # mask_dir= os.path.dirname(os.path.abspath(mask_out_path))
        #os.makedirs(mask_dir,exist_ok=True)

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open video writer for: {out_path}")

        """mask_writer = cv2.VideoWriter(mask_out_path, fourcc, fps, (width, height))
        if not mask_writer.isOpened():
            raise ValueError(f"Could not open video writer for: {mask_out_path}")"""

    processed = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        #else
        fg_mask = algorithm.apply(frame)
        blobs, cleaned = detect_blobs(fg_mask, min_area=min_area, max_area=max_area)
        detections = blobs_to_detections(blobs)
        #status = alarm_state(detections) <-tbd smh
        tracks = tracker.update_tracks(detections, frame=frame)
        frame = draw_zone_overlay(frame, zone)
        draw_tracks(frame, tracks)
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(f"Annotated {processed} frames")

        if writer is not None:
            writer.write(frame)
        if mask_writer is not None:
            # The mp4v writer expects 3-channel frames; the mask is single
            # channel, so promote it to BGR before writing.
            mask_writer.write(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR))

        if display:
            cv2.imshow("DeepSORT", frame)
            cv2.imshow("Foreground (cleaned)", cleaned)
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting...")
                break

    
    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved tracked video to: {out_path}")
        if mask_writer is not None:
            mask_writer.release()
            print(f"Saved foreground mask to: {mask_out_path}")
    cv2.destroyAllWindows()
    return tracker

#this is for the ROI stuff handles the edge case for if we choose the metadata from JSON (file already exists) 
#OR if we want the seg model to generate a new one
def run(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.zone_json:
        zone = load_zone_json(args.zone_json)
        zone_metadata = {"source": "zone_json", "path": str(args.zone_json)}
    elif args.rail_model:
        zone, zone_metadata = detect_zone_from_rail_model(
            args.rail_model,
            args.video,
            args.rail_threshold,
            args.scan_step,
            args.scan_limit,
        )
    else:
        raise SystemExit("Provide either --rail-model or --zone-json.")

    frame, fps, width, height = read_first_frame(args.video)




#---------------------
    # Save the ROI polygon.
    zone_path = args.output_dir / "track_corridor.json"
    zone_path.write_text(json.dumps({"vertices": zone.tolist(), **zone_metadata}, indent=2) + "\n")

    # Save a preview of the ROI over the first frame.
    preview_path = args.output_dir / "roi_preview.png"
    cv2.imwrite(str(preview_path), draw_zone_overlay(frame, zone))

    summary = {
        "video": str(args.video),
        "rail_model": str(args.rail_model) if args.rail_model else None,
        "frame_size": {"width": width, "height": height},
        "zone": {
            **zone_metadata,
            "vertices": zone.tolist(),
            "area_pct": float(cv2.contourArea(zone.astype(np.float32)) / max(1, width * height) * 100.0),
        },
    }
    summary_path = args.output_dir / "roi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary["zone"], indent=2))
    print(f"Wrote {zone_path}")
    print(f"Wrote {preview_path}")
    print(f"Wrote {summary_path}")

    if args.annotate_video:
        annotate_full_video(
            args.video, zone, args.output_dir / "roi_annotated.mp4", fps, width, height
        )

#fix this for sure
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path, help="Input video path.")
    parser.add_argument("--rail-model", type=Path, help="Rail model dir with best_model.pth and config.json.")
    parser.add_argument("--zone-json", type=Path, help="Manual ROI polygon JSON as Nx2 points or {'vertices': ...}.")
    parser.add_argument("--output-dir",required=True ,type=Path)
    parser.add_argument("--rail-threshold", type=float, default=0.35)
    parser.add_argument("--scan-step", type=int, default=30)
    parser.add_argument("--scan-limit", type=int, default=900)
    parser.add_argument("--annotate-video", action="store_true",
                        help="Also render the ROI over every frame of the video.")
    args = parser.parse_args()
    if args.rail_model and args.zone_json:
        parser.error("Use only one of --rail-model or --zone-json.")
    if not args.rail_model and not args.zone_json:
        parser.error("Provide either --rail-model or --zone-json.")
    return args


"""  if not args.rail_model and not args.zone_json:
        parser.error("Provide either --rail-model or --zone-json.")"""

if __name__ == "__main__":
    run(parse_args())
