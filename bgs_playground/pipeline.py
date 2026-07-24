"""DeepSORT tracking on top of the background-subtraction blob detector.

Wires the blob detector in ``blob_analysis.py`` to a DeepSORT tracker
(``deep-sort-realtime``). The pipeline per frame is:

    SuBSENSE foreground mask  ->  detect_blobs()  ->  [bbox, conf, class]
                              ->  DeepSort.update_tracks()  ->  stable IDs

DeepSORT adds two things on top of the raw blobs:
  - a Kalman filter that predicts each object's next position, so a track
    survives a few frames of missed/merged detection (``max_age``), and
  - a MobileNet appearance embedding (the "Deep" in DeepSORT) that helps
    re-attach the right ID after objects cross or briefly occlude.

The blob detector produces no confidence score, so every blob is fed in at a
fixed confidence of 1.0 — motion + appearance do the discriminating.

Note on imports: ``pybgs`` (the SuBSENSE subtractor) is a compiled module that
lives outside the venv. The ``sys.path`` insert below points at the build dir
so ``import pybgs`` works without setting PYTHONPATH by hand. Importing
``blob_analysis`` pulls in pybgs transitively, so this must run first.
"""

import os
import sys

# pybgs is a compiled .so built outside the venv; make it importable before
# anything that needs it (blob_analysis imports pybgs at module load).
_BGS_BUILD = "/home/gaelmarquez/bgslibrary/build_py"
if os.path.isdir(_BGS_BUILD) and _BGS_BUILD not in sys.path:
    sys.path.insert(0, _BGS_BUILD)


import cv2
import pybgs as bgs
from deep_sort_realtime.deepsort_tracker import DeepSort

from blob_analysis import detect_blobs
from extract_roi import * 



def poly_shape(args):
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

    return zone,zone_metadata

def alarm_state(tracks):
    for track in tracks:
            if track.is_confirmed() and track.time_since_update > 15:
                return "ALARM"
    else:
        return "CLEAR"

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


def _mask_output_path(output_path):
    """Derive the mask video path from the tracked-video path.

    ``foo/clip_tracked.mp4`` -> ``foo/clip_tracked_mask.mp4``, so the two
    outputs sit side by side and are easy to pair up.
    """
    root, ext = os.path.splitext(output_path)
    return f"{root}_mask{ext}"


def track_video(video_path, zone: np.ndarray, min_area=1, max_area=None, display=True,
                output_path=None, mask_output_path=None, max_age=30, n_init=3):
    """Run BGS + blob detection + DeepSORT tracking over a video.

    Mirrors ``blob_analysis.analyze_video`` but replaces the per-frame
    ``draw_blobs`` with DeepSORT, so boxes carry persistent IDs across frames.

    Args:
        video_path:  input video.
        min_area:    drop blobs smaller than this (px). Required by
                     ``detect_blobs`` (it compares area < min_area), so it
                     defaults to 1 rather than None.
        max_area:    drop blobs larger than this (px), or None for no cap.
        display:     show annotated + cleaned-mask windows (Esc to quit).
        output_path: if set, write annotated frames here (dirs created).
        mask_output_path: if set, write the cleaned foreground mask here.
                     When None but ``output_path`` is given, defaults to the
                     tracked path with a ``_mask`` suffix.
        max_age:     DeepSORT track lifetime without a match (frames).
        n_init:      detections needed before a track is confirmed.

    Returns:
        The DeepSort tracker (with final internal state), in case the caller
        wants to inspect it after the run.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    algorithm = bgs.SuBSENSE()
    tracker = build_tracker(max_age=max_age, n_init=n_init)


    '''Below is the logic to save the videos into folders for review'''
    writer = None
    mask_writer = None
    if output_path:
        if mask_output_path is None:
            mask_output_path = _mask_output_path(output_path)

        out_dir = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(out_dir, exist_ok=True)
        mask_dir = os.path.dirname(os.path.abspath(mask_output_path))
        os.makedirs(mask_dir, exist_ok=True)

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not writer.isOpened():
            raise ValueError(f"Could not open video writer for: {output_path}")

        mask_writer = cv2.VideoWriter(mask_output_path, fourcc, fps, (width, height))
        if not mask_writer.isOpened():
            raise ValueError(f"Could not open video writer for: {mask_output_path}")
        


    '''directory logic ends here'''

    while True:
        ok, frame = capture.read()
        if not ok:
            break

        fg_mask = algorithm.apply(frame)
        blobs, cleaned = detect_blobs(fg_mask, min_area=min_area, max_area=max_area)
        detections = blobs_to_detections(blobs)
        status = alarm_state(detections)
        # update_tracks needs the frame so the embedder can crop each detection.
        tracks = tracker.update_tracks(detections, frame=frame)

        frame=draw_zone_overlay(frame,zone)

        draw_tracks(frame, tracks)

        # Overlay the alarm status on the frame: red when alarming, green when
        # clear. Drawn after draw_tracks so it sits on top, and before write/
        # imshow so it lands in both the saved video and the live window.
        status_color = (0, 0, 255) if status == "ALARM" else (0, 255, 0)
        cv2.putText(frame, f"Alarm Status: {status.upper()}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        if writer is not None:
            writer.write(frame)
        if mask_writer is not None:
            # The mp4v writer expects 3-channel frames; the mask is single
            # channel, so promote it to BGR before writing.
            mask_writer.write(cv2.cvtColor(cleaned, cv2.COLOR_GRAY2BGR))

        if display:
            cv2.imshow("DeepSORT", frame)
            cv2.imshow("Foreground (cleaned)", cleaned)
            print("Alarm Status: ",status)
            if cv2.waitKey(10) & 0xFF == 27:
                print("Exiting...")
                break


    capture.release()
    if writer is not None:
        writer.release()
        print(f"Saved tracked video to: {output_path}")
    if mask_writer is not None:
        mask_writer.release()
        print(f"Saved foreground mask to: {mask_output_path}")
    cv2.destroyAllWindows()
    return tracker




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video","-v", required=True,type=Path, help="Input video path")
    parser.add_argument("--zone-json","--zj" ,type=Path,help="Input JSON file path")
    parser.add_argument("--rail-model","-r",type=Path, help="Give seg model path")
    parser.add_argument("--output","-o",type=Path,help="Give path to output directory")
    parser.add_argument("--min-area",type=int,help="give minimum pixel area")
    parser.add_argument("--max-area",type=int,help="give maximum pixel area")
    parser.add_argument("--dwell-frames",type=int,help="give dwell frames")
    parser.add_argument("--display",action="store_true")
    parser.add_argument("--rail-threshold", type=float, default=0.35)
    parser.add_argument("--scan-step", type=int, default=30)
    parser.add_argument("--scan-limit", type=int, default=900)

    args=parser.parse_args()
    return args

if __name__ == "__main__":
    args=parse_args()
    print(args)
    print(poly_shape(args))