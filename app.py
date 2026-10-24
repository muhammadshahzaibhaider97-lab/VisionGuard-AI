import os
import tempfile
from datetime import timedelta

import cv2
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="VisionGuard AI", page_icon="🛡️", layout="wide")

MODES = {
    "Workplace & Warehouse Safety": "Flags motion in a user-defined safety zone for human review. This prototype does not verify PPE or prove unsafe behavior.",
    "Restricted-Area Intrusion": "Flags motion detected inside a user-defined restricted zone. Motion is a signal for review, not proof of an intrusion.",
    "Road & Traffic Monitoring": "Summarizes moving regions and scene activity. It does not yet classify vehicles or calculate validated traffic counts/speeds.",
    "General-Purpose Video Inspection": "Detects and summarizes motion across the video for quick review.",
}

st.title("🛡️ VisionGuard AI")
st.caption("Multi-mode video inspection prototype | OpenCV + Streamlit")
st.info(
    "Prototype limitation: this version uses background subtraction and motion contours. "
    "It does not identify people, vehicles, PPE, collisions, or intent. Treat every event as a review prompt."
)

with st.sidebar:
    st.header("Inspection setup")
    mode = st.selectbox("Choose inspection mode", list(MODES.keys()))
    st.write(MODES[mode])
    st.divider()
    st.subheader("Motion sensitivity")
    min_area = st.slider("Minimum moving-region area (pixels)", 100, 10000, 1200, 100)
    threshold = st.slider("Foreground threshold", 10, 100, 25, 5)
    blur_size = st.select_slider("Noise reduction", options=[3, 5, 7, 9], value=5)
    st.caption("Lower area/threshold values detect more motion but can increase false alarms.")
    st.divider()
    st.subheader("Inspection zone")
    st.caption("Zone coordinates are percentages of the frame. The same zone is used for workplace and restricted-area modes.")
    zone_x = st.slider("Left (%)", 0, 95, 20, 1)
    zone_y = st.slider("Top (%)", 0, 95, 20, 1)
    zone_w = st.slider("Width (%)", 5, 100, 45, 1)
    zone_h = st.slider("Height (%)", 5, 100, 55, 1)
    event_interval = st.slider("Minimum seconds between repeated event records", 0.5, 10.0, 2.0, 0.5)

uploaded = st.file_uploader("Upload a video to inspect", type=["mp4", "avi", "mov", "mkv"])

def safe_timestamp(frame_number, fps):
    seconds = frame_number / fps if fps and fps > 0 else 0
    return str(timedelta(seconds=int(seconds))) 

def overlap_ratio(box, zone):
    x, y, w, h = box
    zx, zy, zw, zh = zone
    ix1, iy1 = max(x, zx), max(y, zy)
    ix2, iy2 = min(x + w, zx + zw), min(y + h, zy + zh)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area = max(1, w * h)
    return intersection / area

if uploaded is not None:
    st.write(f"**Selected mode:** {mode}")
    run = st.button("▶ Start inspection", type="primary")
    if run:
        suffix = os.path.splitext(uploaded.name)[1] or ".mp4"
        input_path = None
        output_path = None
        cap = None
        writer = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                f.write(uploaded.getbuffer())
                input_path = f.name

            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("The video could not be opened. Try another MP4/AVI/MOV/MKV file.")
                st.stop()

            fps = cap.get(cv2.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 240:
                fps = 25.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if width <= 0 or height <= 0:
                st.error("Could not read the video's dimensions.")
                st.stop()

            zx = int(width * zone_x / 100)
            zy = int(height * zone_y / 100)
            zw = min(width - zx, int(width * zone_w / 100))
            zh = min(height - zy, int(height * zone_h / 100))
            zone = (zx, zy, zw, zh)

            out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            output_path = out.name
            out.close()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if not writer.isOpened():
                st.error("Could not create the output video on this system.")
                st.stop()

            subtractor = cv2.createBackgroundSubtractorMOG2(
                history=300, varThreshold=threshold, detectShadows=True
            )
            kernel = np.ones((blur_size, blur_size), np.uint8)
            events = []
            frame_no = 0
            last_event_frame = -10**9
            moving_region_samples = []
            progress = st.progress(0)
            status = st.empty()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1

                mask = subtractor.apply(frame)
                _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
                mask = cv2.medianBlur(mask, blur_size)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.dilate(mask, kernel, iterations=2)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                moving = []
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area:
                        continue
                    x, y, w, h = cv2.boundingRect(contour)
                    moving.append((x, y, w, h, area))

                moving_region_samples.append(len(moving))
                # Draw inspection zone; color is for visual guidance, not a severity score.
                cv2.rectangle(frame, (zx, zy), (zx + zw, zy + zh), (255, 180, 0), 2)
                cv2.putText(frame, "INSPECTION ZONE", (zx, max(22, zy - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

                in_zone = []
                for x, y, w, h, area in moving:
                    ratio = overlap_ratio((x, y, w, h), zone)
                    if ratio > 0.15:
                        in_zone.append((x, y, w, h, area, ratio))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 220, 0), 2)

                should_record = False
                event_type = ""
                if mode == "General-Purpose Video Inspection":
                    should_record = len(moving) > 0
                    event_type = "Motion detected"
                elif mode == "Road & Traffic Monitoring":
                    # This is scene-motion activity, not a validated vehicle count.
                    should_record = len(moving) > 0
                    event_type = "Road-scene motion activity"
                elif mode == "Workplace & Warehouse Safety":
                    should_record = len(in_zone) > 0
                    event_type = "Motion in safety review zone"
                elif mode == "Restricted-Area Intrusion":
                    should_record = len(in_zone) > 0
                    event_type = "Motion in restricted review zone"

                if should_record and (frame_no - last_event_frame) / fps >= event_interval:
                    events.append({
                        "timestamp": safe_timestamp(frame_no, fps),
                        "frame": frame_no,
                        "mode": mode,
                        "event": event_type,
                        "moving_regions": len(moving),
                        "regions_in_zone": len(in_zone),
                        "review_note": "Review required; motion detection alone does not confirm a safety violation or intrusion."
                    })
                    last_event_frame = frame_no

                label = f"Mode: {mode} | Moving regions: {len(moving)}"
                cv2.rectangle(frame, (8, 8), (min(width - 8, 760), 42), (20, 20, 20), -1)
                cv2.putText(frame, label[:95], (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
                writer.write(frame)

                if total_frames > 0 and frame_no % 5 == 0:
                    progress.progress(min(frame_no / total_frames, 1.0))
                    status.write(f"Processing frame {frame_no:,} of approximately {total_frames:,}…")

            progress.progress(1.0)
            status.success(f"Inspection finished. Processed {frame_no:,} frames.")
            cap.release()
            cap = None
            writer.release()
            writer = None

            if not events:
                events_df = pd.DataFrame(columns=[
                    "timestamp", "frame", "mode", "event", "moving_regions",
                    "regions_in_zone", "review_note"
                ])
            else:
                events_df = pd.DataFrame(events)

            col1, col2, col3 = st.columns(3)
            col1.metric("Frames processed", f"{frame_no:,}")
            col2.metric("Event records", f"{len(events_df):,}")
            avg_regions = (sum(moving_region_samples) / len(moving_region_samples)) if moving_region_samples else 0
            col3.metric("Average moving regions/frame", f"{avg_regions:.2f}")

            st.subheader("Inspection results")
            if len(events_df) == 0:
                st.info("No events met the current settings. This does not prove that the video is safe; try adjusting sensitivity or the zone.")
            else:
                st.dataframe(events_df, use_container_width=True, hide_index=True)

            st.subheader("Processed video")
            with open(output_path, "rb") as vf:
                video_bytes = vf.read()
            st.video(video_bytes)
            st.download_button(
                "⬇ Download annotated video",
                data=video_bytes,
                file_name="visionguard_inspection.mp4",
                mime="video/mp4"
            )
            csv_bytes = events_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download event report (CSV)",
                data=csv_bytes,
                file_name="visionguard_events.csv",
                mime="text/csv"
            )
            st.caption("For responsible use, have a person review flagged clips before making safety, disciplinary, or security decisions.")
        except Exception as exc:
            st.error(f"Inspection failed: {exc}")
            st.info("If this persists, share the complete error text so we can troubleshoot it.")
        finally:
            if cap is not None:
                cap.release()
            if writer is not None:
                writer.release()
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass
else:
    st.markdown("### How to use")
    st.markdown(
        "1. Choose an inspection mode in the sidebar.  \n"
        "2. Set motion sensitivity and, where relevant, the inspection zone.  \n"
        "3. Upload a short video and click **Start inspection**.  \n"
        "4. Review the annotated video and download the event CSV."
    )
    st.warning("This is an early prototype. Use non-sensitive test footage and do not rely on it as a sole safety or security system.")
