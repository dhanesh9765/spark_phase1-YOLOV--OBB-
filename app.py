import os
import streamlit as st
import tempfile
import cv2
from ultralytics import YOLO
import numpy as np
from collections import defaultdict
import shapely.geometry

# --- App Configuration ---
st.set_page_config(page_title="Spark Detection", layout="centered")
st.title("⚡ Spark Detection (YOLOv8-OBB)")

# --- Load YOLO model ---
@st.cache_resource
def load_model():
    model_path = "C:/Sparkakt/Spark2/runs/obb/train/weights/best.pt"
    fallback_path = "C:/Sparkakt/Spark2/runs/obb/train/weights/last.pt"
    if os.path.exists(model_path):
        st.success("✅ Loaded: best.pt")
        return YOLO(model_path, task="obb")
    elif os.path.exists(fallback_path):
        st.warning("⚠️ best.pt not found. Loaded last.pt instead.")
        return YOLO(fallback_path, task="obb")
    else:
        st.error("❌ No model file found at expected path.")
        return None

model = load_model()

def oriented_box_to_polygon(box_coords_8):
    points = box_coords_8.reshape((4, 2))
    polygon = shapely.geometry.Polygon(points)
    return polygon

# Define class colors for text (Streamlit markdown) and for box drawing (BGR)
class_colors_text = {
    "Camera Stamp": "red",
    "Date Stamp": "blue",
    "No - Spark": "orange",
    "Sparks": "purple",
}

class_colors_bgr = {
    "Camera Stamp": (0, 0, 255),      # Red
    "Date Stamp": (255, 0, 0),        # Blue
    "No - Spark": (0, 165, 255),      # Orange
    "Sparks": (128, 0, 128),          # Purple
}

# --- File Upload ---
if model:
    st.write("🧠 Model Classes:")
    for cls_id, cls_name in model.names.items():
        color = class_colors_text.get(cls_name, "black")
        st.markdown(f":{color}[{cls_name}]")

    video_file = st.file_uploader("📤 Upload a video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        st.video(video_file)

        if st.button("🚀 Detect"):
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            tfile.close()

            cap = cv2.VideoCapture(tfile.name)
            original_fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            output_path = "spark_output.mp4"
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'),
                                  original_fps, (width, height))

            class_counts = defaultdict(int)
            frame_num = 0
            pbar = st.progress(0)

            frame_skip = 3         # process every 3rd frame to simulate 0.3x speed
            frame_duplicate = 3    # duplicate frame to slow playback visually

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_num % frame_skip != 0:
                    frame_num += 1
                    continue

                results = model(frame, conf=0.3, iou=0.5)  # stricter thresholds

                obb_result = getattr(results[0], "obb", None)

                filtered_indices = []
                if obb_result is not None and obb_result.cls is not None and len(obb_result.cls) > 0:
                    boxes_corners = obb_result.xyxyxyxy.cpu().numpy()  # (N,8)
                    polygons = [oriented_box_to_polygon(box) for box in boxes_corners]
                    confs = obb_result.conf.cpu().numpy()

                    for i, poly_i in enumerate(polygons):
                        keep = True
                        for j, poly_j in enumerate(polygons):
                            if i == j:
                                continue
                            inter_area = poly_i.intersection(poly_j).area
                            union_area = poly_i.union(poly_j).area
                            if union_area == 0:
                                continue
                            iou = inter_area / union_area
                            if iou > 0.5 and confs[i] < confs[j]:
                                keep = False
                                break
                        if keep:
                            filtered_indices.append(i)

                # Count filtered detection classes
                for i in filtered_indices:
                    cls_id = int(obb_result.cls[i])
                    class_name = model.names.get(cls_id, f"class_{cls_id}")
                    class_counts[class_name] += 1

                # Draw filtered bounding boxes and labels
                annotated_frame = frame.copy()
                if obb_result is not None and len(filtered_indices) > 0:
                    for i in filtered_indices:
                        pts = boxes_corners[i].reshape((4, 2)).astype(int)
                        cls_id = int(obb_result.cls[i])
                        class_name = model.names.get(cls_id, f"class_{cls_id}")
                        color = class_colors_bgr.get(class_name, (0, 255, 0))  # default green
                        cv2.polylines(annotated_frame, [pts], isClosed=True, color=color, thickness=2)
                        conf = float(obb_result.conf[i])
                        label = f"{class_name} {conf:.2f}"
                        cv2.putText(annotated_frame, label, tuple(pts[0]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                for _ in range(frame_duplicate):
                    out.write(annotated_frame)

                if frame_num % int(original_fps) == 0:
                    st.image(annotated_frame, caption=f"Frame {frame_num}", channels="BGR", use_column_width=True)

                frame_num += 1
                pbar.progress(min(frame_num / total_frames, 1.0))

            cap.release()
            out.release()
            st.success("✅ Detection completed!")

            st.subheader("📈 Detection Summary")
            for cls_name in model.names.values():
                count = class_counts.get(cls_name, 0)
                st.write(f"🔹 **{cls_name.title()}**: {count}")

            st.subheader("🎬 View and Download Result Video")
            with open(output_path, "rb") as f:
                st.download_button("💾 Download Result Video", f, file_name="spark_detection.mp4")
                st.video(f)
