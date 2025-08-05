# %%
pip install roboflow

# %%
# Step 2: Import libraries
from roboflow import Roboflow
import os


# %%
# Step 3: Initialize Roboflow with your API key
print("Initializing Roboflow...")
rf = Roboflow(api_key="YPOp9UwHhsEpGxHGdnJp")

# %%
# Step 4: Access your project
print("Accessing project...")
project = rf.workspace("wheelchair-un2j4").project("spark-1")


# %%
# Step 5: Get the specific version
print("Getting project version...")
version = project.version(1)

# %%
# Step 6: Download the dataset
print("Downloading dataset...")
dataset = version.download("yolov8-obb")

print("Dataset downloaded successfully!")

# %%
# Step 7: Check the downloaded dataset structure
print("\nDataset structure:")
dataset_path = dataset.location
print(f"Dataset location: {dataset_path}")

# %%
# List the contents of the dataset directory
if os.path.exists(dataset_path):
    print("\nDataset contents:")
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            print(f"📁 {item}/")
            # Show contents of subdirectories
            try:
                sub_items = os.listdir(item_path)
                for sub_item in sub_items[:5]:  # Show first 5 items
                    print(f"  - {sub_item}")
                if len(sub_items) > 5:
                    print(f"  ... and {len(sub_items) - 5} more files")
            except:
                pass
        else:
            print(f"📄 {item}")


# %%
# Step 8: Display dataset information
print(f"\nDataset name: {dataset.name}")
print(f"Dataset version: {dataset.version}")
print(f"Dataset location: {dataset.location}")

# %%
# Step 9: Show detailed dataset statistics
print("\n" + "="*50)
print("DATASET STATISTICS")
print("="*50)

try:
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")
    test_path = os.path.join(dataset_path, "test")

    total_images = 0
    total_labels = 0

    for split_name, split_path in [("Training", train_path), ("Validation", valid_path), ("Test", test_path)]:
        if os.path.exists(split_path):
            images_path = os.path.join(split_path, "images")
            labels_path = os.path.join(split_path, "labels")

            if os.path.exists(images_path):
                images = [f for f in os.listdir(images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                image_count = len(images)
                total_images += image_count

                if os.path.exists(labels_path):
                    labels = [f for f in os.listdir(labels_path) if f.lower().endswith('.txt')]
                    label_count = len(labels)
                    total_labels += label_count

                    print(f"{split_name:12}: {image_count:4} images, {label_count:4} labels")
                else:
                    print(f"{split_name:12}: {image_count:4} images, 0 labels")

    print("-" * 30)
    print(f"{'Total':12}: {total_images:4} images, {total_labels:4} labels")

    # Read data.yaml for class information
    data_yaml_path = os.path.join(dataset_path, "data.yaml")
    if os.path.exists(data_yaml_path):
        print(f"\n📋 Dataset Configuration (data.yaml):")
        with open(data_yaml_path, 'r') as f:
            yaml_content = f.read()
            print(yaml_content)

    # Read README files
    readme_files = ['README.roboflow.txt', 'README.dataset.txt']
    for readme_file in readme_files:
        readme_path = os.path.join(dataset_path, readme_file)
        if os.path.exists(readme_path):
            print(f"\n📖 {readme_file}:")
            with open(readme_path, 'r') as f:
                content = f.read()
                # Show first 500 characters
                if len(content) > 500:
                    print(content[:500] + "...")
                else:
                    print(content)

except Exception as e:
    print(f"Could not get detailed statistics: {e}")

print("\n✅ Dataset ready for training!")


# %%
# Step 10: Next steps - Training setup
print("\n" + "="*50)
print("NEXT STEPS FOR YOLOv8-OBB TRAINING")
print("="*50)
print("1. Install YOLOv8: !pip install ultralytics")
print("2. Import YOLO: from ultralytics import YOLO")
print("3. Load model: model = YOLO('yolov8n-obb.pt')")
print("4. Train model: model.train(data='/content/Spark-1-1/data.yaml', epochs=100)")
print("5. Validate: model.val()")
print("6. Predict: model.predict('path/to/image.jpg')")

print("\n💡 Quick training command:")
print("```python")
print("# Install and train YOLOv8-OBB")
print("!pip install ultralytics")
print("from ultralytics import YOLO")
print("model = YOLO('yolov8n-obb.pt')")
print(f"results = model.train(data='{dataset_path}/data.yaml', epochs=50, imgsz=640)")
print("```")

# %%
# Step 1: Install required packages
pip install ultralytics opencv-python-headless

# %%
# Step 2: Import libraries
import cv2
import numpy as np
pip install ultralytics
from ultralytics import YOLO
import os
from IPython.display import HTML, Video
import matplotlib.pyplot as plt
from PIL import Image
import io

# %%
import os
from ultralytics import YOLO
from roboflow import Roboflow
import cv2
import matplotlib.pyplot as plt
from IPython.display import Video, HTML
import numpy as np


# %%
# Step 3: Download your dataset from Roboflow
print("\n" + "="*50)
print("DOWNLOADING DATASET")
print("="*50)

print("🔄 Initializing Roboflow...")
rf = Roboflow(api_key="YPOp9UwHhsEpGxHGdnJp")

print("🔄 Accessing your project...")
project = rf.workspace("wheelchair-un2j4").project("spark-1")
version = project.version(1)

print("📥 Downloading dataset...")
dataset = version.download("yolov8-obb")

print("✅ Dataset downloaded successfully!")
print(f"📁 Dataset location: {dataset.location}")

# %%
# Step 4: Verify dataset structure
print("\n📊 Dataset Structure:")
dataset_path = dataset.location

# Count images in each split
splits = ['train', 'valid', 'test']
total_images = 0

for split in splits:
    split_path = os.path.join(dataset_path, split, 'images')
    if os.path.exists(split_path):
        image_count = len([f for f in os.listdir(split_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        print(f"{split:8}: {image_count:4} images")
        total_images += image_count

print(f"{'Total':8}: {total_images:4} images")

# Show data.yaml content
data_yaml_path = os.path.join(dataset_path, 'data.yaml')
print(f"\n📋 Dataset Configuration:")
with open(data_yaml_path, 'r') as f:
    print(f.read())

# %%
# Step 5: Train YOLOv8-OBB model
print("\n" + "="*50)
print("TRAINING YOLOv8-OBB MODEL")
print("="*50)

print("🔄 Loading pre-trained YOLOv8-OBB model...")
model = YOLO('yolov8n-obb.pt')  # Start with nano model for faster training

print("🏋️ Starting training...")
print("This may take 15-30 minutes depending on your dataset size...")

# Train the model
results = model.train(
    data=data_yaml_path,
    epochs=50,          # Number of training epochs
    imgsz=640,          # Image size
    patience=10,        # Early stopping patience1
    save=True,          # Save checkpoints
    device='cpu',       # Use CPU (change to 0 for GPU if available)
    verbose=True,       # Verbose output
    plots=True,         # Generate training plots
    val=True,           # Validate during training
)

print("✅ Training completed!")

# %%
import os

# Update the runs_dir to the correct absolute path
runs_dir = "/opt/homebrew/runs/obb"
train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
latest_train = max(train_dirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
weights_dir = os.path.join(runs_dir, latest_train, 'weights')

best_model_path = os.path.join(weights_dir, 'best.pt')
last_model_path = os.path.join(weights_dir, 'last.pt')

print(f"📁 Training results saved in: {os.path.join(runs_dir, latest_train)}")
print(f"🏆 Best model: {best_model_path}")
print(f"🔄 Last model: {last_model_path}")



# %%
# Step 7: Load the trained model for testing
print("\n🔄 Loading your trained model...")

# Check if best_model_path was set in the previous step
if 'best_model_path' in globals() and best_model_path is not None and os.path.exists(best_model_path):
    trained_model = YOLO(best_model_path)
    print("✅ Trained model loaded successfully!")

    # Display model info
    print(f"\n📊 Your Trained Model Information:")
    print(f"Model classes: {trained_model.names}")
    print(f"Number of classes: {len(trained_model.names)}")
else:
    print("❌ Cannot load trained model.")
    print("Please ensure the training step (Step 5) completed successfully and generated a 'best.pt' file.")
    # Optionally, you could raise an error or exit here if loading the model is essential.
    # raise FileNotFoundError("Best model file not found. Training may have failed.")

# %%
# Step 8: Validate the model
print("\n📈 Validating model performance...")
validation_results = trained_model.val()
print("✅ Validation completed!")

# %%
# Step 9: Show training plots
print("\n📊 Training Results:")
results_dir = os.path.join(runs_dir, latest_train)

# List available plots
plot_files = [f for f in os.listdir(results_dir) if f.endswith('.png')]
print("Generated plots:")
for plot in plot_files:
    print(f"  📈 {plot}")

# Display some key plots
key_plots = ['results.png', 'confusion_matrix.png', 'val_batch0_labels.png']
for plot_name in key_plots:
    plot_path = os.path.join(results_dir, plot_name)
    if os.path.exists(plot_path):
        print(f"\n📊 {plot_name}:")
        img = plt.imread(plot_path)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.title(plot_name)
        plt.show()


# %%
import os
import cv2
from ultralytics import YOLO
from IPython.display import Video  # Optional, remove if not running in Jupyter

# Path to your trained model
trained_model = YOLO("/opt/homebrew/runs/obb/train/weights/best.pt")


# Step 10: Test on video
print("\n" + "=" * 50)
print("VIDEO TESTING")
print("=" * 50)

# Instead of uploading, specify local path
video_path = '/Users/akshit/Downloads/Untitled video 1 - Made with Clipchamp.mp4'

if os.path.isfile(video_path):
    print(f"✅ Video found: {video_path}")

    # Get video info
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    cap.release()

    print(f"\n📹 Video Information:")
    print(f"Resolution: {width}x{height}")
    print(f"FPS: {fps}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"Total frames: {total_frames}")

    # Process video with trained model
    print("\n🔄 Processing video with your trained model...")
    video_results = trained_model.predict(
        source=video_path,
        save=True,
        conf=0.25,
        iou=0.45,
        show_labels=True,
        show_conf=True,
        line_width=2,
        verbose=False,
        stream=True
    )

    print("✅ Video processing completed!")

    # Find output video
    predict_dirs = []
    for root, dirs, files in os.walk("runs/obb"):
        for dir_name in dirs:
            if dir_name.startswith("predict"):
                predict_dirs.append(os.path.join(root, dir_name))

    if predict_dirs:
        latest_predict_dir = max(predict_dirs, key=os.path.getctime)

        output_video_path = None
        for file in os.listdir(latest_predict_dir):
            if file.endswith(('.mp4', '.avi', '.mov')):
                output_video_path = os.path.join(latest_predict_dir, file)
                print(f"\n📁 Output video saved at: {output_video_path}")
                break

        if output_video_path is None:
            print("❌ Could not find the output video in the predict directory.")
    else:
        print("❌ No prediction directory found.")

    # Detection Statistics
    print("\n📈 Detection Statistics:")
    video_results_stream = trained_model.predict(source=video_path, conf=0.25, save=False, verbose=False, stream=True)

    total_detections = 0
    class_counts = {}

    for result in video_results_stream:
        if result.obb is not None:
            boxes = result.obb
            for box in boxes:
                cls_id = int(box.cls)
                if cls_id < len(trained_model.names):
                    class_name = trained_model.names[cls_id]
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                    total_detections += 1
                else:
                    print(f"Warning: Detected class ID {cls_id} is out of range.")

    print(f"Total detections across all frames: {total_detections}")
    if class_counts:
        print("Detections by class:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")
    else:
        print("No detections found.")
else:
    print("❌ Invalid file path. Please check the path and try again.")


# %%
# Corrected path to match YOLOv8-OBB output
import os # Add the import statement here

# Corrected path to match YOLOv8-OBB output
runs_dir = "runs/obb"

# Check if runs_dir exists and contains train directories
if os.path.exists(runs_dir):
    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
    if train_dirs:
        latest_train = max(train_dirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
        weights_dir = os.path.join(runs_dir, latest_train, 'weights')

        # Define best_model_path and last_model_path if weights_dir exists
        best_model_path = None # Initialize to None
        last_model_path = None # Initialize to None

        if os.path.exists(weights_dir):
            best_model_path = os.path.join(weights_dir, 'best.pt')
            last_model_path = os.path.join(weights_dir, 'last.pt')
            print(f"📁 Training results saved in: runs/obb/{latest_train}")
            print(f"🏆 Best model: {best_model_path}")
            print(f"🔄 Last model: {last_model_path}")
        else:
            print(f"❌ Weights directory not found at: {weights_dir}")
            print("Please ensure the training step generated weights.")
    else:
        print(f"❌ No 'train' directories found in {runs_dir}")
        print("Training may not have completed successfully.")
else:
    print(f"❌ Runs directory not found at: {runs_dir}")
    print("Training may not have started or completed successfully.")

# Now, in the final summary cell (Step 11), add checks before printing the paths.
# Step 11: Summary
print("\n" + "="*50)
print("TRAINING AND TESTING COMPLETE!")
print("="*50)

print("✅ Successfully completed:")
print("  1. Downloaded dataset from Roboflow")
print("  2. Trained YOLOv8-OBB model")
print("  3. Validated model performance")
print("  4. Tested model on video")

print(f"\n📁 Your trained model files:")
# Check if best_model_path was successfully assigned before printing
if 'best_model_path' in globals() and best_model_path is not None:
    print(f"  🏆 Best model: {best_model_path}")
else:
    print("  🏆 Best model: Path not found. Training may have failed.")

# Check if last_model_path was successfully assigned before printing
if 'last_model_path' in globals() and last_model_path is not None:
    print(f"  🔄 Last model: {last_model_path}")
else:
    print("  🔄 Last model: Path not found. Training may have failed.")

# Corrected path to match YOLOv8-OBB output
runs_dir = "runs/obb"

# Check if runs_dir exists and contains train directories
if os.path.exists(runs_dir):
    train_dirs = [d for d in os.listdir(runs_dir) if d.startswith('train')]
    if train_dirs:
        latest_train = max(train_dirs, key=lambda x: os.path.getctime(os.path.join(runs_dir, x)))
        weights_dir = os.path.join(runs_dir, latest_train, 'weights')

        # Define best_model_path and last_model_path if weights_dir exists
        best_model_path = None # Initialize to None
        last_model_path = None # Initialize to None

        if os.path.exists(weights_dir):
            best_model_path = os.path.join(weights_dir, 'best.pt')
            last_model_path = os.path.join(weights_dir, 'last.pt')
            print(f"📁 Training results saved in: runs/obb/{latest_train}")
            print(f"🏆 Best model: {best_model_path}")
            print(f"🔄 Last model: {last_model_path}")
        else:
            print(f"❌ Weights directory not found at: {weights_dir}")
            print("Please ensure the training step generated weights.")
    else:
        print(f"❌ No 'train' directories found in {runs_dir}")
        print("Training may not have completed successfully.")
else:
    print(f"❌ Runs directory not found at: {runs_dir}")
    print("Training may not have started or completed successfully.")

# Now, in the final summary cell (Step 11), add checks before printing the paths.
# Step 11: Summary
print("\n" + "="*50)
print("TRAINING AND TESTING COMPLETE!")
print("="*50)

print("✅ Successfully completed:")
print("  1. Downloaded dataset from Roboflow")
print("  2. Trained YOLOv8-OBB model")
print("  3. Validated model performance")
print("  4. Tested model on video")

print(f"\n📁 Your trained model files:")
# Check if best_model_path was successfully assigned before printing
if 'best_model_path' in globals() and best_model_path is not None:
    print(f"  🏆 Best model: {best_model_path}")
else:
    print("  🏆 Best model: Path not found. Training may have failed.")

# Check if last_model_path was successfully assigned before printing
if 'last_model_path' in globals() and last_model_path is not None:
    print(f"  🔄 Last model: {last_model_path}")
else:
    print("  🔄 Last model: Path not found. Training may have failed.")


print(f"\n💡 To use your model later:")


# %% [markdown]
# ###To Test Video Run Following Command

# %%
%pip install ultralytics opencv-python-headless

# %%
# Step 2: Import libraries
import cv2
import numpy as np
from ultralytics import YOLO
from IPython.display import display, Image
import io


# %%
# Step 3: Load your trained model
# Assuming you have already trained your model and have the best.pt file
# Make sure the best_model_path variable is set correctly from your training step
# Example (replace with your actual path):
# best_model_path = 'runs/obb/train/weights/best.pt'


if 'best_model_path' in globals() and best_model_path is not None and os.path.exists(best_model_path):
    trained_model = YOLO(best_model_path)
    print("✅ Trained model loaded successfully!")
else:
    print("❌ Cannot load trained model. Please ensure the best_model_path is correct.")
    # Exit or handle the error appropriately if the model is not found

# %%
import os

print("\n" + "="*50)
print("UPLOAD VIDEO FOR 'LIVE' DETECTION")
print("="*50)

# Option 1: Prompt user to input the path
video_path = '/Users/akshit/Downloads/Untitled video 1 - Made with Clipchamp.mp4'

# Option 2: Or hardcode the path (uncomment below if preferred)
# video_path = "/Users/akshit/Downloads/your_video.mp4"

if os.path.isfile(video_path):
    print(f"✅ Video found: {video_path}")
else:
    print("❌ Invalid file path. Please check and try again.")


# %%
# Step 5: Process video frame by frame and display results
print("\n" + "="*50)
print("PROCESSING VIDEO FRAME BY FRAME")
print("="*50)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
else:
    try:
        # Create a placeholder for the image display that can be updated
        img_placeholder = display(Image(data=b''), display_id=True)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the frame
            results = trained_model(frame, verbose=False, conf=0.25) # Adjust confidence as needed

            # Process results and draw detections
            annotated_frame = results[0].plot() # This draws bounding boxes on the frame

            # Convert the frame to a format suitable for displaying in Colab
            _, png = cv2.imencode('.png', annotated_frame)
            img_placeholder.update(Image(data=png.tobytes()))

            frame_count += 1
            # Optional: Limit the number of frames processed for a shorter demo
            # if frame_count > 100: # Process first 100 frames
            #     break

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        cap.release()
# Step 5: Process video frame by frame and display results
print("\n" + "="*50)
print("PROCESSING VIDEO FRAME BY FRAME")
print("="*50)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print(f"Error opening video file: {video_path}")
else:
    try:
        # Create a placeholder for the image display that can be updated
        img_placeholder = display(Image(data=b''), display_id=True)

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Run inference on the frame
            results = trained_model(frame, verbose=False, conf=0.25) # Adjust confidence as needed

            # Process results and draw detections
            annotated_frame = results[0].plot() # This draws bounding boxes on the frame

            # Convert the frame to a format suitable for displaying in Colab
            _, png = cv2.imencode('.png', annotated_frame)
            img_placeholder.update(Image(data=png.tobytes()))

            frame_count += 1
            # Optional: Limit the number of frames processed for a shorter demo
            # if frame_count > 100: # Process first 100 frames
            #     break

    except Exception as e:
        print(f"An error occurred during processing: {e}")
    finally:
        cap.release()
        print("\n✅ Video processing finished.")

# %%



