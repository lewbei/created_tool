from ultralytics import YOLO

# Load a pretrained YOLOv11 model
model = YOLO('yolo11n.pt')  # Options: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt

# Train the model
model.train(
    data=r"path\dataset.yaml",  # Path to your data.yaml file
    epochs=50,                      # Number of training epochs
    imgsz=640,                      # Image size
    batch=16,                       # Batch size
    device='cuda'                   # Use 'cuda' for GPU training or 'cpu' for CPU training
)
