from ultralytics import YOLO

# Load the trained model (last.pt is the latest trained model)
model = YOLO("runs/detect/train4/weights/last.pt")  

# Resume training
model.train(
    data="C:/Users/lewka/Downloads/exported/dataset.yaml",
    epochs=50,  
    imgsz=640,  
    batch=16,  
    device="cuda"
)
