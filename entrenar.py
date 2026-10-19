from ultralytics import YOLO

if __name__ == "__main__":
    modelo = YOLO("yolov8n-cls.pt")
    modelo.train(
        data="dataset_yolo",
        epochs=20,
        imgsz=224,
        batch=16,
        workers=2,
        name="supermercado",
    )