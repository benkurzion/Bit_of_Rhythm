from ultralytics import YOLO
import cv2

def train_model():
    model = YOLO("yolov8n.pt")
    print("Loaded model")
    model.train(data="BoR.v1-batch_1_trial1.yolov8/data.yaml", epochs=30, imgsz=640, batch=16)
    metrics = model.val()

    print(f"mAP@0.5: {metrics.box.map50}")
    print(f"mAP@0.5:0.95: {metrics.box.map}")

def inference():
    model = YOLO("deep-learning\\shared_weights\\best.pt")  

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, stream=True) 

        for r in results:
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = map(int, box)
                label = f"{model.names[int(cls)]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Live", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    inference()