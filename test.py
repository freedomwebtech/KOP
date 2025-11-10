import cv2
import threading
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # Lightweight model

streams = [
    "rtsp://admin:admin%401234@41.139.156.38:554/Streaming/Channels/10111",
    "rtsp://admin:so123456@73.180.21.51:554/Streaming/Channels/1302",
]

caps = [cv2.VideoCapture(url) for url in streams]

for cap in caps:
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

def process_stream(index, cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.resize(frame, (640, 360))

        # CPU inference
        results = model.predict(frame, imgsz=320, device="cpu", verbose=False)

        annotated = results[0].plot()
        cv2.imshow(f"CAM {index}", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

threads = []
for i, cap in enumerate(caps):
    t = threading.Thread(target=process_stream, args=(i, cap))
    t.daemon = True
    t.start()
    threads.append(t)

for t in threads:
    t.join()

for cap in caps:
    cap.release()
cv2.destroyAllWindows()
