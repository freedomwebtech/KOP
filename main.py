import cv2
from ultralytics import solutions
import time

# RTSP stream URL - replace with your actual RTSP URL
rtsp_url = "rtsp://username:password@ip_address:port/stream"
# Example formats:
# rtsp_url = "rtsp://admin:password@192.168.1.100:554/stream1"
# rtsp_url = "rtsp://192.168.1.100:554/Streaming/Channels/101"

# Open RTSP stream
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Reduce buffer size for lower latency

assert cap.isOpened(), "Error reading RTSP stream"

# Define counting region - Rectangle for IN/OUT counting
# Adjust coordinates based on your camera view (x1, y1), (x2, y2), (x2, y3), (x1, y3)
region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]  # rectangle region

# Get stream properties
w, h = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize object counter with IN/OUT counting
counter = solutions.ObjectCounter(
    show=True,  # display the output
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model="yolo11n-obb.pt" for OBB model
    line_width=2,  # adjust line thickness
    # classes=[0, 2],  # count specific classes (person=0, car=2 in COCO)
    # tracker="botsort.yaml",  # alternative tracker
)

# Frame skip for performance (process every Nth frame)
frame_skip = 1  # Set to 2 or 3 if stream is too slow
frame_count = 0

# Reconnection settings
max_reconnect_attempts = 5
reconnect_delay = 5  # seconds

print("Starting RTSP stream processing...")
print("Press 'q' to quit")

try:
    while True:
        success, im0 = cap.read()
        
        if not success:
            print("Failed to read frame. Attempting to reconnect...")
            cap.release()
            
            for attempt in range(max_reconnect_attempts):
                print(f"Reconnection attempt {attempt + 1}/{max_reconnect_attempts}")
                time.sleep(reconnect_delay)
                cap = cv2.VideoCapture(rtsp_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                
                if cap.isOpened():
                    print("Reconnected successfully!")
                    break
            else:
                print("Failed to reconnect. Exiting...")
                break
            continue
        
        frame_count += 1
        
        # Skip frames for better performance if needed
        if frame_count % frame_skip != 0:
            continue
        
        # Process frame with object counter
        results = counter(im0)
        
        # Access IN/OUT counts (optional - for custom logging)
        # in_count = counter.in_count
        # out_count = counter.out_count
        # print(f"IN: {in_count}, OUT: {out_count}")
        
        # Check for quit key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Quit signal received")
            break

except KeyboardInterrupt:
    print("\nInterrupted by user")
except Exception as e:
    print(f"Error occurred: {e}")
finally:
    # Cleanup
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    print("Done!")
