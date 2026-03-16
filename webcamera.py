import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

clicked_point = None
tracker = None
tracking = False
bbox = None
detected_boxes = []


def mouse_click(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


cap = cv2.VideoCapture(0)

cv2.namedWindow("Object Tracking")
cv2.setMouseCallback("Object Tracking", mouse_click)

print("Click on detected person/vehicle to track")
print("Press R to reset tracking")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # STEP 1: Detection Mode
    # =========================
    if not tracking:

        results = model(frame)[0]
        boxes = results.boxes
        detected_boxes = []

        for box in boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label in ["person", "car", "bus", "truck", "motorbike"]:

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_boxes.append((x1, y1, x2, y2))

                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              (0, 255, 0), 2)
                cv2.putText(frame, label,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

        # If user clicks inside a detected box
        if clicked_point is not None:
            cx, cy = clicked_point

            for (x1, y1, x2, y2) in detected_boxes:
                if x1 < cx < x2 and y1 < cy < y2:

                    bbox = (x1, y1, x2 - x1, y2 - y1)

                    if hasattr(cv2, "legacy"):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    else:
                        tracker = cv2.TrackerCSRT_create()

                    tracker.init(frame, bbox)
                    tracking = True
                    print("Tracking Started")
                    break

            clicked_point = None

    # =========================
    # STEP 2: Tracking Mode
    # =========================
    else:
        success, box = tracker.update(frame)

        if success:
            x, y, w, h = map(int, box)

            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (255, 0, 0), 3)

            cv2.putText(frame, "TRACKING",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2)
        else:
            print("Tracking Lost")
            tracking = False
            tracker = None

    cv2.imshow("Object Tracking", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord('r'):  # Reset
        tracking = False
        tracker = None
        print("Tracking Reset")

cap.release()
cv2.destroyAllWindows()