from ultralytics import YOLO
import cv2
import pyttsx3

engine = pyttsx3.init()
engine.setProperty("rate", 150)

alert_given = False

model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25, verbose=False)

    phone_detected = False

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            name = model.names[cls]

            if name == "cell phone":
                phone_detected = True

                if not alert_given:
                    engine.say("Warning. Mobile phone detected.")
                    engine.runAndWait()
                    alert_given = True

    # Step 4: Reset alert when phone disappears
    if not phone_detected:
        alert_given = False

    annotated_frame = results[0].plot()

    cv2.imshow("VisionGuard AI", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()