import cv2
from detection import VehicleDetector
from tracker import VehicleCounter
from utils import draw_boxes

def main():
    cap = cv2.VideoCapture("data/sample_video.mp4")

    detector = VehicleDetector()
    counter = VehicleCounter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        total_count = counter.update(detections)

        frame = draw_boxes(frame, detections)

        cv2.putText(frame, f"Vehicle Count: {total_count}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

        cv2.imshow("Traffic Monitoring", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()