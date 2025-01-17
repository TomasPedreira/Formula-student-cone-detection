import ultralytics
import cv2
import time


model = ultralytics.YOLO('./models/best.pt')


def get_frame_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        result = model.predict(frame, conf=0.5)
        cv2.imshow('frame', result[0].plot())
        #cv2.imshow('frame', frame)


        # wait for 'q' key to stop the showing of the frame
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
            
        time.sleep(0.033333)
    cap.release()


def main():
    video_path = './videos/test_video_1.mp4'
    get_frame_from_video(video_path)


if __name__ == '__main__':
    main()