import threading
import cv2
from ultralytics import YOLO
import numpy as np
'''from ultralytics import YOLO

# Load YOLOv5 and custom YOLOv8 models
model_v5 = YOLO('yolov5s.pt')  # pre-trained model
model_v8 = YOLO('path/to/your/custom-yolov8-model.pt')  # custom trained model

# Process an image with both models
results_v5 = model_v5('image.jpg')
results_v8 = model_v8('image.jpg')

# Show results
results_v5.show()
results_v8.show()'''

def run_tracker_in_thread(filename, model, file_index):
    """
    Runs a video file or webcam stream concurrently with the YOLOv8 model using threading.

    This function captures video frames from a given file or camera source and utilizes the YOLOv8 model for object
    tracking. The function runs in its own thread for concurrent processing.

    Args:
        filename (str): The path to the video file or the identifier for the webcam/external camera source.
        model (obj): The YOLOv8 model object.
        file_index (int): An index to uniquely identify the file being processed, used for display purposes.

    Note:
        Press 'q' to quit the video display window.
    """
    video = cv2.VideoCapture(filename)  # Read the video file
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(f"output_{file_index}.avi", cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_width * 2, frame_height))
    while True:
        ret, frame = video.read()  # Read the video frames

        # Exit the loop if no more frames in either video
        if not ret:
            break

        # Track objects in frames if available
        
        results_track = model.track(frame, conf=0.3, save=True, persist=True, tracker="bytetrack.yaml")
        
        res_plotted = results_track[0].plot()
        results = model(frame, conf=0.3, save=True)
        res_plotted2 = results[0].plot()
        
        # Display both frames side by side
        concat = np.hstack((res_plotted2, res_plotted))
        cv2.imshow(f"Tracking_Stream_{file_index}", concat)
        out.write(concat)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    # Release video sources
    video.release()
    out.release()

#C:\Users\user\Downloads\workspace\yolov9\runs\train\exp3\weights\last.pt
# Load the models
model1 = YOLO('../workspace/yolov9/runs/train/exp3/weights/last.pt')

# Define the video files for the trackers
video_file1 = "C:/Users/user/Downloads/workspace/video/CA_00001.mp4" 
video_file2 = 0  # Path to video file, 0 for webcam, 1 for external camera
run_tracker_in_thread(video_file1, model1, 1)
# Create the tracker threads
tracker_thread1 = threading.Thread(target=run_tracker_in_thread, args=(video_file1, model1, 1), daemon=True)
tracker_thread2 = threading.Thread(target=run_tracker_in_thread, args=(video_file2, model1, 2), daemon=True)
# Clean up and close windows
cv2.destroyAllWindows()
