import FreeSimpleGUI as sg
import cv2
import numpy as np
import io
import threading
import time
from ultralytics import YOLO


def openCamera(cam):
    ret, frame = cam.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        return None
    return frame


def numpyToImg(img):
    if img is None:
        return None
    # Encode the image as PNG
    success, encoded_image = cv2.imencode('.png', img)
    if success:
        return encoded_image.tobytes()  # Return as bytes
    return None



def updateImage(window, cam, model):

    kpColors = [
        (255, 0, 0),    # Blue
        (0, 255, 0),    # Green
        (0, 0, 255),    # Red
        (255, 255, 0),  # Cyan
        (0, 255, 255),  # Magenta
        (255, 0, 255),  # Yellow
        (128, 0, 0),    # Dark Red
        (0, 128, 0),    # Dark Green
        (0, 0, 128),    # Dark Blue
        (128, 128, 0),  # Olive
        (0, 128, 128),  # Teal
        (128, 0, 128),  # Purple
        (192, 192, 192), # Silver
        (255, 165, 0),  # Orange
        (255, 20, 147), # Deep Pink
        (255, 105, 180),# Hot Pink
        (128, 128, 128) # Gray
    ]


    while True:
        picture = openCamera(cam)
        if picture is not None:
            results = model.predict(picture, classes=[0], verbose=False)
            for result in results:
                for kp in result.keypoints:
                    for detection in kp.xy:
                        previousPoint = -1
                        for idx, (x, y) in enumerate(detection):
                            if x != 0 and y != 0:
                                cv2.circle(picture, center=(int(x), int(y)), radius=5, color=kpColors[idx], thickness=-1)
                                if previousPoint != -1:
                                    cv2.line(picture, (int(x), int(y)), previousPoint, color=kpColors[idx], thickness=2)
                                previousPoint = (int(x), int(y))

                for obj in result.boxes:
                    for x0, y0, x1, y1 in obj.xyxy:
                        cv2.rectangle(picture, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 0), thickness=2)

            picture_data = numpyToImg(picture)
            window['-IMAGE-'].update(data=picture_data)

            window.refresh()
        time.sleep(0.1)  # Add a small delay to reduce CPU usage



def mainWindow():
    cam   = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    model = YOLO("yolo11n-pose.pt")

    if not cam.isOpened():
        print("Cannot open camera")
        return

    # Initial placeholder image
    placeholder = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    picture_data = numpyToImg(placeholder)

    layout = [
        [sg.Text('Image Viewer 1.0')],
        [sg.Image(data=picture_data, key='-IMAGE-')]
    ]

    window = sg.Window('Window Title', layout, resizable=False, finalize=True)
    threading.Thread(target=updateImage, args=(window, cam, model), daemon=True).start()

    # Event Loop
    while True:
        event, values = window.read()
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

    window.close()
    cam.release()  # Release the camera when done


if __name__ == "__main__":
    mainWindow()
