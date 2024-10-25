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
    while True:
        picture = openCamera(cam)
        if picture is not None:
            results = model.predict(picture, classes=[0])
            for result in results:
                picture = result.save("a.png")
            picture = cv2.imread('a.png')

            picture_data = numpyToImg(picture)
            window['-IMAGE-'].update(data=picture_data)
        time.sleep(0.03)  # Add a small delay to reduce CPU usage



def mainWindow():
    cam   = cv2.VideoCapture(0)
    model = YOLO("yolo11n-pose.pt")

    if not cam.isOpened():
        print("Cannot open camera")
        return

    # Initial placeholder image
    placeholder = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
    picture_data = numpyToImg(placeholder)

    layout = [
        [sg.Text('Image Viewer 1.0')],
        [sg.Image(data=picture_data, key='-IMAGE-')]
    ]

    window = sg.Window('Window Title', layout, resizable=True)
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