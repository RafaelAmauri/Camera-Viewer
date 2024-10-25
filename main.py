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



def updateImage(window, cam, model, userChoices):

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


    picture = openCamera(cam)
    picture = cv2.resize(picture, (540, 400), interpolation=cv2.INTER_LANCZOS4)
    if picture is not None:
        results = model.predict(picture, classes=[0], verbose=False)


        # Edge Detect
        if userChoices['EdgeDetectOn']:
            grad_x = cv2.Sobel(picture, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
            grad_y = cv2.Sobel(picture, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
                
            abs_grad_x = cv2.convertScaleAbs(grad_x)
            abs_grad_y = cv2.convertScaleAbs(grad_y)
            picture = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


        for result in results:
            # Show KeyPoints
            if userChoices['KeyPointsOn']:
                for kp in result.keypoints:
                    for detection in kp.xy:
                        previousPoint = -1
                        for idx, (x, y) in enumerate(detection):
                            if x != 0 and y != 0:
                                cv2.circle(picture, center=(int(x), int(y)), radius=5, color=kpColors[idx], thickness=-1)
                                if previousPoint != -1:
                                    cv2.line(picture, (int(x), int(y)), previousPoint, color=kpColors[idx], thickness=2)
                                previousPoint = (int(x), int(y))
            # Show Bounding Box
            if userChoices['BoundingBoxOn']:
                for obj in result.boxes:
                    for x0, y0, x1, y1 in obj.xyxy:
                        cv2.rectangle(picture, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 0), thickness=2)


        picture_data = numpyToImg(picture)
        window['-IMAGE-'].update(data=picture_data)
        window.refresh()


def mainWindow():
    cam   = cv2.VideoCapture(0)

    cam.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    model = YOLO("yolo11n-pose.pt")

    userChoices = {
                    'BoundingBoxOn': False,
                    'KeyPointsOn': False,
                    'EdgeDetectOn': False
                }
    
    if not cam.isOpened():
        print("Cannot open camera")
        return

    # Initial placeholder image
    placeholder = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
    picture_data = numpyToImg(placeholder)

    layoutLeft  = [[sg.Image(data=picture_data, key='-IMAGE-')]]
    layoutMiddle = [
                    [sg.Button('Ligar caixas de detecção', key='-ENABLE-BOUNDING-BOX-'), sg.Button('Desligar caixas de detecção', key='-DISABLE-BOUNDING-BOX-')],
                    [sg.Button('Ligar marcadores de corpo', key='-ENABLE-KEYPOINTS-'), sg.Button('Desligar marcadores de corpo', key='-DISABLE-KEYPOINTS-')]
                ]
    
    layoutRight = [
                    [sg.Button('Ligar Detector de Bordas', key='-ENABLE-EDGE-DETECT-'), sg.Button('Desligar Detector de Bordas', key='-DISABLE-EDGE-DETECT-')]
    ]


    layout = [
                [
                    sg.vtop(sg.Column(layoutLeft)),   sg.VSeparator(),
                    sg.vtop(sg.Column(layoutMiddle)), sg.VSeparator(),
                    sg.vtop(sg.Column(layoutRight))
                ]
    ]

    window = sg.Window('Window Title', layout, resizable=False, finalize=True)

    # Event Loop
    while True:
        event, values = window.read(timeout=7)
        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        elif event == '-ENABLE-BOUNDING-BOX-':
            userChoices['BoundingBoxOn'] = True
        elif event == '-DISABLE-BOUNDING-BOX-':
            userChoices['BoundingBoxOn'] = False

        elif event == '-ENABLE-KEYPOINTS-':
            userChoices['KeyPointsOn'] = True
        elif event == '-DISABLE-KEYPOINTS-':
            userChoices['KeyPointsOn'] = False

        elif event == '-ENABLE-EDGE-DETECT-':
            userChoices['EdgeDetectOn'] = True
        elif event == '-DISABLE-EDGE-DETECT-':
            userChoices['EdgeDetectOn'] = False

        updateImage(window, cam, model, userChoices)

    window.close()
    cam.release()  # Release the camera when done


if __name__ == "__main__":
    mainWindow()
