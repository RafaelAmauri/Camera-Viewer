import FreeSimpleGUI as sg
import cv2
import numpy as np
import io
from PIL import Image
import time
from ultralytics import YOLO
from PIL import Image, ImageFilter
import skimage.transform


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

def reprojectionEffect(picture):
    rows,cols,ch = picture.shape

    # x,y points are cw from top left
    src_interest_pts = np.float32([[0,0],[640,0],[640,480],[0,480]])
    Affine_interest_pts = np.float32([[551*640/1280,224*480/720],[843*640/1280,67*480/720],[903*640/1280,301*480/720],[608*640/1280,455*480/720]])
    Projective_interest_pts = np.float32([[195,56],[494,158],[432,498],[36,183]])

    M = cv2.estimateAffine2D(src_interest_pts ,Affine_interest_pts)[0]
    Affinedst = cv2.warpAffine(picture,M,(cols,rows))

    M=cv2.getPerspectiveTransform(src_interest_pts ,Projective_interest_pts)
    Projectivedst=cv2.warpPerspective(picture,M,(cols,rows))

    return Affinedst+Projectivedst


def swirlEffect(picture, amount):
    cx = 320
    cy = 240
    dist = 400
    angle = 0
    border_color = (128,128,128)

    return skimage.transform.swirl(picture, center=(cx,cy), rotation=angle, strength=amount, radius=dist, preserve_range=True).astype(np.uint8)


def scharrEdgeDetect(picture):
    picture = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)

    grad_x = cv2.Sobel(picture, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(picture, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    
    return cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)


def rotateImage(image, angle):
    (h, w)  = image.shape[:2]
    center = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

    return rotated

# TODO Terminar
def applyFilter(picture, userChoices, detectedBodyParts, detection):
    if 0 in detectedBodyParts and 1 in detectedBodyParts and 2 in detectedBodyParts:
        glasses     = cv2.imread("./assets/glasses.png", cv2.IMREAD_UNCHANGED)
        b, g, r, a  = cv2.split(glasses)
        glasses     = cv2.merge((b, g, r))
        maskGlasses = a.astype(float) / 255.0

        mustache     = cv2.imread('./assets/mustache.png', cv2.IMREAD_UNCHANGED)
        b, g, r, a   = cv2.split(mustache)
        mustache     = cv2.merge((b, g, r))
        maskMustache = a.astype(float) / 255.0

        topHat      = cv2.imread('./assets/tophat.png', cv2.IMREAD_UNCHANGED)
        b, g, r, a  = cv2.split(topHat)
        topHat      = cv2.merge((b, g, r))
        maskTopHat  = a.astype(float) / 255.0

        nose   = detection[0]
        noseX, noseY = int(nose[0]), int(nose[1])

        leftEye  = detection[2]
        rightEye = detection[1]

        leftEyeX, leftEyeY   = int(leftEye[0]), int(leftEye[1])
        rightEyeX, rightEyeY = int(rightEye[0]), int(rightEye[1])



def updateImage(window, cam, model, userChoices):

    glasses     = cv2.imread("./assets/glasses.png", cv2.IMREAD_UNCHANGED)
    b, g, r, a  = cv2.split(glasses)
    glasses     = cv2.merge((b, g, r))
    maskGlasses = a.astype(float) / 255.0

    mustache     = cv2.imread('./assets/mustache.png', cv2.IMREAD_UNCHANGED)
    b, g, r, a   = cv2.split(mustache)
    mustache     = cv2.merge((b, g, r))
    maskMustache = a.astype(float) / 255.0

    topHat      = cv2.imread('./assets/tophat.png', cv2.IMREAD_UNCHANGED)
    b, g, r, a  = cv2.split(topHat)
    topHat      = cv2.merge((b, g, r))
    maskTopHat  = a.astype(float) / 255.0


    keyPointColors = [
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


    bodyPartConnections = [
        (0, 2,   keyPointColors[3]),  # Nose -> Left Eye
        (2, 1,   keyPointColors[3]),  # Left Eye -> Right Eye
        (0, 1,   keyPointColors[3]),  # Nose -> Right Eye

        (6, 5,   keyPointColors[3]),  # Right Shoulder -> Left Shoulder

        (6, 8,   keyPointColors[3]),  # Right Shoulder -> Right Elbow
        (8, 10,  keyPointColors[3]),  # Right Elbow -> Right Hand

        (5, 7,   keyPointColors[3]),  # Left Shoulder -> Left Elbow
        (7, 9,   keyPointColors[0]),  # Left Elbow -> Left Hand

        (6, 12,  keyPointColors[3]),  # Right Shoulder -> Right Hip
        (12, 14, keyPointColors[3]),  # Right Hip -> Right Knee

        (5, 11,  keyPointColors[3]),  # Left Shoulder -> Left Hip
        (11, 13, keyPointColors[3]),  # Left Hip -> Left Knee

        (14, 16, keyPointColors[3]),  # Right Knee -> Right Foot

        (13, 15, keyPointColors[3]),  # Left Knee -> Left Foot
    ]


    picture = openCamera(cam)
    picture = cv2.resize(picture, (640, 480), interpolation=cv2.INTER_LANCZOS4)
    if picture is not None:
        results = model.predict(picture, verbose=False)

        if userChoices['ProjectionOn']:
            picture = reprojectionEffect(picture)

        if userChoices['SwirlOn']:
            picture = swirlEffect(picture, userChoices['SwirlAmount'])

        # Edge Detect
        if userChoices['EdgeDetectOn']:
            picture = scharrEdgeDetect(picture)
        

        for result in results:
            # Show KeyPoints
            for kp in result.keypoints:
                for detection in kp.xy:
                    previousPoint = -1
                    detectedBodyParts = []
                    for idx, (x, y) in enumerate(detection):
                        if x != 0 and y != 0:
                            detectedBodyParts.append(idx)

                            if userChoices['KeyPointsOn']:
                                cv2.circle(picture, center=(int(x), int(y)), radius=5, color=keyPointColors[idx], thickness=-1)

                                for part1, part2, color in bodyPartConnections:
                                    if part1 in detectedBodyParts and part2 in detectedBodyParts:
                                        x1, y1 = detection[part1]
                                        x2, y2 = detection[part2]
                                        cv2.line(picture, (int(x1), int(y1)), (int(x2), int(y2)), color=color, thickness=2)


                    if userChoices['GlassesOn']:
                        if 1 in detectedBodyParts and 2 in detectedBodyParts:
                            leftEye  = detection[2]
                            rightEye = detection[1]

                            leftEyeX, leftEyeY   = int(leftEye[0]), int(leftEye[1])
                            rightEyeX, rightEyeY = int(rightEye[0]), int(rightEye[1])
                            
                            # Para centralizar o olho no centro da lente, puxar 45px * downsampleFactor pra esquerda e 
                            # 60px * downsampleFactor pra cima
                            downsampleFactor = (rightEyeX-leftEyeX)/60 # distancia que olho esquerdo está do direito / distancia que deveria ser (em px)
                            resX = int(180 * downsampleFactor)
                            resY = int(180 * downsampleFactor)
                            offsetX = int(58 * downsampleFactor)
                            offsetY = int(85 * downsampleFactor)
                            angle = -np.degrees(np.arctan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX))
                            try:                
                                glassesRotated = rotateImage(glasses, angle)
                                maskRotated     = rotateImage(maskGlasses, angle)
                                glassesResized = cv2.resize(glassesRotated, (resX, resY), interpolation=cv2.INTER_LANCZOS4)
                                maskResized    = cv2.resize(maskRotated, (resX, resY), interpolation=cv2.INTER_LANCZOS4)
                                roi            = picture[leftEyeY - offsetY:leftEyeY - offsetY + resY, leftEyeX - offsetX:leftEyeX - offsetX + resY]

                                # Blend the glasses with the ROI respecting the alpha channel
                                for c in range(3):  # For each channel
                                    roi[:, :, c] = roi[:, :, c] * (1 - maskResized) + glassesResized[:, :, c] * maskResized


                                picture[leftEyeY - offsetY:leftEyeY - offsetY + resY, leftEyeX - offsetX:leftEyeX - offsetX + resY] = roi

                            except Exception as e:
                                print(e)

                    if userChoices['MustacheOn']:
                        if 0 in detectedBodyParts and 1 in detectedBodyParts and 2 in detectedBodyParts:
                            nose   = detection[0]
                            noseX, noseY = int(nose[0]), int(nose[1])

                            leftEye  = detection[2]
                            rightEye = detection[1]

                            leftEyeX, leftEyeY   = int(leftEye[0]), int(leftEye[1])
                            rightEyeX, rightEyeY = int(rightEye[0]), int(rightEye[1])
                            
                            
                            # Para centralizar o bigode abaixo do nariz, empurrar o y 52 px pra baixo
                            downsampleFactor = (rightEyeX-leftEyeX)/100 # distancia que olho esquerdo está do direito / distancia que deveria ser (em px)
                            distX = int(150 * downsampleFactor)
                            distY = int(150 * downsampleFactor)
                            offsetX = int(75 * downsampleFactor)
                            offsetY = int(35 * downsampleFactor)
                            angle = -np.degrees(np.arctan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX))

                            try:                
                                mustacheRotated = rotateImage(mustache, angle)
                                maskRotated     = rotateImage(maskMustache, angle)
                                mustacheResized = cv2.resize(mustacheRotated, (distX, distY), interpolation=cv2.INTER_LANCZOS4)
                                maskResized     = cv2.resize(maskRotated, (distX, distY), interpolation=cv2.INTER_LANCZOS4)
                                roi             = picture[noseY - offsetY:noseY - offsetY + distY, noseX - offsetX:noseX - offsetX + distX]
                                
                                # Blend the mustache with the ROI respecting the alpha channel
                                for c in range(3):  # For each channel
                                    roi[:, :, c] = roi[:, :, c] * (1 - maskResized) + mustacheResized[:, :, c] * maskResized

                                picture[noseY - offsetY:noseY - offsetY + distY, noseX - offsetX:noseX - offsetX + distX] = roi

                            except Exception as e:
                                print(e)
                        
                    #TODO FIX
                    '''
                    if True:#userChoices['TopHatOn']:
                        if 3 in detectedBodyParts and 4 in detectedBodyParts:
                            leftEye  = detection[4]
                            rightEye = detection[3]

                            leftEyeX, leftEyeY   = int(leftEye[0]), int(leftEye[1])
                            rightEyeX, rightEyeY = int(rightEye[0]), int(rightEye[1])
                            
                            
                            # Para centralizar o bigode abaixo do nariz, empurrar o y 52 px pra baixo
                            downsampleFactor = (rightEyeX-leftEyeX)/140 # distancia que olho esquerdo está do direito / distancia que deveria ser (em px)
                            distX = int(250 * downsampleFactor)
                            distY = int(250 * downsampleFactor)
                            offsetX = int(50 * downsampleFactor)
                            offsetY = int(240 * downsampleFactor)
                            angle = -np.degrees(np.arctan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX))

                            try:                
                                topHatRotated   = rotateImage(topHat, angle)
                                maskRotated     = rotateImage(maskTopHat, angle)
                                topHatResized   = cv2.resize(topHat, (distX, distY), interpolation=cv2.INTER_LANCZOS4)
                                maskResized     = cv2.resize(maskRotated, (distX, distY), interpolation=cv2.INTER_LANCZOS4)
                                roi             = picture[leftEyeY - offsetY:leftEyeY - offsetY + distY, leftEyeX - offsetX:leftEyeX - offsetX + distX]
                                
                                # Blend the mustache with the ROI respecting the alpha channel
                                for c in range(3):  # For each channel
                                    roi[:, :, c] = roi[:, :, c] * (1 - maskResized) + topHatResized[:, :, c] * maskResized

                                picture[leftEyeY - offsetY:leftEyeY - offsetY + distY, leftEyeX - offsetX:leftEyeX - offsetX + distX] = roi

                            except Exception as e:
                                print(e)
                    '''

            # Show Bounding Box
            if userChoices['BoundingBoxOn']:
                for obj in result.boxes:
                    for (x0, y0, x1, y1), detectedClass, conf in zip(obj.xyxy, obj.cls, obj.conf):
                        cv2.rectangle(picture, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 0), thickness=2)
                        cv2.putText(picture, f"{model.names[detectedClass.item()]} - {int(conf * 100)}%", (int(x0), int(y0) -15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

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
                    'EdgeDetectOn': False,
                    'SwirlOn': False,
                    'ProjectionOn': False,
                    'GlassesOn': False,
                    'MustacheOn': False,
                    'SwirlAmount': 3
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
                    [sg.Button('Ligar marcadores de corpo', key='-ENABLE-KEYPOINTS-'), sg.Button('Desligar marcadores de corpo', key='-DISABLE-KEYPOINTS-')],
                    [sg.Button('Ligar filtro de óculos', key='-ENABLE-GLASSES-'), sg.Button('Desligar filtro de óculos', key='-DISABLE-GLASSES-')],
                    [sg.Button('Ligar filtro de bigode', key='-ENABLE-MUSTACHE-'), sg.Button('Desligar filtro de bigode', key='-DISABLE-MUSTACHE-')]
                ]
    
    layoutRight = [
                    [sg.Button('Ligar Detector de Bordas', key='-ENABLE-EDGE-DETECT-'), sg.Button('Desligar Detector de Bordas', key='-DISABLE-EDGE-DETECT-')],
                    [sg.Button('Ligar Efeito Redemoinho', key='-ENABLE-SWIRL-'), sg.Button('Desligar Efeito Redemoinho', key='-DISABLE-SWIRL-')],
                    [sg.Text('Força do Efeito Redemoinho')], 
                    [sg.Slider((1, 6), orientation="horizontal", tick_interval=2, key='-SWIRL-STRENGTH-')],
                    [sg.Button('Ligar Efeito de Projeção', key='-ENABLE-PROJECTION-'), sg.Button('Desligar Efeito de Projeção', key='-DISABLE-PROJECTION-')]
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

        elif event == '-ENABLE-SWIRL-':
            userChoices['SwirlOn'] = True
        elif event == '-DISABLE-SWIRL-':
            userChoices['SwirlOn'] = False

        elif event == '-ENABLE-PROJECTION-':
            userChoices['ProjectionOn'] = True
        elif event == '-DISABLE-PROJECTION-':
            userChoices['ProjectionOn'] = False
        
        elif event == '-ENABLE-GLASSES-':
            userChoices['GlassesOn'] = True
        elif event == '-DISABLE-GLASSES-':
            userChoices['GlassesOn'] = False
        
        elif event == '-ENABLE-MUSTACHE-':
            userChoices['MustacheOn'] = True
        elif event == '-DISABLE-MUSTACHE-':
            userChoices['MustacheOn'] = False


        userChoices['SwirlAmount'] = values['-SWIRL-STRENGTH-']


        updateImage(window, cam, model, userChoices)

    window.close()
    cam.release()  # Release the camera when done


if __name__ == "__main__":
    mainWindow()
