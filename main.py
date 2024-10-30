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
    center  = (w // 2, h // 2)
    M       = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LANCZOS4)

    return rotated


def applyFilters(picture, userChoices, detectedBodyParts, detection):
    if 0 in detectedBodyParts and 1 in detectedBodyParts and 2 in detectedBodyParts:
        leftEye  = detection[2]
        rightEye = detection[1]

        leftEyeX, leftEyeY   = int(leftEye[0]), int(leftEye[1])
        rightEyeX, rightEyeY = int(rightEye[0]), int(rightEye[1])


        for currentFilter in userChoices['Filters']:
            if userChoices['Filters'][currentFilter]['On']:
                overlay     = cv2.imread(userChoices['Filters'][currentFilter]['PngLocation'], cv2.IMREAD_UNCHANGED)
                b, g, r, a  = cv2.split(overlay)
                overlay     = cv2.merge((b, g, r))
                overlayMask = a.astype(float) / 255.0

                # referencePoint é definido porque cada filtro depende de um lugar pra ser aplicado.
                # O oculos por exemplo usa o olho direito como referencia, porque os oculos vao nos olhos, e facilita as contas dos offsets.
                # O bigode ja usa o nariz como referencia, porque aí sabendo a posição do nariz, é só descer um pouco no eixo Y que chegamos na regiao
                # certa para o bigode. Os reference points estão definidos para cada filtro no json userChoices.
                referencePoint = detection[userChoices['Filters'][currentFilter]['ReferencePoint']]
                referencePointX, referencePointY = int(referencePoint[0]), int(referencePoint[1])


                # distancia que olho esquerdo está do direito / distancia que deveria ser (em px) 
                #(100px é +- a distancia do centro de uma lente para a outra no .png)
                downsampleFactor = (rightEyeX - leftEyeX) / userChoices['Filters'][currentFilter]['IdealDistance']

                resX = int(overlay.shape[1] * downsampleFactor)
                resY = int(overlay.shape[0] * downsampleFactor)

                offsetX = int(userChoices['Filters'][currentFilter]['OffsetX'] * downsampleFactor)
                offsetY = int(userChoices['Filters'][currentFilter]['OffsetY'] * downsampleFactor)

                angle = -np.degrees(np.arctan2(rightEyeY - leftEyeY, rightEyeX - leftEyeX))

                try:
                    overlayResized      = cv2.resize(overlay, (resX, resY), interpolation=cv2.INTER_LANCZOS4)
                    overlayMaskResized  = cv2.resize(overlayMask, (resX, resY), interpolation=cv2.INTER_LANCZOS4)

                    overlayResized             = rotateImage(overlayResized, angle)
                    overlayMaskResized         = rotateImage(overlayMaskResized, angle)
                    roi                 = picture[referencePointY - offsetY:referencePointY - offsetY + resY, referencePointX - offsetX:referencePointX - offsetX + resY]

                    # Multiplica os pixels pela camada alpha para aplicar transparencia
                    for c in range(3):
                        # WTF
                        roi[:, :, c] = roi[:, :, c] * (1 - overlayMaskResized) + overlayResized[:, :, c] * overlayMaskResized

                    # Adiciona à foto o resultado final da roi, que é o overlay do óculos (rotacionado ou nao), com a camada alpha aplicada
                    picture[referencePointY - offsetY:referencePointY - offsetY + resY, referencePointX - offsetX:referencePointX - offsetX + resY] = roi

                except Exception as e:
                    print(e)
        
    return picture


def updateImage(window, cam, model, userChoices):

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

        if userChoices['SwirlOn']:
            picture = swirlEffect(picture, userChoices['SwirlAmount'])


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


                    # Applies instagram-like filters
                    picture = applyFilters(picture, userChoices, detectedBodyParts, detection)
                        

            # Show Bounding Box
            if userChoices['BoundingBoxOn']:
                for obj in result.boxes:
                    for (x0, y0, x1, y1), detectedClass, conf in zip(obj.xyxy, obj.cls, obj.conf):
                        cv2.rectangle(picture, (int(x0), int(y0)), (int(x1), int(y1)), color=(255, 255, 0), thickness=2)
                        cv2.putText(picture, f"{model.names[detectedClass.item()]} - {int(conf * 100)}%", (int(x0), int(y0) -15), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)


        if userChoices['ProjectionOn']:
            picture = reprojectionEffect(picture)


        # Edge Detect
        if userChoices['EdgeDetectOn']:
            picture = scharrEdgeDetect(picture)
        

        picture = cv2.resize(picture, (int(640*1.2), int(480 * 1.2)), interpolation=cv2.INTER_LANCZOS4)

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
                    'SwirlAmount': 3,
                    'Filters': 
                                {
                                    'Glasses': 
                                            {
                                                'On': False,
                                                'PngLocation': "./assets/glasses1.png",
                                                'IdealDistance': 60,
                                                'ReferencePoint': 2, # Olho direito
                                                'OffsetX': 58,
                                                'OffsetY': 85
                                            },
                                    'Mustache':
                                            {
                                                'On': False,
                                                'PngLocation': "./assets/mustache.png",
                                                'IdealDistance': 100,
                                                'ReferencePoint': 0, # Nariz
                                                'OffsetX': 75,
                                                'OffsetY': 35
                                            },
                                    'Cap':
                                            {
                                                'On': False,
                                                'PngLocation': "./assets/lakers.png",
                                                'IdealDistance': 55,
                                                'ReferencePoint': 2, # OlhoDireito
                                                'OffsetX': 122,
                                                'OffsetY': 260
                                            },
                                    'ClownNose':
                                            {
                                                'On': False,
                                                'PngLocation': "./assets/clown.png",
                                                'IdealDistance': 250,
                                                'ReferencePoint': 0, # Nariz
                                                'OffsetX': 75,
                                                'OffsetY': 0
                                            }
                                }
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
                    [sg.Button('Ligar filtro de bigode', key='-ENABLE-MUSTACHE-'), sg.Button('Desligar filtro de bigode', key='-DISABLE-MUSTACHE-')],
                    [sg.Button('Ligar filtro de nariz de palhaço', key='-ENABLE-CLOWN-'), sg.Button('Desligar filtro de nariz de palhaço', key='-DISABLE-CLOWN-')],
                    [sg.Button('Ligar filtro de boné', key='-ENABLE-CAP-'), sg.Button('Desligar filtro de boné', key='-DISABLE-CAP-')]
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

    window = sg.Window('Window Title', layout, resizable=True)

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
            userChoices['Filters']['Glasses']['On'] = True
        elif event == '-DISABLE-GLASSES-':
            userChoices['Filters']['Glasses']['On'] = False
        
        elif event == '-ENABLE-MUSTACHE-':
            userChoices['Filters']['Mustache']['On'] = True
        elif event == '-DISABLE-MUSTACHE-':
            userChoices['Filters']['Mustache']['On'] = False

        elif event == '-ENABLE-CLOWN-':
            userChoices['Filters']['ClownNose']['On'] = True
        elif event == '-DISABLE-CLOWN-':
            userChoices['Filters']['ClownNose']['On'] = False

        elif event == '-ENABLE-CAP-':
            userChoices['Filters']['Cap']['On'] = True
        elif event == '-DISABLE-CAP-':
            userChoices['Filters']['Cap']['On'] = False


        userChoices['SwirlAmount'] = values['-SWIRL-STRENGTH-']

        updateImage(window, cam, model, userChoices)

    window.close()
    cam.release()  # Release the camera when done


if __name__ == "__main__":
    mainWindow()
