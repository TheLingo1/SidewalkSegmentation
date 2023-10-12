import numpy as np
from ultralytics import YOLO
import cv2
import threading

# DONE change to custom trained model for sidewalks
# DONE get more data and use transfer learning
# DONE change input to video feed
# DONE change output to modified video feed with lines
# TODO test outside

old_model = YOLO('runs/segment/train6/weights/best.pt')
model = YOLO('runs/segment/train7/weights/best.pt')


def train():
    model.train(data='SidewalksDatasetV4/data.yaml', epochs=10, imgsz=640, save=True, save_period=5, device=0)


savedMask = None


def runPredict(image, useNew):
    global savedMask
    if useNew:
        results = model(image, stream=True, half=True)
    else:
        results = old_model(image, stream=True)
    for r in results:
        savedMask = r.masks
    return


predictCounter = 0


def predict(image, blackAndWhite, useNew):
    global predictCounter

    if predictCounter == 0:
        runPredict(image, useNew)
    elif (predictCounter % 3) == 0:
        t1 = threading.Thread(target=runPredict, args=(image, useNew))
        t1.start()
    # creates a gray, (210,210,210) RGB, 640x640 image
    blankImage = np.ones((640, 640, 3), np.uint8) * 210

    if useNew:
        cv2.putText(image, "V4 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
        cv2.putText(blankImage, "V4 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "V3 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(blankImage, "V3 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)

    if blackAndWhite:
        cv2.putText(blankImage, "Blank Canvas", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 0, 115), 2, cv2.LINE_AA)

    coordsArr = []

    masks = savedMask

    if masks is None:
        predictCounter += 1
        return image

    for coordPair in masks.xy[0]:
        coordsArr.append((int(coordPair[0]), int(coordPair[1])))
    print(coordsArr)
    x = 0

    centered = False

    parallelCoords = []
    for coord in coordsArr:
        if abs(coord[1] - 320) <= 5:
            parallelCoords.append(coord)
    print(parallelCoords)

    if len(parallelCoords) != 0:

        # x value of left edge
        leftSide = parallelCoords[0][0]
        # x value of right edge
        rightSide = (parallelCoords[len(parallelCoords) - 1][0])

        if (320 - ((leftSide + rightSide) / 2)) <= 10:
            centered = True

        if centered:
            cv2.putText(image, "Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
            cv2.putText(blankImage, "Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Not Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)
            cv2.putText(blankImage, "Not Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2,
                        cv2.LINE_AA)

        cv2.line(image, parallelCoords[0], parallelCoords[len(parallelCoords) - 1], (230, 50, 115), 5, cv2.LINE_4)

    # connecting all points except first and last with a line
    while x < len(coordsArr) - 1:
        cv2.line(blankImage, coordsArr[x], coordsArr[x + 1], (115, 230, 0), 5, cv2.LINE_4)
        cv2.line(image, coordsArr[x], coordsArr[x + 1], (115, 230, 0), 5, cv2.LINE_4)
        x += 1
    # connecting first and last point
    cv2.line(blankImage, coordsArr[0], coordsArr[len(coordsArr) - 1], (115, 230, 0), 5, cv2.LINE_4)
    cv2.line(image, coordsArr[0], coordsArr[len(coordsArr) - 1], (115, 230, 0), 5, cv2.LINE_4)

    predictCounter += 1
    if not blackAndWhite:
        return image
    else:
        return blankImage


def video():
    videoCap = cv2.VideoCapture('15minuteWalk.MOV')
    blackAndWhite = False
    useNew = True
    while videoCap.isOpened():
        returnVal, img = videoCap.read()
        if returnVal:
            img = cv2.resize(img, (640, 640))
            editedImg = predict(img, blackAndWhite, useNew)

            cv2.imshow('image', editedImg)

            key = cv2.waitKey(16)
            if key == ord('q'):
                break
            elif key == ord('x'):
                blackAndWhite = not blackAndWhite
                print("BLACK AND WHITE")
            elif key == ord('c'):
                useNew = not useNew
                print("MODEL SWITCHED")
        else:
            break

    videoCap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video()
