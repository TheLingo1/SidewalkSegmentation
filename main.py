import numpy as np
from ultralytics import YOLO
import cv2
import threading

# DONE change to custom trained model for sidewalks
# DONE get more data and use transfer learning
# DONE change input to video feed
# DONE change output to modified video feed with lines
# DONE add drawing to black and white
# TODO test outside

old_model = YOLO('V4DataModel.pt')
model = YOLO('runs/segment/train8/weights/best.pt', task="segment")
# model = YOLO("V4DataModel.onnx", task="segment")


def export():
    model.export(format="tflite")


def train():
    model.train(data='datasets/Sidewalks-5/data.yaml', epochs=10, imgsz=320, save=True, save_period=5)


savedMask = None

predictCounter = 0


def predict(image, blackAndWhite, useNew, usePolylines):
    masks = None
    if useNew:
        results = model(image, stream=True)
    else:
        results = old_model(image, stream=True)

    # creates a gray, (210,210,210) RGB, 640x640 image
    blankImage = np.ones((640, 640, 3), np.uint8) * 210
    if blackAndWhite:
        image = blankImage

    if useNew:
        cv2.putText(image, "V5 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "V4 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)


    coordsArr = []
    for r in results:
        masks = r.masks
    if masks is None:
        return image

    for coordPair in masks.xy[0]:
        coordsArr.append([int(coordPair[0]), int(coordPair[1])])
    print(coordsArr)
    x = 0

    centered = False

    parallelCoords = []
    for coord in coordsArr:
        if abs(coord[1] - 320) <= 5:
            parallelCoords.append(coord)
    print(parallelCoords)
    midpoint = None
    if len(parallelCoords) != 0:

        # x value of left edge
        leftSide = parallelCoords[0][0]
        # x value of right edge
        rightSide = (parallelCoords[len(parallelCoords) - 1][0])
        midpoint = ((leftSide + rightSide) / 2)
        if abs(320 - midpoint) <= 50:
            centered = True

        if centered:
            cv2.putText(image, "Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
        else:
            cv2.putText(image, "Not Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)

        cv2.line(image, parallelCoords[0], parallelCoords[len(parallelCoords) - 1], (230, 50, 115), 5, cv2.LINE_4)

    coordsNP = np.array(coordsArr, np.int32)
    print("SHAPE: ", coordsNP.shape)
    ymax = np.array(coordsNP.max(axis=1)).max()
    print("YMAX:", ymax)
    ymin = np.array(coordsNP.min(axis=1)).min()
    print("YMIN:", ymin)

    y = ymin
    leftArray = []
    rightArray = []
    while y < ymax:

        lidx = np.argsort(np.abs(coordsNP[:, 1] - y))[0]
        ridx = np.argsort(np.abs(coordsNP[:, 1] - y))[1]
        if abs(coordsNP[lidx][0] - coordsNP[ridx][0]) < 15:
            y += 1
            continue
        elif coordsNP[lidx][0] < coordsNP[ridx][0]:
            leftArray.append(coordsNP[lidx])
            rightArray.append(coordsNP[ridx])
        else:
            leftArray.append(coordsNP[ridx])
            rightArray.append(coordsNP[lidx])

        y += 1
    leftArray = np.array(leftArray)
    rightArray = np.array(rightArray)
    leftArray = [tuple(row) for row in leftArray]
    leftArray = np.unique(leftArray, axis=0)

    if (coordsNP.shape[0] < 150) or not usePolylines:
        # connecting all points except first and last with a line
        x = 0
        while x < len(coordsArr) - 1:
            cv2.line(blankImage, coordsArr[x], coordsArr[x + 1], (115, 230, 0), 5, cv2.LINE_AA)
            cv2.line(image, coordsArr[x], coordsArr[x + 1], (115, 230, 0), 5, cv2.LINE_AA)
            x += 1
    else:
        leftpoly = np.polyfit(leftArray[:, 1], leftArray[:, 0], 2)
        leftrange = np.linspace(ymin, ymax, 20)
        leftdomain = np.polyval(leftpoly, leftrange)

        x = 0
        while x < len(leftrange)-1:
            if leftrange[x] > ymin:
                x += 1
                break
            x += 1
        print("X: ", x, " leftRange[x]: ", leftrange[x])
        rightpoly = np.polyfit(rightArray[:, 1], rightArray[:, 0], 2)
        rightrange = np.linspace(ymin, ymax, 20)
        rightdomain = np.polyval(rightpoly, rightrange)

        leftpts = (np.asarray([leftdomain[x:], leftrange[x:]]).T).astype(np.int32)
        rightpts = (np.asarray([rightdomain[x:], rightrange[x:]]).T).astype(np.int32)
        cv2.polylines(image, [leftpts], False, (115, 230, 0), thickness=5, lineType=cv2.LINE_AA)
        cv2.polylines(image, [rightpts], False, (115, 230, 0), thickness=5, lineType=cv2.LINE_AA)
    return image


def video():
    videoCap = cv2.VideoCapture("15minutewalk.MOV")
    blackAndWhite = False
    useNew = True
    usePolylines = False
    while videoCap.isOpened():
        returnVal, img = videoCap.read()
        if returnVal:
            img = cv2.resize(img, (640, 640))
            editedImg = predict(img, blackAndWhite, useNew, usePolylines)
            # editedImg = tfliteInference.inference(img)
            # editedImg = onnxInference.inference(img)
            cv2.imwrite('frame.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
            # editedImg = roboflowInference.inference('frame.jpg')
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
            elif key == ord('v'):
                usePolylines = not usePolylines
                print("LINES SWITCHED")
        else:
            break

    videoCap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video()
