import numpy as np
from ultralytics import YOLO
import cv2


# DONE change to custom trained model for sidewalks
# TODO get more data and use transfer learning
# DONE change input to video feed
# TODO change output to modified video feed with lines
# TODO test outside

old_model = YOLO('runs/segment/train6/weights/best.pt')
model = YOLO('runs/segment/train7/weights/best.pt')


def train():
    model.train(data='SidewalksDatasetV4/data.yaml', epochs=10, imgsz=640, save=True, save_period=5, device=0)


def predict(image, blackAndWhite, useNew):
    if useNew:
        results = model(image)
    else:
        results = old_model(image)

    # creates a gray, (210,210,210) RGB, 640x640 image
    blankImage = np.ones((640, 640, 3), np.uint8) * 210

    if useNew:
        cv2.putText(image, "V4 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115,230,0), 2, cv2.LINE_AA)
        cv2.putText(blankImage, "V4 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, "V3 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)
        cv2.putText(blankImage, "V3 Data Model", (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 0, 230), 2, cv2.LINE_AA)

    if blackAndWhite:
        cv2.putText(blankImage, "Blank Canvas", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 0, 115), 2, cv2.LINE_AA)
    coordsArr = []
    if results[0].masks is None:
        return image

    for coordPair in results[0].masks.xy[0]:
        coordsArr.append((int(coordPair[0]), int(coordPair[1])))
    print(coordsArr)
    x = 0


    # connecting all points except first and last with a line

    while x < len(coordsArr)-1:
        cv2.line(blankImage, coordsArr[x], coordsArr[x+1], (115, 230, 0), 5, cv2.LINE_4)
        cv2.line(image, coordsArr[x], coordsArr[x + 1], (115, 230, 0), 5, cv2.LINE_4)
        x += 1
    # connecting first and last point
    cv2.line(blankImage,coordsArr[0], coordsArr[len(coordsArr)-1],(115,230,0), 5, cv2.LINE_4 )
    cv2.line(image, coordsArr[0], coordsArr[len(coordsArr) - 1], (115, 230, 0), 5, cv2.LINE_4)

    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # cv2.imshow("image", blankImage)
    # cv2.waitKey(0)
    if not blackAndWhite:
        return image
    else:
        return blankImage


def video():
    videoCap = cv2.VideoCapture('testVideo.MOV')
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