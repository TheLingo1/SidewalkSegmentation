import multiprocessing
import cv2
from ultralytics import YOLO
import numpy as np

def capture_frames(frame_buffer):
    videoCapture = cv2.VideoCapture("shortCurve.MOV")
    while True:
        ret, image = videoCapture.read()
        if not ret:
            break
        image = cv2.resize(image, (640, 640))
        frame_buffer.put(image)

def inference(frame_buffer, result_buffer):
    model = YOLO("V4DataModel.pt")
    while True:
        image = frame_buffer.get()
        results = model(image, stream=True)

        masks = None
        for r in results:
            masks = r.masks
        if masks is None:
            masks = image

        result_buffer.put((image, masks))


def post_process(result_buffer):
    while True:
        image, masks = result_buffer.get()
        blankImage = np.ones((640, 640, 3), np.uint8) * 210
        coordsArr = []
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
        if len(coordsArr) >= 2:
            cv2.line(blankImage, coordsArr[0], coordsArr[len(coordsArr) - 1], (115, 230, 0), 5, cv2.LINE_4)
            cv2.line(image, coordsArr[0], coordsArr[len(coordsArr) - 1], (115, 230, 0), 5, cv2.LINE_4)

        cv2.imshow("image", image)
        key = cv2.waitKey(2)
        if key == ord('q'):
            break

if __name__ == "__main__":
    frame_buffer = multiprocessing.Queue()
    result_buffer = multiprocessing.Queue()

    capture_process = multiprocessing.Process(target=capture_frames, args=(frame_buffer,))
    inference_process = multiprocessing.Process(target=inference, args=(frame_buffer,result_buffer))
    post_processing_process = multiprocessing.Process(target=post_process, args=(result_buffer,))

    capture_process.start()
    inference_process.start()
    post_processing_process.start()

    capture_process.join()
    inference_process.join()
    post_processing_process.join()