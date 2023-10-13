import queue
import cv2
from ultralytics import YOLO
import numpy as np
import threading


quit = False

def capture_frames(frame_buffer):
    global quit
    videoCapture = cv2.VideoCapture("15minuteWalk.MOV")
    while True:
        if quit:
            break
        ret, image = videoCapture.read()
        if not ret:
            break
        image = cv2.resize(image, (640, 640))
        frame_buffer.put(item=image)


def inference(frame_buffer, result_buffer):
    global quit
    model = YOLO("V4DataModel.pt")
    while True:
        if quit:
            break
        image = frame_buffer.get()
        results = model(image, stream=True)

        masks = None
        for r in results:
            masks = r.masks

        result_buffer.put((image, masks))


def post_process(result_buffer):
    global quit
    while True:
        if quit:
            break
        image = result_buffer.get()[0]
        masks = result_buffer.get()[1]
        blankImage = np.ones((640, 640, 3), np.uint8) * 210
        coordsArr = []

        if masks is None:
            cv2.imshow("image", image)
            key = cv2.waitKey(16)
            if key == ord('q'):
                quit = True
            continue

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
                cv2.putText(blankImage, "Centered", (0, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (115, 230, 0), 2,
                            cv2.LINE_AA)
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
        key = cv2.waitKey(16)
        if key == ord('q'):
            quit = True


if __name__ == "__main__":
    # torch.multiprocessing.set_start_method("spawn")
    frame_buffer = queue.Queue()
    result_buffer = queue.Queue()
    # lock = multiprocessing.Lock()

    capture_process = threading.Thread(target=capture_frames, args=(frame_buffer,))
    inference_process = threading.Thread(target=inference, args=(frame_buffer, result_buffer))
    post_processing_process = threading.Thread(target=post_process, args=(result_buffer,))

    capture_process.start()
    inference_process.start()
    post_processing_process.start()

    capture_process.join()
    inference_process.join()
    post_processing_process.join()
