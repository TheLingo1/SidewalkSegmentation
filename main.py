from ultralytics import YOLO
import cv2
model = YOLO('yolov8n-seg.pt')

# TODO change to custom trained model for sidewalks
# TODO change input to video feed
# TODO change output to modified video feed with lines
# TODO test outside

# model.train(data='/Users/gabjimmy/Documents/programming/PyCharmProjects/SidewalkSegmentation/datasets/Sidewalks-1/data.yaml', epochs=1, imgsz=640)

results = model.predict('testLight.jpg')
image = cv2.imread('testLight.jpg')
# print(results[0].masks.xy)
coordsArr = []
for coordPair in results[0].masks.xy[0]:
    coordsArr.append((int(coordPair[0]), int(coordPair[1])))
print(coordsArr)
x = 0

# connecting all points except first and last with a line
while x < len(coordsArr)-1:
    cv2.line(image, coordsArr[x], coordsArr[x+1], (115,230,0), 50, cv2.LINE_4)
    x += 1
# connecting first and last point
cv2.line(image,coordsArr[0], coordsArr[len(coordsArr)-1],(115,230,0), 50, cv2.LINE_4 )
cv2.imshow("image", image)
cv2.waitKey(0)
