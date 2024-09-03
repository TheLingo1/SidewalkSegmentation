import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
from shapely.geometry import Polygon

model = YOLO("LatestSidewalk.pt", task='segment') 

# Open the video file
cap = cv2.VideoCapture('StraightSidewalk.mov')

# Define the four corner points of the region to be transformed
# These points should be in the order: top-left, top-right, bottom-right, bottom-left
# Define the desired output dimensions and the corresponding corner points
width, height = 600, 1080
# Create a VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (width, height))
skips = 0
def select_corners(polygon, oldCoords, min_side_length=100):
    """Selects 4 points that form a square from the bottom of the mask, with at least min_side_length pixels per side."""
    minx, miny, maxx, maxy = polygon.bounds
    
    # Find the bottom edge of the polygon
    coordinates = np.array(polygon.exterior.coords)
    bottom_edge = np.array(polygon.exterior.coords)
    bottom_edge_idx = np.argmax([coord[1] for coord in polygon.exterior.coords])
    bottom_edge = bottom_edge[np.argmax([coord[1] for coord in polygon.exterior.coords])]
    print("Bottom edge index", bottom_edge_idx)
    print("bottom edge: ", bottom_edge)
    print("Polygon coords: ", coordinates)
    split_coords = np.vsplit(coordinates, [bottom_edge_idx+1])
    print("Split coords", split_coords)
    left_coords = split_coords[0]
    right_coords = split_coords[1]
    print("minx: %i, miny: %i, maxx: %i, maxy: %i" % (minx,miny,maxx,maxy))
    
    toFindTopLeft = [bottom_edge[0], bottom_edge[1]-500]
    toFindTopRight = [bottom_edge[0]+500, bottom_edge[1]-500]
    toFindBottomRight = [bottom_edge[0]+500, bottom_edge[1]]
    toFindBottomLeft = [bottom_edge[0], bottom_edge[1]]

    idxTopLeft = np.sum( (left_coords-toFindTopLeft)**2, axis=1, keepdims=True).argmin(axis=0)
    top_left = left_coords[idxTopLeft]
    print("Found Top Left: ", left_coords[idxTopLeft])

    idxTopRight = np.sum( (right_coords-toFindTopRight)**2, axis=1, keepdims=True).argmin(axis=0)
    top_right = right_coords[idxTopRight]
    print("Found Top Right: ", right_coords[idxTopRight])
    
    idxBottomRight = np.sum( (right_coords-toFindBottomRight)**2, axis=1, keepdims=True).argmin(axis=0)
    bottom_right = right_coords[idxBottomRight]
    print("Found Bottom Right: ", right_coords[idxBottomRight])
    
    idxBottomLeft = np.sum( (left_coords-toFindBottomLeft)**2, axis=1, keepdims=True).argmin(axis=0)
    bottom_left = left_coords[idxBottomLeft]
    print("Found Bottom Left: ", left_coords[idxBottomLeft])
    global skips
    if pts1 is None:
        pass
    elif (abs(pts1[2][0][0]-bottom_right[0][0]) > 10) and skips < 15:
        
        skips += 1
        print("SKIPPED: ", skips)
        return pts1
    elif (abs(pts1[3][0][0]-bottom_left[0][0]) > 10) and skips < 15:
        
        skips += 1
        print("SKIPPED: ", skips)
        return pts1
    else:
        print("DIFF ",pts1[2][0][0]-bottom_right[0][0])

    skips = 0
    return [top_left, top_right, bottom_right, bottom_left]

# Process each frame of the video
counter = 0
pts1 = None
while True:

    ret, frame = cap.read()
    if not ret:
        break

    if (counter % 3) == 0:
        results = model(frame)
        

        top_down_canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # Process each detected object
        for result in results:
            # Get the segmentation mask coordinates
            mask_coords = result.masks.xy.pop() 
            
            # Create a polygon from the mask coordinates
            polygon = Polygon(mask_coords) 

            # Select four corners close to each other
            corners = select_corners(polygon, pts1, 200)
            print(corners[1][0])
            frame = cv2.circle(frame, np.array(corners[0][0]).astype(int), 20, (255,0,0), -1)
            frame = cv2.circle(frame, np.array(corners[1][0]).astype(int), 20, (0,255,0), -1)
            frame = cv2.circle(frame, np.array(corners[2][0]).astype(int), 20, (0,0,255), -1)
            frame = cv2.circle(frame, np.array(corners[3][0]).astype(int), 20, (255,255,0), -1)
            #[(264.0, 0.0), (1242.0, 0.0), (1242.0, 1062.0), (264.0, 1062.0)]
            pts1 = np.float32([corners[0], corners[1], corners[2], corners[3]])
            # +(corners[1][1]-corners[0][1])
            print(corners[1][0][1])
        pts2 = np.float32([[(width/2)-250, height-500], [(width/2)+250, height+(-500-(corners[0][0][1]-corners[1][0][1]))], [(width/2)+250, height-(corners[3][0][1]-corners[2][0][1])], [(width/2)-250, height]])

        # Calculate the perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # Apply the perspective transformation to the frame
    output = cv2.warpPerspective(frame, matrix, (width, height))

    # Write the transformed frame to the output video
    out.write(output)
    cv2.imshow("Original Frame", frame)
    # Display the output frame (optional)
    cv2.imshow('Top-Down View', output)
    if cv2.waitKey(1) == ord('q'):
        break
    counter += 1
# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()