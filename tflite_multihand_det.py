import cv2
from hand_tracker import HandTracker
import time
WINDOW = "Hand Tracking"
PALM_MODEL_PATH = "palm_detection_without_custom_op.tflite"
LANDMARK_MODEL_PATH = "hand_landmark.tflite"
ANCHORS_PATH = "anchors.csv"

POINT_COLOR = (0, 255, 0)
CONNECTION_COLOR = (255, 0, 0)
THICKNESS = 2


#        8   12  16  20
#        |   |   |   |
#        7   11  15  19
#    4   |   |   |   |
#    |   6   10  14  18
#    3   |   |   |   |
#    |   5---9---13--17
#    2    \         /
#     \    \       /
#      1    \     /
#       \    \   /
#        ------0-
connections = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17)
]

detector = HandTracker(
    PALM_MODEL_PATH,
    LANDMARK_MODEL_PATH,
    ANCHORS_PATH,
    box_shift=0.2,
    box_enlarge=1.3
)

def runModel(img, threshold):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    points = detector(image,threshold)

    output_json = list()
    for points_ in points:
        if points_ is not None:
            
            #save outputs into json
            model_output_json = {}
            model_output_json['x'] = list(points_[:,0])
            model_output_json['y'] = list(points_[:,1])
            for point in points_:
                x, y = point
                cv2.circle(img, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
            for connection in connections:
                x0, y0 = points_[connection[0]]
                x1, y1 = points_[connection[1]]
                cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)

            output_json.append(model_output_json)
    cv2.imshow(WINDOW, img)
    cv2.waitKey(0)
    return output_json
# cv2.namedWindow(WINDOW)
# capture = cv2.VideoCapture(0)

# if capture.isOpened():
#     hasFrame, frame = capture.read()
# else:
#     hasFrame = False
# while hasFrame:
#     image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     start_t = time.time()
#     points = detector(image, 0.7)
#     stop_t = time.time()
#     print("elapsed time:",stop_t-start_t)
#     for points_ in points:
#         if points_ is not None:
#             for point in points_:
#                 x, y = point
#                 cv2.circle(frame, (int(x), int(y)), THICKNESS * 2, POINT_COLOR, THICKNESS)
#             for connection in connections:
#                 x0, y0 = points_[connection[0]]
#                 x1, y1 = points_[connection[1]]
#                 cv2.line(frame, (int(x0), int(y0)), (int(x1), int(y1)), CONNECTION_COLOR, THICKNESS)
#     cv2.imshow(WINDOW, frame)
#     hasFrame, frame = capture.read()
#     key = cv2.waitKey(1)
#     if key == 27:
#         break

# capture.release()
# cv2.destroyAllWindows()

frame = cv2.imread("multi_hands.png")
output_json = runModel(frame,0.8)