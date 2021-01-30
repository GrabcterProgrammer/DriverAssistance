import cv2
import numpy as np

# Setting
img_size = [400, 200]
area_setting = np.float32([
    [40, 180],  # x1 + y1 - Нижняя левая точка
    [390, 180],  # x2 + y2 - Нижняя правая точка
    [280, 100],  # x3 + y3 - Верхняя правая точка
    [120, 100]    # x4 + y4 - Верхняя левая точка
])
cords = np.float32([
    [0, img_size[1]],
    [img_size[0], img_size[1]],
    [img_size[0], 1],
    [0, 0]
])
draw_area = np.array(area_setting, dtype=np.int32)

video = cv2.VideoCapture("video/DriveOnRoad.mp4")


# function
def binary(img, value):
    channel = img[:, :, 2]
    bin = np.zeros_like(channel)
    bin[(channel > value)] = 255

    return bin


while video.isOpened():
    _, frame = video.read()
    frame = cv2.resize(frame, (img_size[0], img_size[1]))
    frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

    # Бинарезуем изображение
    binary_frame = binary(frame, 200)
    binary_hls = binary(frame_hls, 160)

    all_binary = np.zeros_like(binary_frame)
    all_binary[((binary_frame == 255) | (binary_hls == 255))] = 255

    # crop video
    all_binary_visual = all_binary.copy()
    cv2.polylines(all_binary_visual, [draw_area], True, 255)
    M = cv2.getPerspectiveTransform(area_setting, cords)
    warped = cv2.warpPerspective(all_binary, M, (img_size[0], img_size[1]), flags=cv2.INTER_LINEAR)

    # search while pix
    warped_visual = warped.copy()
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0]//2
    left_column = np.argmax(histogram[:midpoint])
    right_column = np.argmax(histogram[midpoint:]) + midpoint
    cv2.line(warped_visual, (left_column, 0), (left_column, warped_visual.shape[0]), 110, 2)
    cv2.line(warped_visual, (right_column, 0), (right_column, warped_visual.shape[0]), 110, 2)

    # Вывод изображений
    # cv2.imshow("Main Video", frame)
    # cv2.imshow("Binary video", binary_frame)
    # cv2.imshow("BinaryHLS video", binary_hls)
    # cv2.imshow("AllBinary video", all_binary)
    # cv2.imshow("AllBinary area video", all_binary_visual)
    cv2.imshow("Warped img", warped)
    cv2.imshow("Warped_visual img", warped_visual)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()