import cv2
import numpy as np

# Setting

setting = {
    # red
    "lower_red": np.array([0, 85, 110], dtype="uint8"),
    "upper_red": np.array([15, 255, 255], dtype="uint8"),

    # green
    "lower_green": np.array([40, 85, 110], dtype="uint8"),
    "upper_green": np.array([91, 255, 255], dtype="uint8")
}

video = cv2.VideoCapture("video/test.mp4")


def search_color(hsv, lower_color, upper_color):
    mask_frame = cv2.inRange(hsv, lower_color, upper_color)

    st1 = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21), (10, 10))
    st2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11), (5, 5))

    mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_CLOSE, st1)
    mask_frame = cv2.morphologyEx(mask_frame, cv2.MORPH_CLOSE, st2)

    # Gaussian blur
    mask_frame = cv2.GaussianBlur(mask_frame, (5, 5), 2)

    _, dif_frame = cv2.threshold(mask_frame, 127, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(hsv, hsv, mask=dif_frame)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    return dif_frame, result


while True:
    success, frame = video.read()
    if not success:
        print("Видеопоток закончился")
        break

    frame = cv2.resize(frame, (400, 300))

    frame = frame[100:400, 0:400]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    red_dif_frame, result_red = search_color(hsv, setting["lower_red"], setting["upper_red"])
    green_dif_frame, result_green = search_color(hsv, setting["lower_green"], setting["upper_green"])

    cv2.imshow("hsv", hsv)

    cv2.imshow("red filter", red_dif_frame)
    cv2.imshow("green filter", green_dif_frame)

    cv2.imshow("finally result red", result_red)
    cv2.imshow("finally result green", result_green)

    if cv2.waitKey(1) == ord('q'):
        break


video.release()
cv2.destroyAllWindows()