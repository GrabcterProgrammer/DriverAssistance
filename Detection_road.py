import cv2
import numpy as np

# Setting
SETTING = {
    "size": [1300, 800],  # img size
}

video = cv2.VideoCapture("video/DriveOnRoad.mp4")

if not video.isOpened():
    print("Видео поток закончился")


# Function from detection
def canny(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    blur = cv2.GaussianBlur(img, (5, 5), 0)

    return cv2.Canny(blur, 50, 150)  # recommend param 1 to 2 or 1 to 3


def make_coordinates(image, line_parameters):
    # Y = MX + B
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3 / 5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []

    while lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_coordinates(image, left_fit_average)
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_coordinates(image, right_fit_average)
        return np.array([left_line, right_line])


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for x1, y1, x2, y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 8)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([(450, height // 2), (900, height // 2), (1400, 700), (0, 700)])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, np.array([polygons], dtype=np.int64), 1024)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


# main
while video.isOpened():
    _, frame = video.read()
    frame = cv2.resize(frame, (SETTING["size"][0], SETTING["size"][1]))

    try:
        canny_frame = canny(frame)
        canny_frame = region_of_interest(canny_frame)

        lines = cv2.HoughLinesP(canny_frame, 2, np.pi / 180, 100, np.array([()]), minLineLength=20, maxLineGap=5)
        average_lines = average_slope_intercept(frame, lines)
        line_image = display_lines(frame, average_lines)
        combo = cv2.addWeighted(frame, 0.8, line_image, 0.5, 1)

        # cv2.imshow("Main Video", frame)
        # cv2.imshow("Canny Video", canny_frame)
        cv2.imshow("Combo Video", combo)

    except:
        pass

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Закрываю видеопоток")
        video.release()
        cv2.destroyAllWindows()
