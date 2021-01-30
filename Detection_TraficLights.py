import cv2
import numpy as np

frameWidth = 200  # РАЗРЕШЕНИЕ КАМЕРЫ
frameHeight = 200
brightness = 180

cap = cv2.VideoCapture("video/TraficLights.mp4")
# cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

def red_mask(frame):
    # красный цвет представляет из себя две области в пространстве HSV
    lower_red = np.array([0, 85, 110], dtype="uint8")
    upper_red = np.array([15, 255, 255], dtype="uint8")

    # красный в диапазоне фиолетового оттенка
    lower_violet = np.array([165, 85, 110], dtype="uint8")
    upper_violet = np.array([180, 255, 255], dtype="uint8")

    red_mask_orange = cv2.inRange(frame, lower_red, upper_red)  # применяем маску по цвету
    red_mask_violet = cv2.inRange(frame, lower_violet, upper_violet)  # для красного таких 2

    red_mask_full = red_mask_orange + red_mask_violet  # полная масква предствавляет из себя сумм

    return red_mask_full


def green_mask(frame):
    # с зеленым все проще - он в центре диапазона
    lower_green = np.array([40, 85, 110], dtype="uint8")
    upper_green = np.array([91, 255, 255], dtype="uint8")

    # применяем маску
    green_mask = cv2.inRange(frame, lower_green, upper_green)

    return green_mask


def detect_rect(frame):

    _, contours0, hierarchy = cv2.findContours(green_mask(frame).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    _, contours1, hierarchy = cv2.findContours(red_mask(frame).copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # перебираем все найденные контуры в цикле
    for cnt in contours0:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник

    for cnt in contours1:
        rect = cv2.minAreaRect(cnt)  # пытаемся вписать прямоугольник
        box = cv2.boxPoints(rect)  # поиск четырех вершин прямоугольника
        box = np.int0(box)  # округление координат
        cv2.drawContours(frame, [box], 0, (255, 0, 0), 2)  # рисуем прямоугольник

    cv2.imshow("Test", frame)


def detect_ellipse(frame, red, green):
    _, contours, hierarchy = cv2.findContours(red.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # отображаем контуры поверх изображения
    cv2.drawContours(frame, contours, -1, (255, 0, 255), 3, cv2.LINE_AA, hierarchy, 1)

    cv2.imshow('contours', frame)


while True:
    success, frame = cap.read()

    if frame is None:
        print("Неполадки с видео потоком или видеоролик закончился")
        break

    frame = cv2.resize(frame, (300, 300))

    # рассчитываем зеленые и красные маски для текущего кадра
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask_frame = red_mask(frame_hsv)
    green_mask_frame = green_mask(frame_hsv)

    # здесь мы делаем отсечку для изображений, так как в процессе вычитания мы получаем следующие цифры
    # 255-0=0    0-0=0    0-255=1 мы боремся с последним случаем
    _, red_dif_frame = cv2.threshold(red_mask_frame, 127, 255, cv2.THRESH_BINARY)
    _, green_dif_frame = cv2.threshold(green_mask_frame, 127, 255, cv2.THRESH_BINARY)

    result = cv2.bitwise_and(frame_hsv, frame_hsv, mask=red_dif_frame + green_dif_frame)
    result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)

    _, contours, hierarchy = cv2.findContours(red_dif_frame.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        s = np.pi * np.power(hierarchy[0], 2, dtype="float32")
        for test in s:
            print(test)
            cv2.drawContours(result, contours, -1, (255, 0, 255), 3, cv2.LINE_AA, hierarchy, 1)

    cv2.imshow("DetectionTraffic", result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
