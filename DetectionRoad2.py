import cv2
import numpy as np

# Настройки
img_size = [400, 200]  # Размер изображения weight, height
area_setting = np.float32([
    [40, 180],  # x1 + y1 - Нижняя левая точка
    [390, 180],  # x2 + y2 - Нижняя правая точка
    [280, 100],  # x3 + y3 - Верхняя правая точка
    [120, 100]    # x4 + y4 - Верхняя левая точка
])  # координаты для рисования прямоугольника
cords = np.float32([
    [0, img_size[1]],
    [img_size[0], img_size[1]],
    [img_size[0], 1],
    [0, 0]
])
draw_area = np.array(area_setting, dtype=np.int32)  # Рисования поля
nwindow = 12  # Кол-во окон для распознования разметки

video = cv2.VideoCapture("video/DriveOnRoad.mp4")  # Чтение видео


# Бинаризация изображения
# Приводим темные участки к 0, а светлые к 1
def binary(img, value):
    channel = img[:, :, 2]
    bin = np.zeros_like(channel)
    bin[(channel > value)] = 255

    return bin


while video.isOpened():
    _, frame = video.read()  # Чтение видео
    frame = cv2.resize(frame, (img_size[0], img_size[1]))  # Изминения размера, для быстроты алгоритма
    frame_hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)  # Перевод в формат hsl

    # Бинарезуем изображение
    binary_frame = binary(frame, 200)
    binary_hls = binary(frame_hls, 160)

    # Объединение двух бинаризуемых изображений
    all_binary = np.zeros_like(binary_frame)
    all_binary[((binary_frame == 255) | (binary_hls == 255))] = 255

    # Рисование прямоугольника
    all_binary_visual = all_binary.copy()
    cv2.polylines(all_binary_visual, [draw_area], True, 255)
    M = cv2.getPerspectiveTransform(area_setting, cords)
    warped = cv2.warpPerspective(all_binary, M, (img_size[0], img_size[1]), flags=cv2.INTER_LINEAR)

    # Поиск белых пикселей
    warped_visual = warped.copy()
    histogram = np.sum(warped[warped.shape[0]//2:, :], axis=0)
    midpoint = histogram.shape[0]//2  # нахождение центральной точки
    left_column = np.argmax(histogram[:midpoint])  # нахождение левой колонки
    right_column = np.argmax(histogram[midpoint:]) + midpoint  # нахождение правой колонки
    # Отображение линий
    cv2.line(warped_visual, (left_column, 0), (left_column, warped_visual.shape[0]), 110, 2)
    cv2.line(warped_visual, (right_column, 0), (right_column, warped_visual.shape[0]), 110, 2)

    # Поиск беблых линий в окне
    nonzero = warped.nonzero()
    WhitePixelIndY = np.array(nonzero[0])
    WhitePixelIndX = np.array(nonzero[1])

    # Рисование окон на линиях
    window_height = np.int(warped.shape[0]/nwindow)
    window_half_width = 25

    XCenterLeftWindow = left_column
    XCenterRightWindow = right_column

    left_line_inds = np.array([], dtype=np.int16)
    right_line_inds = np.array([], dtype=np.int16)

    out_img = np.dstack((warped, warped, warped))

    for window in range(nwindow):
        win_y1 = warped.shape[0] - (window + 1) * window_height
        win_y2 = warped.shape[0] - window * window_height

        left_win_x1 = XCenterLeftWindow - window_half_width
        left_win_x2 = XCenterLeftWindow + window_half_width
        right_win_x1 = XCenterRightWindow - window_half_width
        right_win_x2 = XCenterRightWindow + window_half_width

        cv2.rectangle(out_img, (left_win_x1, win_y1), (left_win_x2, win_y2), (50 + window * 21, 0, 0), 2)
        cv2.rectangle(out_img, (right_win_x1, win_y1), (right_win_x2, win_y2), (0, 0, 50 + window * 21), 2)

        # Поиск линий в окне
        good_left_inds = ((WhitePixelIndY>=win_y1) & (WhitePixelIndY<=win_y2) & (WhitePixelIndX>=left_win_x1) & (WhitePixelIndX<=left_win_x2)).nonzero()[0]
        good_right_inds = ((WhitePixelIndY >= win_y1) & (WhitePixelIndY <= win_y2) & (WhitePixelIndX >= right_win_x1) & (
                    WhitePixelIndX <= right_win_x2)).nonzero()[0]

        left_line_inds = np.concatenate((left_line_inds, good_left_inds))
        right_line_inds = np.concatenate((right_line_inds, good_right_inds))

        # Смещение координат линии
        if len(good_left_inds > 50):
            XCenterLeftWindow = np.int(np.mean(WhitePixelIndX[good_left_inds]))
        if len(good_right_inds > 50):
            XCenterRightWindow = np.int(np.mean(WhitePixelIndX[good_right_inds]))

    # search lines in windows
    out_img[(WhitePixelIndY[left_line_inds], WhitePixelIndX[left_line_inds])] = [255, 0, 0]
    out_img[(WhitePixelIndY[right_line_inds], WhitePixelIndX[right_line_inds])] = [0, 0, 255]

    # Вывод изображений
    cv2.imshow("Main Video", frame)
    cv2.imshow("Binary video", binary_frame)
    cv2.imshow("BinaryHLS video", binary_hls)
    cv2.imshow("AllBinary video", all_binary)
    cv2.imshow("AllBinary area video", all_binary_visual)
    cv2.imshow("Warped img", warped)
    cv2.imshow("Warped_visual img", warped_visual)
    cv2.imshow("Windows", out_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        video.release()
        cv2.destroyAllWindows()