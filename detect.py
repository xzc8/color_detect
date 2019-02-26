import cv2
import numpy as np

element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))


def rgb_detect(matrix):
    mask_matrix = np.zeros(shape=(2000, 2000), dtype=np.uint8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j][1] < matrix[i][j][2]):
                if (matrix[i][j][1] < matrix[i][j][2] * 0.37 + 65.28):
                    if (matrix[i][j][1] < matrix[i][j][2] * 0.95 - 15.92):
                        mask_matrix[i][j] = 255
    mask_matrix = cv2.dilate(mask_matrix, kernel=element, iterations=3)
    return mask_matrix


def hsv_detect(origin_matrix):
    matrix = cv2.cvtColor(origin_matrix, cv2.COLOR_BGR2HSV)
    mask_matrix = np.zeros(shape=(2000, 2000), dtype=np.uint8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j][0] >= 11.43 and matrix[i][j][0] <= 27.56):
                if (matrix[i][j][2] >= 256*(matrix[i][j][0] * 0.0054 + 0.6 )):
                    if(matrix[i][j][2] <= 256*(1.16 - 0.0095 * matrix[i][j][0])):
                        mask_matrix[i][j] = 255
    mask_matrix = cv2.dilate(mask_matrix, kernel=element,iterations=3)
    return mask_matrix


def lab_detect(origin_matrix):
    matrix=cv2.cvtColor(origin_matrix,cv2.COLOR_BGR2LAB)
    mask_matrix = np.zeros(shape=(2000, 2000), dtype=np.uint8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if (matrix[i][j][1] > 135):
                mask_matrix[i][j] = 255
    mask_matrix = cv2.dilate(mask_matrix, kernel=element,iterations=3)
    return mask_matrix


def draw_wilt(origin_matrix, mask_matrix, src,mode):
    canvas = np.copy(origin_matrix)
    img, cnt, hiera = cv2.findContours(mask_matrix, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    for area in cnt:
        if (cv2.contourArea(area) > 200 and cv2.contourArea(area) < 10000):
            counter += 1
            rect = cv2.minAreaRect(area)
            box = cv2.boxPoints(rect)
            box_d = np.int0(box)
            canvas = cv2.drawContours(canvas, [box_d], 0, (238, 48, 167), 2)
    text = mode + str(counter)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(canvas, text, (20, 60), font, 1.5, (48, 255, 255), 2)
    cv2.imwrite(src+'_'+mode+'_result.jpg', canvas)
    print(src,mode,'is done.')
