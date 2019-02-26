import detect
import cv2

src='test2.jpg'
origin_matrix=cv2.imread(src)

# RGB detection
rgb_mask_matrix=detect.rgb_detect(origin_matrix)
cv2.imwrite('test2_rgb_mask.jpg',rgb_mask_matrix)
detect.draw_wilt(origin_matrix,rgb_mask_matrix,src[:5],'RGB ')

# HSV detection
hsv_mask_matrix=detect.hsv_detect(origin_matrix)
cv2.imwrite('test2_hsv_mask.jpg',hsv_mask_matrix)
detect.draw_wilt(origin_matrix,hsv_mask_matrix,src[:5],'HSV ')

# LAB detection
lab_mask_matrix=detect.lab_detect(origin_matrix)
cv2.imwrite('test2_lab_mask.jpg',lab_mask_matrix)
detect.draw_wilt(origin_matrix,lab_mask_matrix,src[:5],'LAB ')