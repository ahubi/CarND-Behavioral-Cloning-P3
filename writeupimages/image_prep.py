import cv2
center_image = cv2.imread('original2.jpg')
flipped = cv2.flip(center_image, 1)
cv2.imwrite('flipped2.jpg', flipped)

cropped = center_image[70:135, 0:320]
cv2.imshow("cropped", cropped)
cv2.waitKey(0)
cv2.imwrite('cropped2.jpg', cropped)
