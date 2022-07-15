import cv2
import os
os.chdir('D:\Projects\Computer Vision\Image Segmentation')

def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('Deer.jpg')
img.shape    # (401, 600, 3)

# show(img)

reshaped_img = img.reshape(-1, 3).astype('float32')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

K=4
inertia, labels, centers = cv2.kmeans(reshaped_img, K, None, criteria, 10,
                                cv2.KMEANS_PP_CENTERS)


centers = centers.astype('uint8')

res = centers[labels.flatten()]
res2 = res.reshape(img.shape)

show(res2)

cv2.imwrite('SegmentedDeer.jpg', res2)











