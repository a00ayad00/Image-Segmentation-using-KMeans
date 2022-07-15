import cv2
import os
os.chdir('D:\Projects\Computer Vision\Image Segmentation\GMM')

def show(img, title=''):
    cv2.imshow(title, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

img = cv2.imread('Plant Cell.jpg')
img.shape    # (401, 600, 3)

# show(img)

reshaped_img = img.reshape(-1, 3)


from sklearn.mixture import GaussianMixture as gmm
gmm_model = gmm(n_components=3, covariance_type='tied').fit(reshaped_img)
labels = gmm_model.predict(reshaped_img)


segmented_img = labels.reshape(img.shape[0], img.shape[1]).astype('uint8')

cv2.imwrite('SegmentedCell_GMM.jpg', segmented_img)

