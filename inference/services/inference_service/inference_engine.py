# from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import cv2
import mahotas
import pickle


# model_path_name = 'model/disease-classification.pkl'


# def get_model():
#     model_file = open(model_path_name, "rb")
#     model = pickle.load(model_file)
#     model_file.close()
#     return model
#
#
# def rgb_to_bgr(image):
#     rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     return rgb_img
#
#
# def bgr_to_hsv(rgb_img):
#     hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
#     return hsv_img
#
#
# def img_segmentation(rgb_img, hsv_img):
#     lower_green = np.array([25, 0, 20])
#     upper_green = np.array([100, 255, 255])
#     healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
#     result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)
#
#     lower_brown = np.array([10, 0, 10])
#     upper_brown = np.array([30, 255, 255])
#     disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
#     disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)
#
#     final_mask = healthy_mask + disease_mask
#     final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
#     return final_result
#
#
# def fd_hu_moments(image):
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     feature = cv2.HuMoments(cv2.moments(image)).flatten()
#     return feature
#
#
# def fd_haralick(image):
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     haralick = mahotas.features.haralick(gray).mean(axis=0)
#     return haralick
#
#
# def fd_histogram(image, mask=None):
#     bins = 8
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#     hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
#     cv2.normalize(hist, hist)
#     return hist.flatten()
#
#
# def process_image(image):
#     # Convert RGB to BGR
#     rgb_bgr = rgb_to_bgr(image)
#
#     # Convert BGR to HSV
#     bgr_hsv = bgr_to_hsv(rgb_bgr)
#
#     # Image Segmentation
#     img_segment = img_segmentation(rgb_bgr, bgr_hsv)
#
#     # Call for Global Feature Descriptors
#     fv_hu_moments = fd_hu_moments(img_segment)
#     fv_haralick = fd_haralick(img_segment)
#     fv_histogram = fd_histogram(img_segment)
#
#     # Concatenate
#     processed_image = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])
#
#     return processed_image
#
#
# def image_classification(image):
#     # Get the Model
#     img_model = get_model()
#
#     # Process the image
#     processed_img = process_image(image)
#
#     # Reshape to 2D
#     processed_img = processed_img.reshape(1, -1)
#
#     # Predict
#     y_predict = img_model.predict(processed_img)
#
#     del img_model
#
#     # 0 - Health / 1 - Disease
#     return y_predict[0]
#
#
# #############################################
# # TEST
# #############################################
# if __name__ == '__main__':
#     # Open Image and Convert to Bytes
#     fixed_size = tuple((500, 500))
#     dirs = ['diseased', 'healthy']
#     for path in dirs:
#         print('##########################################')
#         print(path)
#         print('##########################################')
#         for n in range(1, 10):
#             # get the image file name
#             test_image = 'test/' + path + "/" + str(n) + ".jpg"
#             print(test_image)
#             with open(test_image, 'rb') as fp:
#                 img_bytes = fp.read()
#                 img_buffer = np.frombuffer(img_bytes, np.uint8)
#                 img_np = cv2.imdecode(img_buffer, cv2.IMREAD_COLOR)
#                 img_np = cv2.resize(img_np, fixed_size)
#                 healthy = image_classification(img_np)
#                 print(healthy)


class Inference:

    def __init__(self, model_path_name='inference/services/inference_service/model/disease-classification.pkl'):
        self.model_path_name = model_path_name

    def image_classification(self, image):
        # Get the Model
        img_model = self.get_model()

        # Process the image
        processed_img = self.process_image(image)

        # Reshape to 2D
        processed_img = processed_img.reshape(1, -1)

        # Predict
        y_predict = img_model.predict(processed_img)

        del img_model

        # 0 - Health / 1 - Disease
        return y_predict[0]

    def process_image(self,image):
        # Convert RGB to BGR

        rgb_bgr = self.rgb_to_bgr(image)

        # Convert BGR to HSV
        bgr_hsv = self.bgr_to_hsv(rgb_bgr)

        # Image Segmentation
        img_segment = self.img_segmentation(rgb_bgr, bgr_hsv)

        # Call for Global Feature Descriptors
        fv_hu_moments = self.fd_hu_moments(img_segment)
        fv_haralick = self.fd_haralick(img_segment)
        fv_histogram = self.fd_histogram(img_segment)

        # Concatenate
        processed_image = np.hstack([fv_histogram, fv_haralick, fv_hu_moments])

        return processed_image

    def get_model(self):
        model_file = open(self.model_path_name, "rb")
        model = pickle.load(model_file)
        model_file.close()
        return model

    def rgb_to_bgr(self, image):
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return rgb_img

    def bgr_to_hsv(self, rgb_img):
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
        return hsv_img

    def img_segmentation(self, rgb_img, hsv_img):
        lower_green = np.array([25, 0, 20])
        upper_green = np.array([100, 255, 255])
        healthy_mask = cv2.inRange(hsv_img, lower_green, upper_green)
        result = cv2.bitwise_and(rgb_img, rgb_img, mask=healthy_mask)

        lower_brown = np.array([10, 0, 10])
        upper_brown = np.array([30, 255, 255])
        disease_mask = cv2.inRange(hsv_img, lower_brown, upper_brown)
        disease_result = cv2.bitwise_and(rgb_img, rgb_img, mask=disease_mask)

        final_mask = healthy_mask + disease_mask
        final_result = cv2.bitwise_and(rgb_img, rgb_img, mask=final_mask)
        return final_result

    def fd_hu_moments(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        feature = cv2.HuMoments(cv2.moments(image)).flatten()
        return feature

    def fd_haralick(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        return haralick

    def fd_histogram(self, image, mask=None):
        bins = 8
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
