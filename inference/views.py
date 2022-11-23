from django.shortcuts import render
from django.views import View
import cv2, numpy as np
from .services.inference_service.inference_engine import Inference
from PIL import Image
# Create your views here.
# def home_page(request):
#     return render(request, 'inference/index.html')


class InferenceView(View):
    def get(self, request):

        return render(request, 'inference/index-2.html')

    def post(self, request):
        # print(request)
        # import code
        # code.interact(local=dict(globals(), **locals()))
        img = cv2.imdecode(np.fromstring(request.FILES['imagefile'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        predict = Inference()
        if predict.image_classification(img):
            messege = 'Crop is Healthy'
        else:
            messege = 'Crop is Diseased'
        messege = {'predict_msg': messege,
                   'success': True}
        return render(request, 'inference/index-2.html',messege)
