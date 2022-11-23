from django.conf import settings
from django.conf.urls.static import static
from django.urls import path

from inference.views import InferenceView

urlpatterns = [
    path('', InferenceView.as_view(), name='index'),
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)