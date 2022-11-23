from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from django.views.generic.base import RedirectView
from inference.views import InferenceView

urlpatterns = [
    path('', InferenceView.as_view(), name='index'),
    path('favicon.ico', RedirectView.as_view(url=staticfiles_storage.url('img/favicon.ico')))
] + static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
