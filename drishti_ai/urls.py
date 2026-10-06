from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('image_module.urls')),
    path('video/', include('video_module.urls')),
    path('text/', include('text_module.urls')),
]
