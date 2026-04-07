from django.contrib import admin
from django.urls import path, include

# ADD THESE
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('image_module.urls')),
    path('video/', include('video_module.urls')),
    path('text/', include('text_module.urls')),
]

# ADD THIS BLOCK
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)