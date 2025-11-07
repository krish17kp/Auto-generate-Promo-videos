from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # your old main page
    path('generate_promo/', views.generate_promo, name='generate_promo'),  # legacy route if used
    path('generate/', views.generate_video, name='generate_video'),  # updated working route
]
