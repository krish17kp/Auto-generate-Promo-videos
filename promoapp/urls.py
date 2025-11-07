from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('generate_promo/', views.generate_promo, name='generate_promo'),
    path('generate/', views.generate, name='generate'),  # 👈 this is the Railway test route
]
