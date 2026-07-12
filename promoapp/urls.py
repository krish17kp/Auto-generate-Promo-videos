from django.contrib.auth import views as auth_views
from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('generate/', views.generate_video, name='generate_video'),
    path('job/<uuid:job_id>/status/', views.job_status, name='job_status'),
    path('job/<uuid:job_id>/', views.job_result, name='job_result'),
    path('accounts/signup/', views.signup, name='signup'),
    path('accounts/login/', auth_views.LoginView.as_view(template_name='promoapp/login.html'), name='login'),
    path('accounts/logout/', auth_views.LogoutView.as_view(), name='logout'),
    path('history/', views.history, name='history'),
]
