from django.urls import path
from . import views

app_name = 'chatbot_app' # 앱 이름 공간 설정

urlpatterns = [
    path('', views.index, name='index'),
]