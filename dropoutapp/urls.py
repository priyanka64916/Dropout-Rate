from django.urls import path
from . import views

urlpatterns = [
    path('', views.forecast_dropout, name='forecast_dropout'),
]
