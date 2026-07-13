from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('records/', views.records, name='records'),
    path('records/export/', views.records_export, name='records_export'),
    path('records/<int:pk>/', views.record_detail, name='record_detail'),
    path('settings/', views.settings_page, name='settings'),
]
