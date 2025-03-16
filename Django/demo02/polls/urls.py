from django.conf.urls import url
from .views import *

app_name="polls"

urlpatterns=[
    url(r'^index/$',index,name="index"),
    url(r'^detail/(\d+)/$',detail,name="detail"),
    url(r'^result/(\d+)/$',result,name="result"),
]