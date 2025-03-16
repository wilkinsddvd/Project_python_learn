from django.conf.urls import url
from .views import *
app_name="booktest"
urlpatterns=[
    url(r'^index/$',index,name="index"),
    url(r'^list/$',list,name="list"),
    url(r'^detail/(\d+)/$',detail,name="detail"),
    url(r'^deletehero/(\d+)/$',deletehero,name="deletehero"),
    url(r'^deletebook/(\d+)/$',deletebook,name="deletebook"),
    url(r'^addhero/(\d+)/$',addhero,name="addhero"),
    url(r'^addbook$',addbook,name="addbook")
]