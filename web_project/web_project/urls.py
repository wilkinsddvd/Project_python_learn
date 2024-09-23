"""web_project URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path
# from django.conf.urls import url, include     django.conf.urls的url已经被淘汰   参考https://blog.csdn.net/shizheng_Li/article/details/124568350
from django.urls import re_path as url
from django.conf.urls import include
from django.shortcuts import HttpResponse
from website_site_management_system import views

#   http://127.0.0.1:8080 ——>   欢迎

def index(request):
    return HttpResponse("欢迎来到Website-management-system")

urlpatterns = [
    path('admin/', admin.site.urls),
    # path("S", index)    # http://127.0.0.1:8000/S
    # http://127.0.0.1:8000/book?id=1
    path('book', views.book_detail_query_string),

    # 在book_id前指定参数类型 优势：
    # 1.以后在浏览器中，如果book_id输入的是一个非整型，那么会出现404错误: /book/abc
    # 2.在视图函数中，得到的book_id就是一个整形，否则，默认是str类型

    path('book/<int:book_id>', views.book_detail_path)

    # url(r'^admin/', admin.site.urls),
    # # include 作用：在 django 匹配 url 时候匹配完 blog/ 后，再次匹配下层地址，所以在 blog/
    # # 后面不可以添加 "$" 符号，不然会导致不能匹配到地址，namespace 为了区分不同应用下同名的模版
    # url(r'^blog/', include('blog.urls', namespace="blog")),
    # 新增内容
]
