from django.shortcuts import render,HttpResponse

# Create your views here.

# 1.通过查询字符串，(query string)： https://www.baidu.com
# 2.在path中携带 http://127.0.0.1:8000//book/2

# 查询字符串: http://127.0.0.1:8000

def book_detail_query_string(request):
    book_id = request.GET.get('id')     # 较为安全
    name = request.GET.get('name')
#   request.GET['id'] #直接报错
    #   http://127.0.0.1:8000/book?id=1
    return HttpResponse(f"您查找的图书id是：{book_id},图书的名称是:{name}")

# http://127.0.0.1:8000/book/1

def book_detail_path(request,book_id):
    return HttpResponse(f"您查找的图书id是:{book_id}")



from website_site_management_system.models import Category

import tag  # 新增尝试

c = Category('test category')
c.save()
