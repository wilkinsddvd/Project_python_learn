from django.shortcuts import render,reverse,redirect
from .models import *
from django.http import HttpResponse
# Create your views here.
def index(request):

    return render(request,"booktest/index.html")

def list(request):
    books = BookInfo.objects.all
    return render(request,"booktest/list.html",{"books":books})


def detail(request,id):
    book=BookInfo.objects.get(pk=id)
    return render(request,"booktest/detail.html",{"book":book})

def deletehero(request,id):
    hero=heroInfo.objects.get(pk=id)
    bookid=hero.book.id
    hero.delete()
    # return render(request, "booktest/detail.html", {"hero": hero})
    # return HttpResponse("删除成功")
    return redirect(reverse("booktest:detail",args=(bookid,)))

def deletebook(request,id):
    book=BookInfo.objects.get(pk=id)
    book.delete()
    # return HttpResponse("删除成功")
    return redirect(reverse("booktest:list"))

def addhero(request,id):
    book = BookInfo.objects.get(pk=id)
    if request.method=="GET":
        return render(request, "booktest/addhero.html", {"book": book})
    elif request.method=="POST":
        name=request.POST.get("name")
        content=request.POST.get("content")
        hero=heroInfo()
        hero.name=name
        hero.content=content
        hero.book=book
        hero.save()
        return redirect(reverse("booktest:detail",args=(id,)))
def addbook(request):
    if request.method=="GET":
        return render(request, "booktest/addbook.html")
    elif request.method=="POST":
        title=request.POST.get("title")
        book=BookInfo()
        book.title=title
        book.save()
        return redirect(reverse("booktest:list"))