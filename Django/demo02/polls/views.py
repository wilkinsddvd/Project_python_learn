from django.shortcuts import render
from .models import Questions,Choices
from django.http import HttpResponse


# Create your views here.

def index(request):
    questions=Questions.objects.all
    return render(request,"polls/index.html",{"questions":questions})


def detail(request,id):
    question = Questions.objects.get(pk=id)
    if request.method=="GET":
        return render(request,"polls/detail.html",{"question":question})
    elif request.method=="POST":
        choiceid=request.POST.get("choice")
        print(choiceid)
        choice=Choices.objects.get(pk=choiceid)
        print(choice)
        choice.votes+=1
        choice.save()
        return render(request,"polls/result.html",{"question":question})
def result(request,id):
    question = Questions.objects.get(pk=id)
    return  render(request,"polls/result.html")