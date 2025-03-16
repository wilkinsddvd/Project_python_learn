from django.contrib import admin
from .models import Questions,Choices
# Register your models here.

class QuestionInline(admin.StackedInline):
    model = Choices
    extra = 1


class QuestionsAdmin(admin.ModelAdmin):
    inlines = []
    list_display = ("desc","desc_time")


class ChoicesAdmin(admin.ModelAdmin):
    list_display = ("desc","votes","question")


admin.site.register(Questions)
admin.site.register(Choices)