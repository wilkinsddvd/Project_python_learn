from django.contrib import admin
from .models import BookInfo,heroInfo
# Register your models here.

class heroInfoInline(admin.StackedInline):
    model = BookInfo
    extra = 1

class BookInfoAdmin(admin.ModelAdmin):
    # list_display = ("title","pub_time")
    inlines = []

# class heroInfoAdmin(admin.ModelAdmin):
    # list_display =

admin.site.register(BookInfo,BookInfoAdmin)
admin.site.register(heroInfo)