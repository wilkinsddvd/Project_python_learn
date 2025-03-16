from django.db import models

# Create your models here.
class Questions(models.Model):
    desc=models.CharField(max_length=20)
    desc_time=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.desc

class Choices(models.Model):
    desc=models.CharField(max_length=10)
    votes=models.IntegerField(default=0)
    question=models.ForeignKey(Questions,on_delete=models.CASCADE)
    def __str__(self):
        return self.desc