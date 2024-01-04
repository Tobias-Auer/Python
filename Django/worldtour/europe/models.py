from django.db import models


# Create your models here.

class Trip(models.Model):
    origin = models.CharField(max_length=64)
    destination = models.CharField(max_length=64)
    nights = models.IntegerField()
    price = models.IntegerField()
