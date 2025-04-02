from django.db import models
class Profile(models.Model):
    first_name = models.CharField(max_length=70)
    last_name = models.CharField(max_length=70)
    age = models.IntegerField()
    identi = models.CharField(max_length=200)
    nationality = models.CharField(max_length=200)
    crime = models.CharField(max_length=200)
    gender = models.CharField(max_length=200)
    present = models.BooleanField(default=False)
    image = models.ImageField()
    def __str__(self):
        return self.first_name +' '+self.last_name


class LastFace(models.Model):
    last_face = models.CharField(max_length=200)
    date = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.last_face

