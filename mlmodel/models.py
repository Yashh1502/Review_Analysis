from django.db import models

# Create your models here.

class Review(models.Model):
    item=models.TextField()
    review_text = models.TextField()
    rating = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f'Review {self.id}'

