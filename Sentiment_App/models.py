from django.db import models


class Review(models.Model):
    """
    Stores each product review with its sentiment score.
    """
    product_id = models.CharField(max_length=100, blank=True, null=True)
    product_name = models.CharField(max_length=255, blank=True, null=True)
    review_text = models.TextField()
    rating = models.FloatField(blank=True, null=True)

    sentiment = models.CharField(
        max_length=10,
        choices=[
            ("positive", "Positive"),
            ("negative", "Negative"),
            ("neutral", "Neutral"),
        ],
        default="neutral"
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product_name} — {self.sentiment}"


class AspectSentiment(models.Model):
    """
    Stores positive, negative, neutral counts for each aspect.
    Example aspects: 'price', 'quality', 'delivery', 'packaging'
    """
    aspect = models.CharField(max_length=100, unique=True)
    positive = models.IntegerField(default=0)
    negative = models.IntegerField(default=0)
    neutral = models.IntegerField(default=0)

    def __str__(self):
        return self.aspect
