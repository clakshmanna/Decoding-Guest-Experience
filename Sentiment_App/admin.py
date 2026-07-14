from django.contrib import admin
from .models import Review, AspectSentiment


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin):
    list_display = ("product_name", "sentiment", "rating", "created_at")
    search_fields = ("product_name", "review_text", "sentiment")
    list_filter = ("sentiment", "product_name")
    ordering = ("-created_at",)


@admin.register(AspectSentiment)
class AspectSentimentAdmin(admin.ModelAdmin):
    list_display = ("aspect", "positive", "negative", "neutral")
    search_fields = ("aspect",)
