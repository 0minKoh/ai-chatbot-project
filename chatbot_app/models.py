# ai_chatbot_project/chatbot_app/models.py
from django.db import models

class FAQCategory(models.Model):
    name = models.CharField(max_length=100, unique=True)
    description = models.TextField(blank=True, null=True)

    class Meta:
        verbose_name_plural = "FAQ Categories" # 관리자 페이지에서 보여지는 이름 설정

    def __str__(self):
        return self.name

class FAQ(models.Model):
    category = models.ForeignKey(FAQCategory, on_delete=models.CASCADE, related_name='faqs')
    question = models.TextField()
    answer = models.TextField()
    # 임베딩 벡터를 저장할 필드 (실제 DB에는 저장하지 않고, FAISS에만 저장할 것이지만,
    # 나중에 FAQ 수정 시 임베딩을 다시 생성하기 위한 참조용으로 빈 필드를 둘 수도 있습니다.
    # 여기서는 FAISS에만 저장하는 것으로 가정하고 모델에는 추가하지 않습니다.)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "FAQs" # 관리자 페이지에서 보여지는 이름 설정
        ordering = ['category', 'created_at']

    def __str__(self):
        return f"{self.category.name}: {self.question[:50]}..." # 질문의 앞부분만 표시
