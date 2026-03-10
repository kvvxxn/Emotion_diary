**`팀원`**: 권도현, 이준현, 서교빈, 정환

**`시작일`**: 2025-07-04


## 아이디어

> **AI 감정 인식 일기장**
> 🤖 사진/텍스트를 기반으로 감정을 파악하고 기록해주는 웹 사이트 


## 기능

> 📆 **감정 캘린더**	      하루마다 감정 색 점 찍는 달력

> 📈 **감정 추세 그래프**	주/월/년 단위 감정 변화 트렌드 보기


## 모델

**Text Encoder**

- `Model`: BERT (Bidirectional Encoder Representations from Transformers)

**Image Encoder**

- `Model`: ViT (Vision Transformer)

**Fusion Layer**

- `Mechanism`: 두 인코더에서 나온 고차원 특징 벡터를 Late Fusion 방식으로 결합.

**Classification**

- 결합된 벡터를 Fully Connected Layer에 통과시켜 최종 7가지 감정에 대한 확률 값을 도출.



## 어플리케이션

[https://diary-web-qyme.vercel.app/](https://diary-web-qyme.vercel.app/)

## TODO

1. 모델 구조 수정 및 성능 개선
2. 더 좋은 데이터 수집 후 Re-training
