# 🧠 NLP & Machine Learning Toy Projects  

텍스트 마이닝과 자연어 처리(NLP)를 활용한 토이 프로젝트 
논문 데이터 토픽 모델링과 감성 분석(Sentiment Analysis) 실험을 진행함

## 🔍 프로젝트 개요  

### **1. 논문 데이터 토픽 모델링**  
- 9000여 개의 논문 데이터를 분석하여 **최적의 토픽 수를 결정**하는 것이 목표  
- 전처리 과정  
  - 불용어(Stopwords) 제거  
  - **두 가지 토큰화 방식 비교**  
    - RegexpTokenizer → 정규 표현식을 기반으로 토큰화  
    - PorterStemmer → 어근을 추출하며 토큰화  
- **LDA 모델 적용**  
  - Gensim을 사용하여 Dictionary 생성 및 **Bag of Words 모델 구축**  
  - 각 토픽의 **혼란도(Perplexity)** 및 **일관성(Cohesion Score)** 계산  
  - 실험 결과 **PorterStemmer 방법이 가장 좋은 성과**  

📁 관련 파일: topic_modeling.ipynb
---

### **2. 감성 분석 (Sentiment Analysis)**  
- **Twitter Sentiment Analysis 데이터셋**을 사용하여 긍정/부정을 분류  
- **감정 점수 분석 (Afinn 라이브러리 사용)**  
  - 단어별 감정 점수를 매기고, 전체 문장의 감성을 예측  
- **벡터화 후 머신러닝 모델 적용**  
  - TF-IDF, CountVectorizer를 활용하여 벡터화  
  - 모델: Naive Bayes, Logistic Regression, SVM
  - 성능 평가: **Confusion Matrix 활용**  

📁 관련 파일: sentiment_analysis.ipynb

---
### **3. LSTM을 활용한 감성 분석**  
- 감정 분석과 동일한 데이터셋을 활용하여 **Word2Vec 임베딩 + LSTM 모델 적용**  
- 문장의 길이 분포 확인 후, **최적의 문장 길이(20단어)** 설정  
- LSTM 기반 감성 분석 모델을 학습 및 평가  

📁 관련 파일: sentanaly_LSTM.ipynb

---

## 📌 사용된 주요 라이브러리  
- **NLP 전처리**: NLTK, Gensim, RegexpTokenizer, PorterStemmer
- **토픽 모델링**: LDA, Bag of Words 
- **감성 분석**: Afinn, TF-IDF, Word2Vec  
- **모델링**: Naive Bayes, Logistic Regression, SVM, LSTM  
- **시각화 및 분석**: Matplotlib, Seaborn, Scikit-learn  

---
