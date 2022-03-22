import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

data=pd.read_excel("C:\\Users\\Yuyeon PARK\\Desktop\\21삼성카드공모전\\데이터\\공모전_제공_데이터(1차).xlsx", sheet_name=0, engine="openpyxl")
data.rename(columns = {'최종분류(우선순위 가장 높은것 선택)' : '최종분류'}, inplace = True)
numbering=pd.read_excel("C:\\Users\\Yuyeon PARK\\Desktop\\21삼성카드공모전\\데이터\\공모전_제공_데이터(1차).xlsx", sheet_name=1, engine="openpyxl")
numbering.rename(columns = {'의도명(유형)' : '최종분류'}, inplace = True)
dat=pd.merge(data, numbering, how='inner', on='최종분류')

dat = dat.drop(dat.columns[[5,6,7,8]], axis=1) 
dat

# 우선순위 44개가 각각 차지하는 비율 bar plot
import seaborn as sns
fig, axe = plt.subplots(ncols=1)
fig.set_size_inches(12,5)
sns.countplot(dat['우선순위'])

# 띄어쓰기 수정
from pykospacing import Spacing
spacing = Spacing()
    
rev1=[]
for i in range(len(dat)):
    new_sent=dat['발화'][i]
    kospacing_sent = spacing(new_sent) 
    rev1.append(kospacing_sent)

dat['rev1'] = rev1

# 정규 표현식 함수 정의: 특수문자 제거

import re

def apply_regular_expression(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 한글 추출 규칙: 띄어 쓰기(1 개)를 포함한 한글
    result = hangul.sub('', text)  # 위에 설정한 "hangul"규칙을 "text"에 적용(.sub)시킴
    return result

rev2=[]
for i in range(len(dat)):
    rev2.append(apply_regular_expression(dat['rev1'][i]))

dat['rev2'] = rev2 # 정규표현식을 이용한 형태소 리스트.

# 단어의 빈도를 보고 불용어를 추가 해야겠다.
from konlpy.tag import Okt
from collections import Counter

okt = Okt() # 명사 형태소 추출 함수
aa=[]
for i in range(len(dat)):
    nouns = okt.nouns(apply_regular_expression(dat['rev2'][i]))
    aa.append(nouns)

aa

# 말뭉치 생성
corpus = "".join(dat['rev2'].tolist())
corpus

# 정규 표현식 적용
apply_regular_expression(corpus)

# 전체 말뭉치(corpus)에서 명사 형태소 추출
nouns = okt.nouns(apply_regular_expression(corpus))
print(nouns)

# 빈도 탐색
counter = Counter(nouns)

counter.most_common(20)

# 한 글자 명사 제거
available_counter = Counter({x: counter[x] for x in counter if len(x) > 1})
available_counter.most_common(20)

# 불용어
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

stop_words = pd.read_csv("C:\\Users\\Yuyeon PARK\\Desktop\\21삼성카드공모전\\불용어.txt",header=None)
add_stop_words=['상담','연결','통화','시간','카드','화면','직원','고객','상담사','전화','대기','회비','설명','안내','응대','처리','답변']
stop_words = stop_words.append(add_stop_words)

stop_words

#시간, 응대 라는 단어 뒤에 긍정이 올지 부정이 올지를 계산해보자.
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

# 형태소 분석기 OKT를 사용한 토큰화 작업
okt = Okt()
tokenized_data = []
for sentence in dat['rev2']:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words] # 불용어 제거
    tokenized_data.append(temp_X)

from gensim.models import Word2Vec
model = Word2Vec(sentences = tokenized_data, window = 5, min_count = 5, workers = 4, sg = 0)

model.wv.vectors.shape

print(model.wv.most_similar("응대"))

print(model.wv.most_similar("시간"))

# 응대, 시간 넣어도 될 듯.
# 다시 올라가서 불용어 추가하기.

# 불용어 빼서 다시 데이터 정리하기.

from sklearn.feature_extraction.text import CountVectorizer

def text_cleaning(text):
    hangul = re.compile('[^ ㄱ-ㅣ 가-힣]')  # 정규 표현식 처리
    result = hangul.sub('', text)
    okt = Okt()  # 형태소 추출
    nouns = okt.nouns(result)
    nouns = [x for x in nouns if len(x) > 1]  # 한글자 키워드 제거
    nouns = [x for x in nouns if x not in stop_words]  # 불용어 제거
    return nouns

vect = CountVectorizer(tokenizer = lambda x: text_cleaning(x))
bow_vect = vect.fit_transform(dat['rev2'].tolist())
word_list = vect.get_feature_names()
count_list = bow_vect.toarray().sum(axis=0)

word_list[:5]

# 각 단어가 전체 리뷰중에 등장한 총 횟수
count_list

# 각 단어의 리뷰별 등장 횟수
bow_vect.toarray()

bow_vect.shape

# "단어" - "총 등장 횟수" Matching
word_count_dict = dict(zip(word_list, count_list))
word_count_dict

# 단어 size, 희귀단어 파악.
threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

# TF-IDF 변환
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_vectorizer = TfidfTransformer()
tf_idf_vect = tfidf_vectorizer.fit_transform(bow_vect)

print(tf_idf_vect.shape)

# 벡터와 단어 매핑
vect.vocabulary_

invert_index_vectorizer = {v: k for k, v in vect.vocabulary_.items()}
print(str(invert_index_vectorizer)[:100]+'...') # 아니근데, 불용어 사전에 추가한거 왜 안빠짐,,?

# categorization 될까,,?
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from sklearn.feature_extraction.text import TfidfVectorizer

## softmax regression을 써보자.

print("우선순위:", dat["우선순위"].unique(), sep="\n")

from sklearn.model_selection import train_test_split
data_X = dat['rev2'].values # X 데이터에 해당됩니다. X는 총 1개입니다.
data_y = dat['우선순위'].values # Y 데이터에 해당됩니다. 예측해야하는 값입니다.

(X_train, X_test, y_train, y_test) = train_test_split(data_X, data_y, train_size=0.8, random_state=1)
# 훈련 데이터와 테스트 데이터를 8:2로 나눕니다. 또한 데이터의 순서를 섞습니다.
from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# 훈련 데이터와 테스트 데이터에 대해서 원-핫 인코딩
print(y_train[:5])
print(y_test[:5])

# or 다음 두가지 방법.
## 다시 빌드업. 근데 이거 말고 간단한 코드 있던 걸로 기억하는데 어디갔는지 모르겠네
X_train1, X_test1, y_train1, y_test1 = train_test_split(dat['rev2'], dat['우선순위'], test_size= 0.2, random_state=1234)
# train set tokenization
X_train11 = []
for sentence in X_train1:
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stop_words] # 불용어 제거
    X_train11.append(temp_X)

# test set tokenization
X_test11 = []
for sentence in X_test1:
    temp_X1 = okt.morphs(sentence, stem=True) # 토큰화
    temp_X1 = [word for word in temp_X1 if not word in stop_words] # 불용어 제거
    X_test11.append(temp_X1)

# or 정수 인코딩
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train11)

threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)

# 전체 단어 개수 중 빈도수 2이하인 단어는 제거.
# 0번 패딩 토큰을 고려하여 + 1
vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

tokenizer = Tokenizer(vocab_size) 
tokenizer.fit_on_texts(X_train11)
X_train2 = tokenizer.texts_to_sequences(X_train11)
X_test2 = tokenizer.texts_to_sequences(X_test11)

# or 패딩
print('리뷰의 최대 길이 :',max(len(l) for l in X_train2))
print('리뷰의 평균 길이 :',sum(map(len, X_train2))/len(X_train2))
plt.hist([len(s) for s in X_train2], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

def below_threshold_len(max_len, nested_list):
    cnt = 0
    for s in nested_list:
        if(len(s) <= max_len):
            cnt = cnt + 1
    print('전체 샘플 중 길이가 %s 이하인 샘플의 비율: %s'%(max_len, (cnt / len(nested_list))*100))

max_len =25
below_threshold_len(max_len, X_train2)

X_train3 = pad_sequences(X_train2, maxlen = max_len)
X_test3 = pad_sequences(X_test2, maxlen = max_len)

# 위에까지 대충 전처리. 소프트맥스 회귀 돌려보기.
from tensorflow.keras.models import Sequential # 케라스의 Sequential()을 임포트
from tensorflow.keras.layers import Dense # 케라스의 Dense()를 임포트
from tensorflow.keras import optimizers # 케라스의 옵티마이저를 임포트

model=Sequential()
model.add(Dense(3, input_dim=4, activation='softmax'))
sgd=optimizers.SGD(lr=0.01)

# 학습률(learning rate, lr)은 0.01로 합니다.
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
# 옵티마이저는 경사하강법의 일종인 adam을 사용합니다.
# 손실 함수(Loss function)는 크로스 엔트로피 함수를 사용합니다.

history=model.fit(X_train3,y_train1, batch_size=64, epochs=200, validation_data=(X_test3, y_test1))
# 주어진 X와 y데이터에 대해서 오차를 최소화하는 작업을 200번 시도합니다. <-근데 여기서부터 안돌아감..ㅎ..

# 감성어 매칭,,이거 git hub에서 가져오긴 했는데 어째 느낌이 KOSAC 아니면 lexicon 한국어 버전인듯,,? 기억이 잘 안나는 것이 함정.
dic = pd.read_csv("C:\\Users\\Yuyeon PARK\\Desktop\\21삼성카드공모전\\lexicon\\polarity.csv", engine="python")
dic.head(5)

