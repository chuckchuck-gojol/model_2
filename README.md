# 모델 1 - 기능 1, 2, 3에 대해 처리 가능한 모델1 작성 (1)

#### ver 3.3

#### 모델 1은 기능 3의 지하철 트리거까지 포함하는 모델임.

#### 모델 2는 모델 1에서 기능 3 지하철 트리거 인식 시 작동되는 모델임

#### 최종 수정일 : 20-05-27

------

## 1. 데이터 전처리

### 1.1 wav 파일 로딩 및 피처 생성

> [변경사항]
>
> - 피처를 최대한 많이 생성하여 데이터를 가공함 -> drop out 효과 극대화 및 과적합 방지, 정확도 향상
> - 이전 기능 별 분리되었던 모델(ver 2.*) 을 통합함
> - 각 데이터 별 193 피처 추출
> - row 통일 안함 (3~4sec)
> - 원핫인코딩 안함
> - 라벨을 1차원 배열로 변경 -> 카테고리 별 int값 출력하게 함

------

> [기존과 동일]
>
> - 기능 1,2에 대해 사용 데이터는 뉴욕대학교 MARL의 URBANSOUND8K DATASET 일부와 일상 생활에서 녹음한 녹음 파일(.wav)를 활용 (2,622개, 1.96G)
> - 기능 3에 대해 환승역 알림음을 트리거로 적용 (894개, 0.71G)

```python
import numpy as np
import pandas as pd
#wav 파일들의 피처 생성
#librosa 사용
#사용 특성은 mfcc, chroma_stft, melspectorgram, spectral_contrast, tonnetz로 총193
#딥러닝 모델만 사용할 예정 -> 피처 축소 생략
import glob
import librosa

# 오디오 불러오기 + 피쳐 생성
# 피쳐 193개
# row 통일 안시킴
def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz
    
#데이터 가공
#행렬로 변환
def parse_audio_files(filenames):
    rows = len(filenames)
    # feature는 각 파일 별 row(window) * 피처 의 2차원 행렬
    # labels은 파일 별 카테고리 int 값
    features, labels = np.zeros((rows,193)), np.zeros((rows, 1))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            y_col = int(fn.split('-')[0])
        except:
            print("error : "+fn)
        else:
            features[i] = ext_features
            labels[i] = y_col
            print(y_col)
            i += 1
    return features, labels

audio_files = []
#0 : 사이렌
#1 : 자동차가 다가오는 소리(엔진소리)
#2 : 자동차 경적소리
#4 : 환승역 안내음
audio_files.extend(glob.glob('*.wav'))

print(len(audio_files))

files = audio_files
X, y= parse_audio_files(files)

#?.npz
np.savez('data', X=X, y=y)
```

***

## 2. 모델링

### 2.1 데이터 구성

> [변경사항]

------

> [기존과 동일]
>
> - data.npz 불러오기

```python
import glob
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 알림, 차량 엔진, 차량 경적, 지하철 트리거 순
sound_data = np.load('model_1.npz')
X_train = sound_data['X']
y_train = sound_data['y']
X_train.shape, y_train.shape

X_train.shape, y_train.shape
```

***

### 2.2 모델 학습

> [변경사항]
>
> - lstm 사용
> - 이전 모델의 경우 파라미터 조정에 초점을 두었으나 ver 3.5에서는 layer 구성에 초점을 두어 진행

------

> [기존과 동일]

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os

from keras import models
from keras import layers
from keras.layers import *
from keras import optimizers

from keras.layers import LSTM
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping

K.clear_session()
model = Sequential() # Sequeatial Model
model.add(LSTM(20, input_shape=(193, 1))) # (timestep, feature)
model.add(Dense(1)) # output = 1
model.compile(loss='mean_squared_error', optimizer='adam')

model.summary()

#X_train = X_train.values
X_train = X_train.reshape(X_train.shape[0], 193, 1)

early_stop = EarlyStopping(monitor='loss', patience=1, verbose=1)

model.fit(X_train, y_train, epochs=100,
          batch_size=30, verbose=1, callbacks=[early_stop])
```

***

### 2.3 모델 저장

> [변경사항]
>
> - pkl, json, pb, tflite로 저장

------

> [기존과 동일]

```python
# 모델 pkl로 저장하기
import joblib
joblib.dump(model, 'model/pkl/model_1.pkl')

# 모델 json으로 저장하기
model_1 = model.to_json()
# model = model_from_json(json_string)

# 모델 h5로 저장하기
from keras.models import load_model
model.save('model/h5/model_1')
model.save('model/h5/model_1.h5')

# 모델 pb로 저장하기
model = keras.models.load_model('model/h5/model_1', compile=False)
model.save('model/pb/',save_format=tf)

#모델 tflite 로 저장하기
saved_model_dir='model/pb/'
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS,
                                     tf.lite.OpsSet.SELECT_TF_OPS]
tfilte_mode=converter.convert()
open('model/tflite/model_1.tflite','wb').write(ftlite_model)
```

***

## 3. 테스트

### 3.1 녹음 파일 생성

> [변경사항]
>
> - 10초로 제한

------

> [기존과 동일]
>
> - 블루투스 이어폰의 외부 마이크 사용 (채널 조정)

```python
import librosa
import scipy.signal as signal
import numpy as np
import pandas as pd
import joblib

#환경 확인
import pyaudio
import wave
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 10
WAVE_OUTPUT_FILENAME = "test_file.wav"
audio = pyaudio.PyAudio()

# start Recording
stream = audio.open(format=pyaudio.paInt16,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    input_device_index=1,
                    frames_per_buffer=CHUNK)
print ("recording...")
frames = []
for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)
print ("finished recording")

stream.stop_stream()
stream.close()
audio.terminate()
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()
```

***

### 3.2 전처리

> [변경사항]
>
> - 모듈화 할 것

------

> [기존과 동일]

```python
import numpy as np
import pandas as pd 
import glob

def extract_feature(file_name):
    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(filenames):
    rows = len(filenames)
    # feature는 각 파일 별 row(window) * 피처 의 2차원 행렬
    # labels은 파일 별 카테고리 int 값
    features = np.zeros((rows,193))
    i = 0
    for fn in filenames:
        try:
            mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
        except:
            print("error : "+fn)
        else:
            features[i] = ext_features
            print("성공")
            i += 1
    return features

audio_files = []
audio_files.extend(glob.glob(WAVE_OUTPUT_FILENAME))

print(len(audio_files))
files = audio_files
X_test = parse_audio_files(files)
```

***

### 3.3 모델 적용

> [변경사항]

------

> [기존과 동일]
>
> - 0 : 사이렌, 민방위 등 알림음
> - 1,2 : 차량 경적, 엔진소리
> - 3 : 환승 트리거

```python
X_test = X_test.reshape(X_test.shape[0], 193, 1)

model = joblib.load('model_2.pkl')

pred = model.predict_proba(X_test)
ans = float(pred)

print(pred)

ans = round(float(pred))
de_label = pd.read_csv('de_train_label.csv', engine='python', index_col = None)
print("이번 역은 "+de_label['name'][ans]+"역 입니다.")
```





