import os
import pandas as pd
import numpy as np
import scipy
from tqdm import tqdm
from glob import glob
from scipy.io import wavfile

# wav 파일로부터 데이터를 불러오는 함수, 파일 경로를 리스트 형태로 입력
def data_loader(files):
    out = []
    for file in tqdm(files):
        fs, data = wavfile.read(file)
        out.append(data)
    out = np.array(out)
    return out


# Wav 파일로부터 Feature를 만듭니다.
x_data = glob('data/train/*.wav')
x_data = data_loader(x_data)
x_data = x_data[:, ::8] # 매 8번째 데이터만 사용
x_data = x_data / 30000 # 최대값 30,000 을 나누어 데이터 정규화
x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], 1)

# 정답 값을 불러옵니다
y_data = pd.read_csv('data/train_answer.csv', index_col=0)
y_data = y_data.values

# Feature, Label Shape을 확인합니다.
print(x_data.shape)
print(y_data.shape)

automl  = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)


