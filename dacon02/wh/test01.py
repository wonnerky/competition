import pandas as pd

y_data = pd.read_csv('data/train_answer.csv', index_col=0)
y_labels = y_data.columns.values
y_data = y_data.values

# Feature, Label Shape을 확인합니다.
print(y_data.shape)