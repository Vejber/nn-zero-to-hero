import pandas as pd
from sklearn.model_selection import train_test_split

# Загрузка датасета
data = pd.read_csv('allnames.txt', header=None, names=['Name'])

# Разделение на обучающую и тестовую выборки
train, test = train_test_split(data, test_size=0.2)

# Сохранение выборок в файлы
train.to_csv('train.txt', index=False, header=False)
test.to_csv('test.txt', index=False, header=False)
