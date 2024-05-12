#!/usr/bin/env python3
import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime
import matplotlib.pyplot as plt


# Создание данных
def generate_record():
    return {
        "id": fake.uuid4(),
        "name": fake.name(),
        "age": random.randint(18, 80),
        "email": fake.email(),
        "created_at": fake.date_time_between(start_date='-2y', end_date='now'),
        "score": np.random.random() * 100
    }


# Проверка на цифру
def contains_digit(value):
    if pd.isna(value):
        return False
    return bool(pd.Series(value).str.contains(r'\d').any())

# Проверка на время (от 1 до 3)
def is_in_range(time):
    return time.time() < datetime.strptime('03:00', '%H:%M').time() and time.time() >= datetime.strptime('01:00', '%H:%M').time()


# Сохранение данных в csv
def save_to_csv():
    # Создание DataFrame
    data = [generate_record() for _ in range(num_records)]

    # Преобразование в DataFrame
    df = pd.DataFrame(data)

    # Добавление 10% дубликатов
    duplicates = df.sample(frac=0.1, random_state=42)
    df = pd.concat([df, duplicates])

    # Пустые строки в начале
    blank_count = 10
    blank_df = pd.concat([pd.DataFrame({c: [np.nan] for c in df})] * blank_count, ignore_index=True)
    df = pd.concat([df, blank_df])

    # Добавление текстовых строк
    text_count = 10
    text_df = pd.concat([pd.DataFrame({c: [fake.name()] for c in df})] * text_count, ignore_index=True)
    df = pd.concat([df, text_df])

    # Сохранение в CSV
    df.to_csv("dataset.csv", index=False)


# Расчет метрик
def calc_metrics(df):
    # Рассчитать метрики
    agg_metrics = df.groupby('hour').agg({
    'name': pd.Series.nunique,  # Кол-во уникальных строк
    'score': ['mean', 'median']  # Среднее и медиана
    })
    agg_metrics.columns = ['unique_names', 'mean_amount', 'median_amount']
    return agg_metrics


# Рисуем гистограмму по метрикам
def histogram_metric(df):
    # Построение гистограммы для числовой колонки 'score'
    plt.hist(df['score'], bins=50, color='blue', alpha=0.7)
    plt.title('Гистограмма распределения баллов')
    plt.xlabel('Баллы')
    plt.ylabel('Частота')
    plt.grid(True)
    plt.show()

# Рисуем график среднего значения numeric колонки (y) по месяцам (x)
def drow_graph(df):
    # Группировка данных по месяцам и расчет средних значений
    df['month'] = df['created_at'].dt.to_period('M')
    monthly_avg = df.groupby('month')['score'].mean().reset_index()
    monthly_avg['month'] = monthly_avg['month'].dt.to_timestamp()  # Конвертация обратно в timestamp для визуализации

    # Визуализация
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_avg['month'], monthly_avg['score'], marker='o')
    plt.title('Среднее значение баллов по месяцам')
    plt.xlabel('Месяц')
    plt.ylabel('Среднее значение баллов')
    plt.grid(True)
    plt.show()


def read_data_and_process():
    # Чтение данных
    df_from_csv = pd.read_csv('dataset.csv', delimiter=',')
    # Если нет цифр, то очищаем
    mask = df_from_csv.applymap(contains_digit).any(axis=1)
    df_cleaned = df_from_csv[mask]
    # Убираем none
    df_without_na= df_cleaned.dropna(how='all')
    # Убираем дубли
    df_unique = df_without_na.drop_duplicates()
    # Меняет на формат datetime
    df_unique['created_at'] = pd.to_datetime(df_unique['created_at'])
    # Применим фильтр к DataFrame
    filtered_df = df_unique[~df_unique['created_at'].apply(is_in_range)]
    # Группировать по часу
    filtered_df['hour'] = filtered_df['created_at'].dt.floor('H')
    # Меняем формат на numeric
    filtered_df['score'] = pd.to_numeric(filtered_df['score'])

    # Метрики
    df_metrics = calc_metrics(filtered_df)
    # Использование merge_asof для слияния данных
    result = pd.merge_asof(filtered_df.sort_values('hour'), df_metrics.sort_values('hour'), on='hour', direction='nearest')
    histogram_metric(result)
    drow_graph(result)


if __name__ == "__main__":
     # Инициализация генератора данных
    fake = Faker()
    Faker.seed(42)
    np.random.seed(42)
    # Количество записей
    num_records = 100_000

    save_to_csv()
    read_data_and_process()
