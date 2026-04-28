# ad-click-ctr-prediction

> **Предсказание вероятности клика (CTR) на рекламный баннер с калибровкой модели**

Проект реализует полный ML-пайплайн для задачи бинарной классификации: предсказание вероятности клика пользователя по рекламному объявлению (Click-Through Rate). Особый акцент сделан на **калибровке предсказанных вероятностей** — ключевом требовании для корректной работы рекламного аукциона.

---

## Содержание

- [Бизнес-контекст](#бизнес-контекст)
- [Структура репозитория](#структура-репозитория)
- [Данные](#данные)
- [Методология](#методология)
- [Результаты](#результаты)
- [Установка и запуск](#установка-и-запуск)
- [Артефакты модели](#артефакты-модели)
- [Стек технологий](#стек-технологий)

---

## Бизнес-контекст

Рекламная платформа **Advandex** участвует в аукционах за показ баннеров. Ставка на аукционе напрямую зависит от предсказанного CTR: если модель завышает вероятность клика — платформа переплачивает; если занижает — проигрывает аукционы и теряет доход.

**Цель:** построить модель, у которой предсказанная вероятность 20% означает, что клик произойдёт ровно в 20 случаях из 100 — то есть модель должна быть не только точной, но и **хорошо откалиброванной**.

---

## Структура репозитория

```
ad-click-ctr-prediction/
│
├── Sprint_12_proj - Copy.ipynb   # Основная тетрадка: EDA → обучение → калибровка
├── 12_save_model.py              # Скрипт сохранения и проверки артефактов (раздел 12)
├── 11.md                         # Финальный отчёт и выводы (раздел 11)
├── Project_explanation.md        # Техническое задание проекта
├── requirements.txt              # Зафиксированные версии зависимостей
│
└── artifacts/                    # Создаётся автоматически при запуске 12_save_model.py
    ├── preprocessor.joblib       # Обученный пайплайн предобработки
    ├── calibrated_model.joblib   # Откалиброванная финальная модель
    └── feature_info.joblib       # Метаданные о признаках
```

---

## Данные

Датасет `ds_s16_ad_click_dataset.csv` — аналитическая витрина событий показа рекламных баннеров.

| Группа признаков | Примеры | Тип |
|---|---|---|
| Идентификаторы и время | `id`, `hour` | int |
| Рекламная площадка | `site_id`, `site_domain`, `site_category` | object |
| Рекламируемое приложение | `app_id`, `app_domain`, `app_category` | object |
| Устройство пользователя | `device_id`, `device_ip`, `device_model`, `device_type`, `device_conn_type` | object / int |
| Параметры баннера и аукциона | `banner_pos`, `C1`, `C14`–`C21` | int |
| ML-признаки | `ml_feature_1`–`ml_feature_10` | int / object |
| **Целевая переменная** | **`click`** (0 / 1) | int |

**Дисбаланс классов:** ~83% — нет клика, ~17% — клик (соотношение 4.8:1).

> Данные анонимизированы: категориальные значения представлены в виде хэшей (`50e219e0`, `3e814130` и т.п.).

---

## Методология

### 1. EDA и предобработка
- Анализ распределений, выбросов, корреляций (метрика **phi_k** для смешанных типов данных)
- Удаление признаков без предсказательной силы (`id`) и высококардинальных с переобучением (`device_id`, `device_ip`)
- Пайплайн предобработки через `sklearn.pipeline.Pipeline` + `ColumnTransformer`:
  - Числовые: `SimpleImputer(median)` → `StandardScaler`
  - Бинарные: `SimpleImputer(most_frequent)`
  - Категориальные с кардинальностью ≤ 10: `OneHotEncoder`
  - Категориальные с кардинальностью > 10: `TargetEncoder(cv=5, smooth=30)`

### 2. Отбор признаков
- Фильтрационный метод: phi_k корреляция с таргетом → удаление признаков с нулевой связью
- `VarianceThreshold` для выявления константных/квазиконстантных признаков
- Метод-обёртка: `RFE` с `LogisticRegression` (отбор топ-20 признаков)
- Удаление одного признака из каждой высококоррелированной пары (14 пар, порог phi_k > 0.9)

**Итоговый набор:** 16 признаков из исходных 39.

### 3. Обучение моделей
| Модель | Описание |
|---|---|
| `DummyClassifier` | Базовый уровень (stratified) |
| `LogisticRegression` | Линейная вероятностная модель |
| `LinearSVC` | Линейный SVM (без вероятностей) |
| `SVC(rbf/poly)` | Нелинейный SVM (тест, отклонён) |

Кросс-валидация: `StratifiedKFold(n_splits=5)`, метрика оптимизации: `average_precision` (PR-AUC).

### 4. Подбор гиперпараметров
`GridSearchCV` для `LogisticRegression` и `LinearSVC`:
- LR: `C ∈ {0.1, 1.0}`, `penalty ∈ {l1, l2}`, `class_weight ∈ {balanced, None}`
- SVC: `C ∈ {0.001, 0.01, 0.1, 1}`, `class_weight ∈ {None, balanced}`, `dual ∈ {auto, False}`

### 5. Калибровка
- Проверка калибровки через `calibration_curve` (до калибровки LinearSVC систематически завышает вероятности)
- Калибровка: `CalibratedClassifierCV(method='isotonic', cv='prefit')` на **отдельной** калибровочной выборке (25% от train)
- Метрики калибровки: **Brier Score**, **ECE** (Expected Calibration Error), **MCE** (Maximum Calibration Error)

---

## Результаты

### Сравнение моделей

| Модель | PR-AUC (CV) | PR-AUC (test) | Brier Score |
|---|---|---|---|
| DummyClassifier | ~0.171 | — | — |
| LogisticRegression (best) | ~0.414 | ~0.414 | ~0.125 |
| LinearSVC (до калибровки) | ~0.414 | ~0.414 | ~0.215 |
| **LinearSVC (после калибровки)** | — | — | **~0.125** |

### Метрики калибровки LinearSVC

| | Brier Score | ECE | MCE |
|---|---|---|---|
| До калибровки | 0.215 | высокое | ~0.332 |
| **После калибровки** | **0.125** | **низкое** | **~0.267** |

### Топ-5 важных признаков

| Ранг | Признак | Тип кодирования |
|---|---|---|
| 1 | `app_id` | Target Encoding |
| 2 | `site_id` | Target Encoding |
| 3 | `device_model` | Target Encoding |
| 4 | `ml_feature_9` | Числовой |
| 5 | `ml_feature_10` | Числовой |

**Вывод:** финальная откалиброванная модель (LinearSVC + isotonic) превосходит базовый уровень в **2.4 раза** по PR-AUC и имеет Brier Score, сопоставимый с LogisticRegression. Предсказанные вероятности достаточно достоверны для использования в рекламном аукционе.

---

## Установка и запуск

### 1. Клонирование репозитория

```bash
git clone https://github.com/<your-username>/ad-click-ctr-prediction.git
cd ad-click-ctr-prediction
```

### 2. Создание виртуального окружения

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Запуск тетрадки

```bash
jupyter notebook "Sprint_12_proj - Copy.ipynb"
```

Выполните ячейки последовательно от начала до конца. Разделы 1–10 содержат полный ML-пайплайн.

### 5. Сохранение артефактов (раздел 12)

После выполнения всех ячеек тетрадки вставьте и запустите код из `12_save_model.py` в ячейки раздела 12, либо выполните его как скрипт в том же Python-окружении, где запущена тетрадка:

```bash
# Только если переменные уже в памяти (например, через %run в Jupyter)
python 12_save_model.py
```

Артефакты будут сохранены в папку `artifacts/`.

---

## Артефакты модели

После запуска раздела 12 в папке `artifacts/` появятся три файла:

| Файл | Содержимое |
|---|---|
| `preprocessor.joblib` | Обученный `ColumnTransformer` с пайплайнами предобработки |
| `calibrated_model.joblib` | `CalibratedClassifierCV` (LinearSVC + isotonic regression) |
| `feature_info.joblib` | Словарь с группами признаков, стратегиями кодирования и списком удалённых признаков |

### Пример инференса в продакшене

```python
import joblib
import pandas as pd

# Загрузка артефактов
model = joblib.load("artifacts/calibrated_model.joblib")
feature_info = joblib.load("artifacts/feature_info.joblib")

# Новые данные (DataFrame с теми же столбцами, что в обучающей выборке)
X_new = pd.DataFrame([{
    "hour": 14102100,
    "site_id": "1fbe01fe",
    "app_id": "ecad2386",
    "device_model": "8a4875bd",
    "device_conn_type": 0,
    # ... остальные признаки
}])

# Предсказание откалиброванной вероятности клика
ctr_proba = model.predict_proba(X_new)[:, 1]
print(f"Предсказанный CTR: {ctr_proba[0]:.4f}")
```

---

## Стек технологий

| Категория | Библиотеки |
|---|---|
| Данные и анализ | `pandas`, `numpy`, `scipy` |
| Визуализация | `matplotlib`, `seaborn` |
| Корреляции | `phik` |
| ML-пайплайн | `scikit-learn` (Pipeline, ColumnTransformer, GridSearchCV) |
| Предобработка | `StandardScaler`, `OneHotEncoder`, `TargetEncoder`, `SimpleImputer` |
| Отбор признаков | `VarianceThreshold`, `RFE`, `mutual_info_classif` |
| Модели | `LogisticRegression`, `LinearSVC`, `DummyClassifier` |
| Калибровка | `CalibratedClassifierCV`, `calibration_curve` |
| Метрики | `average_precision_score`, `brier_score_loss` + кастомные ECE/MCE |
| Сериализация | `joblib` |

---

## Лицензия

MIT License. Данные анонимизированы и предоставлены в учебных целях.
