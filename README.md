
## Расположение данных
Данные загружнены в s3 хранилище MinIO. Трекинг осуществляется с помощью DVC

## Обучение моделей
Проведен первичный выбор двух моделей. Подробности в `/notebooks/experiments.ipynb`.
Скрипты для обучения и выбора моделей и их параметров в `/src/model_selection.py`.

## Трекинг
В качестве инструмента для трекинга выбран ClearML.

Параметры для трекинга логистической регрессии: `max_iter`, `solver`, `penalty`, `class_weight`.
Параметры для трекинга случайного леса: `max_iter`, `solver`, `penalty`, `class_weight`.
Метрики контроля качества: `Accuracy`, `F1_score`, `ROC_AUC`.

Так как случайный лес имеет склонность к переобучению, то для него дополнительно контролируется качество на тренировочной выборке.

Ссылка на эксперименты:
- Логистическая регрессия ([ссылка](https://app.clear.ml/projects/9a44d916d93440a5814968acb7b87d9e/experiments/90f5ea79c00e4d5dbf40f9c80a038b7e/output/execution)) - эксперимент с наилучшими метриками на тестовой выборке
- Случайный лес на тестовой выборке ([ссылка](https://app.clear.ml/projects/9a44d916d93440a5814968acb7b87d9e/experiments/292ea858b7bd431e89f6a9d503b9cfbf/output/execution)) и на тренировочной выборке ([ссылка](https://app.clear.ml/projects/9a44d916d93440a5814968acb7b87d9e/experiments/7ebc2468d379400ebd094f165dbb88c4/output/execution)) - эксперименты с одними из лучших метрик на тестовой выборке и наименьшим разрвывом между качеством на тренировочной и тестовой выборках.

Гиперпараметры тестов вынесены в `/src/config.py` для дальнейшего использования

Для дальнейшей работы выбрана логистическая регрессия, т.к.:
- качество на тестовой выборке близко к решающему лесу
- отсутствуют признаки переобучения
- модель более легкая и быстрая

## Оркестратор
Реализован ежедневный запуск обучения модели в Airflow. Сервер поднимается с помощью docker-compose. Модель и ее метрики хранятся в s3 хранилище.

## API
Реализовано FastAPI приложение и обертка в Streamlit.
Скрипт запуска FasTAPI в `api.py`.
Запуск обертки сервиса из директории `/front` командой `streamlit run front.py`.

## Тестирование
Реализовано автоматизированное тестирование приложения в Gitlab CI. 
Параметры в `.gitlab-ci.yml`
