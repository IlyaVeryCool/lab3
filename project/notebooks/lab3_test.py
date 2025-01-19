import pytest
import numpy as np
from sklearn.metrics import accuracy_score
from mnist_pipeline import MNISTPipeline

# Укажите пути к тестовым данным
TEST_DATA_PATH = "../data"

@pytest.fixture
def mnist_pipeline():
    """Фикстура для инициализации экземпляра MNISTPipeline."""
    pipeline = MNISTPipeline(TEST_DATA_PATH)
    return pipeline

def test_load_data(mnist_pipeline):
    """Тест загрузки данных."""
    mnist_pipeline.load_data()
    assert mnist_pipeline.X_train is not None, "Не удалось загрузить обучающие изображения."
    assert mnist_pipeline.y_train is not None, "Не удалось загрузить обучающие метки."
    assert mnist_pipeline.X_test is not None, "Не удалось загрузить тестовые изображения."
    assert mnist_pipeline.y_test is not None, "Не удалось загрузить тестовые метки."
    assert mnist_pipeline.X_train.shape[1:] == (28, 28), "Размер изображений не соответствует 28x28."

def test_preprocess_data(mnist_pipeline):
    """Тест предобработки данных."""
    mnist_pipeline.load_data()
    mnist_pipeline.preprocess_data()
    assert mnist_pipeline.X_train.min() >= 0.0 and mnist_pipeline.X_train.max() <= 1.0, "Обучающие данные не нормализованы."
    assert mnist_pipeline.X_test.min() >= 0.0 and mnist_pipeline.X_test.max() <= 1.0, "Тестовые данные не нормализованы."

def test_validate_inputs(mnist_pipeline):
    """Тест валидации входных данных."""
    mnist_pipeline.load_data()
    mnist_pipeline.preprocess_data()
    # Проверяем, что метод validate_inputs не выбрасывает исключений
    try:
        mnist_pipeline.validate_inputs(mnist_pipeline.X_train)
        mnist_pipeline.validate_inputs(mnist_pipeline.X_test)
    except ValueError as e:
        pytest.fail(f"Ошибка валидации входных данных: {e}")

def test_model_training(mnist_pipeline):
    """Тест обучения модели."""
    mnist_pipeline.load_data()
    mnist_pipeline.preprocess_data()
    mnist_pipeline.validate_inputs(mnist_pipeline.X_train)
    mnist_pipeline.validate_inputs(mnist_pipeline.X_test)
    mnist_pipeline.train_model()
    assert mnist_pipeline.model is not None, "Модель не была обучена."

def test_model_evaluation(mnist_pipeline):
    """Тест оценки модели."""
    mnist_pipeline.load_data()
    mnist_pipeline.preprocess_data()
    mnist_pipeline.train_model()
    accuracy, conf_matrix = mnist_pipeline.evaluate_model(False)
    assert accuracy > 0.8, f"Точность модели ниже ожидаемой: {accuracy:.4f}."
    assert conf_matrix.shape == (10, 10), "Размер матрицы ошибок неверный."

def test_batch_processing(mnist_pipeline):
    """Тест обработки данных батчами."""
    mnist_pipeline.load_data()
    mnist_pipeline.preprocess_data()
    batch_size = 1000
    batch_count = 0
    for batch in mnist_pipeline.process_in_batches(mnist_pipeline.X_train, batch_size):
        assert batch.shape[0] <= batch_size, "Размер батча превышает ожидаемый."
        batch_count += 1
    assert batch_count > 0, "Батчи не были обработаны."
