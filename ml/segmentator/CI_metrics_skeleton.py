import json
import numpy as np
from scipy import stats
from ml.segmentator.config import Config

config = Config()
METRICS_SKELETON_JSON = config.METRICS_SKELETON_JSON


def compute_confidence_interval(data, confidence=0.95):
    """
    Вычисляет доверительный интервал для массива чисел.

    Параметры:
        data (list or np.array): числовые значения
        confidence (float): уровень доверия, по умолчанию 0.95 (95%)

    Возвращает:
        (mean, lower, upper)
    """
    a = np.array(data)
    mean = np.mean(a)
    sem = stats.sem(a)  # стандартная ошибка среднего
    h = sem * stats.t.ppf((1 + confidence) / 2., len(a)-1)
    return mean, mean - h, mean + h


def summarize_metrics(values):
    """
    Собирает полную статистику для списка/массива чисел:
        - mean (среднее)
        - median (медиана)
        - std (стандартное отклонение)
        - min, max
        - 95% доверительный интервал

    Возвращает словарь с этими значениями.
    """
    values = np.array(values)
    mean = np.mean(values)
    median = np.median(values)
    std = np.std(values)
    min_val = np.min(values)
    max_val = np.max(values)
    mean_ci, ci_lower, ci_upper = compute_confidence_interval(
        values, confidence=0.95)

    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "min": float(min_val),
        "max": float(max_val),
        "95%_CI_mean": [float(ci_lower), float(ci_upper)]
    }


def summarize_fold_metrics(fold_metrics):
    """
    fold_metrics: dict с ключами "per_image" и "aggregate"

    - per_image: метрики по каждому изображению
    - aggregate: средние метрики по fold

    Возвращает:
        dict с ключами метрик и статистикой:
            {
                "dice": {...},
                "iou": {...},
                ...
            }
    """
    per_image = fold_metrics["per_image"]
    summary = {}

    # по каждой метрике собираем список значений
    metrics_keys = list(next(iter(per_image.values())).keys())

    for key in metrics_keys:
        values = [v[key] for v in per_image.values()]
        summary[key] = summarize_metrics(values)

    return summary


def summarize_all_folds(all_fold_metrics):
    """
    all_fold_metrics: dict[fold_idx] = {"per_image":..., "aggregate":...}

    Возвращает:
        1) Статистику по каждому fold
        2) Общую статистику по всем изображениям всех fold
    """
    folds_summary = {}
    all_values = {}  # собираем все значения по метрикам

    # инициализация
    first_fold = next(iter(all_fold_metrics))
    for key in all_fold_metrics[first_fold]["aggregate"]:
        all_values[key] = []

    for fold_idx, fold_data in all_fold_metrics.items():
        # статистика по fold
        fold_summary = summarize_fold_metrics(fold_data)
        folds_summary[fold_idx] = fold_summary

        # собираем все per-image для глобальной статистики
        for img_metrics in fold_data["per_image"].values():
            for k, v in img_metrics.items():
                all_values[k].append(v)

    # глобальная статистика
    overall_summary = {}
    for k, v in all_values.items():
        overall_summary[k] = summarize_metrics(v)

    return folds_summary, overall_summary


def print_metrics_summary(folds_summary, overall_summary):
    """
    Красиво выводит статистику по всем fold и общую статистику по всему датасету.

    Параметры:
        folds_summary: dict, ключ = fold_idx, значение = статистика fold
        overall_summary: dict, глобальная статистика по всем fold
    """
    print("\n" + "="*80)
    print("СТАТИСТИКА ПО КАЖДОМУ FOLD")
    print("="*80)

    for fold_idx, fold_stats in folds_summary.items():
        print(f"\nFold {fold_idx}:")
        for metric, stats in fold_stats.items():
            print(f"  {metric:10} | mean: {stats['mean']:.4f} | median: {stats['median']:.4f} | "
                  f"std: {stats['std']:.4f} | min: {stats['min']:.4f} | max: {stats['max']:.4f} | "
                  f"95% CI: [{stats['95%_CI_mean'][0]:.4f}, {stats['95%_CI_mean'][1]:.4f}]")

    print("\n" + "="*80)
    print("ОБЩАЯ СТАТИСТИКА ПО ВСЕМ FOLD")
    print("="*80)

    for metric, stats in overall_summary.items():
        print(f"  {metric:10} | mean: {stats['mean']:.4f} | median: {stats['median']:.4f} | "
              f"std: {stats['std']:.4f} | min: {stats['min']:.4f} | max: {stats['max']:.4f} | "
              f"95% CI: [{stats['95%_CI_mean'][0]:.4f}, {stats['95%_CI_mean'][1]:.4f}]")


# Загружаем метрики
with open(METRICS_SKELETON_JSON, "r") as f:
    all_fold_metrics = json.load(f)

# Собираем статистику
folds_summary, overall_summary = summarize_all_folds(all_fold_metrics)

# Печатаем
print_metrics_summary(folds_summary, overall_summary)

# Сохраняем JSON для дальнейшего анализа
with open("all_fold_metrics_stats.json", "w") as f:
    json.dump({
        "per_fold": folds_summary,
        "overall": overall_summary
    }, f, indent=4)
