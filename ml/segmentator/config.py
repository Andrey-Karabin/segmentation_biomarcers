"""Configuration of the fundus image segmentation service"""

import os


class Config():
    #
    TRAIN_IMAGE_DIR: str = 'ml/data/AV-Seg/train/image'
    #
    TEST_IMAGE_DIR: str = 'ml/data/AV-Seg/test/image'
    #
    TRAIN_MASK_DIR: str = 'ml/data/AV-Seg/train/mask'
    #
    TEST_MASK_DIR: str = 'ml/data/AV-Seg/test/mask'
    #
    PATCH_SIZE: int = 512
    #
    BATCH_SIZE: int = 4
    #
    STRIDE: int = 256
    #
    NUM_WORKERS: int = 0
    # TODO переименовать, одна модель для двух сегформеров
    SEGFORMER_SKELETON: str = 'nvidia/segformer-b0-finetuned-ade-512-512'
    #
    EPOCHS: int = 50
    #
    EPOCHS_SEG: int = 50
    # TODO переименовать
    SAVE_DIR: str = 'ml/data/checkpoints/skeleton'
    #
    SAVE_DIR_SEG: str = 'ml/data/checkpoints/segmentation'
    #
    MODEL_SKELETON_BEST: str = 'ml/service/inference/data_skeleton/skeleton_best.pth'
    MODEL_SEGFORMER_BEST: str = "ml/service/inference/data_segmentator/segmentation_best.pth"
    #
    SAVE_DIR_PREDICTION_MASK: str = "ml/data/checkpoints/predictions_mask"
    #
    N_FOLDS: int = 5
    # Гиперпараметры для GridSearch
    LR_LIST: list = [1e-4, 3e-4]
    ALPHA_LIST: list = [0.1, 0.3, 0.5, 0.7, 0.9]
    BETA_LIST: list = [0.9, 0.7, 0.5, 0.3, 0.1]
    RESULTS_PATH: str = "ml/data/search/skeleton_search_results.json"
    RESULT_PATH_SEG: str = "ml/data/search/segmentation_search_results.json"
    # путь для сохранения подобранных метрик
    PATH_SEARCH: str = "ml/data/search"
    SEARCH_EPOCH: int = 10
    # путь для сохранения предсказаний
    PRED_SAVE_DIR: str = 'ml/data/checkpoints/predictions_skeleton'
    # путь для метрик (json / csv)
    METRICS_SAVE_DIR: str = "ml/data/checkpoints/skeleton/metrics"
    #
    METRICS_SAVE_DIR_SEG: str = "ml/data/checkpoints/segmentation/metrics"
    # путь к файлу статистики по метрикам
    METRICS_SKELETON_JSON: str = "ml/data/checkpoints/skeleton/metrics/all_fold_metrics.json"
    METRICS_SEGMENTATION_JSON: str = 'ml/data/checkpoints/segmentation/metrics/all_fold_metrics.json'
    #
    RESULTS_PATH_SEG: str = "ml/data/checkpoints/segmentation/metrics"
    LR_LIST_SEG: list = [1e-4, 3e-4, 1e-3]
    CL_DICE_LIST: list = [0.0, 0.25, 0.5, 1.0]
