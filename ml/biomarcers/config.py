import numpy as np


class Config():
    #: игнорируем цвет
    IGNORE_INDEX = 255
    #: размер патча
    PATCH_SIZE = 512
    #: перекрытие патча
    STRIDE = PATCH_SIZE // 2
    #: размер батча
    BATCH_SIZE = 8
    #: количество эпох обучения
    EPOCHS = 20
    #: на чем идет обучение
    DEVICE = "cuda"
    #: Классы
    CLASSES = {
        "ERROR": (211, 255, 5),
        "OD": (250, 250, 55),
        "background": (0, 0, 0),
        "drusen": (115, 71, 30),
        "edema": (109, 230, 213),
        "epiretinal_fibrosis": (88, 4, 46),
        "fibrosis": (196, 67, 237),
        "geographic_atrophy": (250, 189, 124),
        "hard_exudates": (184, 61, 245),
        "hemorrhages": (42, 125, 209),
        "laser_coagulates": (70, 109, 209),
        "macular_hole": (32, 218, 142),
        "microaneurysms": (250, 50, 83),
        "neovascularization": (192, 245, 197),
        "soft_exudates": (61, 245, 61),
        "subretinal_hemorrhage": (98, 243, 161),
        "venous_anomalies": (94, 76, 209),
    }

    BACKGROUND_CLASSES = ["ERROR", "OD", "background"]
    FOREGROUND_CLASSES = [cls for cls in CLASSES.keys() if cls not in [
        "ERROR", "OD", "background"]]
    NUM_CLASSES = len(FOREGROUND_CLASSES) + 1  # + background
    CLASS_TO_ID = {cls: i + 1 for i, cls in enumerate(FOREGROUND_CLASSES)}

    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    CHECKPOINT_DIR = "ml/biomarcers/checkpoints_segformer_15hours"
