from ml.segmentator.config import Config


class TestConfig(Config):
    #
    MIN_VESSEL_RATIO: float = 0.01
    #
    BATCH_SIZE_SK_MODEL: int = 2
    #
    BATCH_SIZE_SEG_MODEL: int = 2
