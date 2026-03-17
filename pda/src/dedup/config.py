from dataclasses import dataclass


@dataclass(frozen=True)
class DedupConfig:
    name_similarity_threshold: int = 88
    phone_digits: int = 10


DEFAULT_CONFIG = DedupConfig()
