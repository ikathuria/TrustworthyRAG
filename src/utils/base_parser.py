from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class ParsedContent:
    text: str
    tables: List[Dict]
    images: List[Dict]
    metadata: Dict[str, Any]
    source_file: str
    parsed_at: datetime = None

    def __post_init__(self):
        if self.parsed_at is None:
            self.parsed_at = datetime.now()


class BaseParser(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = self._setup_logging()

    @abstractmethod
    def parse(self, file_path: str) -> ParsedContent:
        pass

    @abstractmethod
    def parse_batch(self, file_paths: List[str]) -> List[ParsedContent]:
        pass

    def _setup_logging(self):
        import logging
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
