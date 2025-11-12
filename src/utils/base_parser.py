import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from datetime import datetime


@dataclass
class ParsedContent:
    text: Dict[str, str]
    tables: List[Dict]
    images: List[Dict]
    metadata: Dict[str, Any]
    source_file: str
    parsed_at: datetime = None

    def __post_init__(self):
        if self.parsed_at is None:
            self.parsed_at = datetime.now()
    
    def to_json(self) -> str:
        return json.dumps({
            "text": self.text,
            "tables": self.tables,
            "images": self.images,
            "metadata": self.metadata,
            "source_file": self.source_file,
            "parsed_at": self.parsed_at.isoformat()
        }, indent=4)
    
    def from_json(json_str: str) -> 'ParsedContent':
        data = json.loads(json_str)
        data['parsed_at'] = datetime.fromisoformat(data['parsed_at'])
        return ParsedContent(**data)

    def to_pkl(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def from_pkl(file_path: str) -> 'ParsedContent':
        with open(file_path, 'rb') as f:
            return pickle.load(f)


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

    def load_from_json(self, json_str: str):
        data = json.loads(json_str)
        data['parsed_at'] = datetime.fromisoformat(data['parsed_at'])
        return ParsedContent(**data)

    def load_from_pkl(self, file_path: str):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
