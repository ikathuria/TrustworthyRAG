from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
import logging


@dataclass
class Entity:
    """Cybersecurity entity representation"""
    text: str                        # Entity text (e.g., "WannaCry")
    type: str                       # Entity type (e.g., "MALWARE")
    start_pos: int = 0               # Start position in text
    end_pos: int = 0                 # End position in text
    confidence: float = 1.0          # Confidence score
    source_doc: str = ""             # Source document
    metadata: Dict[str, Any] = None  # Additional metadata
    properties: Dict[str, Any] = None  # Additional properties

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.properties is None:
            self.properties = {}


@dataclass
class Relation:
    """Relationship between entities"""
    source: Entity              # Source entity
    target: Entity              # Target entity
    type: str                   # Relation type (e.g., "EXPLOITS")
    confidence: float = 0.8     # Confidence score
    context: str = ""           # Supporting context
    metadata: Dict[str, Any] = None
    properties: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        if self.properties is None:
            self.properties = {}


class BaseExtractor(ABC):
    """Abstract base class for entity and relation extraction"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._logger = self._setup_logging()

    @abstractmethod
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text"""
        pass

    @abstractmethod
    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities"""
        pass

    def _setup_logging(self) -> logging.Logger:
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
