from typing import List, Dict, Any
import re
import spacy
from spacy.matcher import Matcher
from spacy.tokens import Span

from src.utils.base_extractor import BaseExtractor, Entity, Relation


class CybersecEntityExtractor(BaseExtractor):
    """Cybersecurity entity extractor using SpaCy + custom patterns"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._initialize_model()
        self._load_patterns()

    def _initialize_model(self):
        """Initialize SpaCy model"""
        try:
            # Use English model with NER
            model_name = self.config.get('spacy_model', 'en_core_web_lg')
            self.nlp = spacy.load(model_name)
            self._logger.info(f"Loaded SpaCy model: {model_name}")
        except Exception as e:
            self._logger.error(f"Error loading SpaCy model: {e}")
            raise

    def _load_patterns(self):
        """Load cybersecurity-specific patterns"""
        self.regex_patterns = {
            'CVE': re.compile(r'CVE-\d{4}-\d{4,7}', re.IGNORECASE),
            'IP': re.compile(r'\b(?:\d{1,3}\.){3}\d{1,3}\b'),
            'HASH_MD5': re.compile(r'\b[a-fA-F0-9]{32}\b'),
            'HASH_SHA1': re.compile(r'\b[a-fA-F0-9]{40}\b'),
            'HASH_SHA256': re.compile(r'\b[a-fA-F0-9]{64}\b'),
            'EMAIL': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'URL': re.compile(r'https?://[^\s<>"{}|\\^`\[\]]+'),
            'DOMAIN': re.compile(r'\b(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}\b'),
            'PORT': re.compile(r'(?:port|tcp|udp)\s+(\d+)', re.IGNORECASE)
        }

        # Malware keywords
        self.malware_keywords = {
            'trojan', 'backdoor', 'rootkit', 'ransomware', 'worm', 'virus',
            'spyware', 'adware', 'botnet', 'exploit', 'dropper', 'loader',
            'ginwui', 'ripgof', 'dasher', 'pcshare', 'mdropper', 'daserf'
        }

        # Attack pattern keywords
        self.attack_keywords = {
            'zero-day', 'phishing', 'spear-phishing', 'ddos', 'dos',
            'sql injection', 'xss', 'csrf', 'buffer overflow', 'privilege escalation'
        }

        # System/software keywords
        self.system_keywords = {
            'windows', 'linux', 'macos', 'microsoft', 'office', 'word', 'excel',
            'powerpoint', 'adobe', 'java', 'apache', 'nginx', 'iis'
        }

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract cybersecurity entities from text"""
        entities = []

        # Process with SpaCy
        doc = self.nlp(text)

        # Extract standard NER entities (ORG, PERSON, GPE, etc.)
        for ent in doc.ents:
            entity = Entity(
                text=ent.text,
                label=self._map_spacy_label(ent.label_),
                start_pos=ent.start_char,
                end_pos=ent.end_char,
                confidence=0.85,
                metadata={'extraction_method': 'spacy_ner'}
            )
            entities.append(entity)

        # Extract pattern-based entities
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)

        # Extract cybersecurity-specific entities
        cybersec_entities = self._extract_cybersec_entities(text, doc)
        entities.extend(cybersec_entities)

        # Deduplicate and enhance
        entities = self._deduplicate_entities(entities)

        self._logger.info(f"Extracted {len(entities)} entities")
        return entities

    def _map_spacy_label(self, spacy_label: str) -> str:
        """Map SpaCy labels to cybersecurity taxonomy"""
        mapping = {
            'ORG': 'ORGANIZATION',
            'PERSON': 'PERSON',
            'GPE': 'LOCATION',
            'PRODUCT': 'SYSTEM',
            'DATE': 'DATE',
            'CARDINAL': 'NUMBER'
        }
        return mapping.get(spacy_label, spacy_label)

    def _extract_pattern_entities(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []

        for entity_type, pattern in self.regex_patterns.items():
            for match in pattern.finditer(text):
                entity = Entity(
                    text=match.group(),
                    label=entity_type,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.95,
                    metadata={'extraction_method': 'regex'}
                )
                entities.append(entity)

        return entities

    def _extract_cybersec_entities(self, text: str, doc) -> List[Entity]:
        """Extract cybersecurity-specific entities with better patterns"""
        entities = []

        # Look for "X is the author of Y" patterns
        author_pattern = re.compile(
            r'(\w+(?:\s+\w+)?)\s+is\s+the\s+author\s+of\s+(?:the\s+)?(\w+)',
            re.IGNORECASE
        )

        for match in author_pattern.finditer(text):
            author = match.group(1)
            creation = match.group(2)

            # Add author as PERSON
            entities.append(Entity(
                text=author,
                label='PERSON',
                start_pos=match.start(1),
                end_pos=match.end(1),
                confidence=0.95,
                metadata={'role': 'author'}
            ))

            # Add creation as MALWARE
            entities.append(Entity(
                text=creation,
                label='MALWARE',
                start_pos=match.start(2),
                end_pos=match.end(2),
                confidence=0.95,
                metadata={'created_by': author}
            ))

        # Existing malware keyword detection...
        # (keep your current code)

        return entities

    def _get_entity_span(self, text: str, start: int, end: int, doc) -> str:
        """Extend entity span to capture full name"""
        # Simple heuristic: extend to adjacent capitalized words
        extended_text = text[start:end]

        # Look ahead for capitalized words
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1

        while pos < len(text) and (text[pos].isupper() or text[pos].isdigit()):
            word_end = pos
            while word_end < len(text) and text[word_end].isalnum():
                word_end += 1
            extended_text += " " + text[pos:word_end]
            pos = word_end
            while pos < len(text) and text[pos].isspace():
                pos += 1

        return extended_text.strip()

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate entities"""
        seen = {}
        unique_entities = []

        for entity in entities:
            # Create key based on text and position
            key = (entity.text.lower(), entity.start_pos, entity.end_pos)

            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing and patterns"""
        relations = []

        # Process text with SpaCy
        doc = self.nlp(text)

        # Define relation patterns
        relation_patterns = {
            'CREATED_BY': ['created by', 'authored by', 'developed by', 'written by'],
            'EXPLOITS': ['exploits', 'targets', 'attacks', 'compromises'],
            'USES': ['uses', 'utilizes', 'employs', 'leverages'],
            'AFFECTS': ['affects', 'impacts', 'targets'],
            'MEMBER_OF': ['member of', 'part of', 'affiliated with']
        }

        # Check entity pairs
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Skip if too far apart
                if abs(ent1.start_pos - ent2.start_pos) > 200:
                    continue

                # Get context
                start = min(ent1.start_pos, ent2.start_pos)
                end = max(ent1.end_pos, ent2.end_pos)
                context = text[start:end].lower()

                # Check for relation patterns
                for rel_type, patterns in relation_patterns.items():
                    for pattern in patterns:
                        if pattern in context:
                            relation = Relation(
                                head_entity=ent1,
                                tail_entity=ent2,
                                relation_type=rel_type,
                                confidence=0.8,
                                context=context[:200]
                            )
                            relations.append(relation)
                            break

        self._logger.info(f"Extracted {len(relations)} relations")
        return relations
