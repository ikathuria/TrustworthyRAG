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
            'PORT': re.compile(r'(?:port|tcp|udp)\s+(\d+)', re.IGNORECASE),
            'MS_VULN': re.compile(r'MS\d{2}-\d{3}', re.IGNORECASE)
        }

        # Malware keywords
        self.malware_keywords = {
            'trojan', 'backdoor', 'rootkit', 'ransomware', 'worm', 'virus',
            'spyware', 'adware', 'botnet', 'exploit', 'dropper', 'loader',
            'ginwui', 'ripgof', 'dasher', 'pcshare', 'mdropper', 'daserf',
            'booli', 'flux', 'ppdropper', 'malware'
        }

        # Attack pattern keywords
        self.attack_keywords = {
            'zero-day', '0-day', 'phishing', 'spear-phishing', 'ddos', 'dos',
            'sql injection', 'xss', 'csrf', 'buffer overflow', 'privilege escalation',
            'exploit', 'attack', 'compromise', 'breach'
        }

        # System/software keywords
        self.system_keywords = {
            'windows', 'linux', 'macos', 'microsoft', 'office', 'word', 'excel',
            'powerpoint', 'adobe', 'java', 'apache', 'nginx', 'iis', 'system'
        }

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract cybersecurity entities from text"""
        entities = []

        # Process with SpaCy
        doc = self.nlp(text)

        # 1. Extract standard NER entities (ORG, PERSON, GPE, DATE)
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

        # 2. Extract pattern-based entities (CVE, IP, hashes, etc.)
        pattern_entities = self._extract_pattern_entities(text)
        entities.extend(pattern_entities)

        # 3. Extract cybersecurity-specific entities (MALWARE, ATTACK_PATTERN, etc.)
        cybersec_entities = self._extract_cybersec_entities(text, doc)
        entities.extend(cybersec_entities)

        # 4. Deduplicate
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
            'CARDINAL': 'NUMBER',
            'TIME': 'TIME'
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
        """Extract cybersecurity-specific entities"""
        entities = []
        text_lower = text.lower()

        # 1. Look for "X is the author of Y" patterns
        author_patterns = [
            (r'(\w+(?:\s+\w+)?)\s+is\s+the\s+author\s+of\s+(?:the\s+)?(\w+)', 'author'),
            (r'(\w+(?:\s+\w+)?)\s+authored\s+(?:the\s+)?(\w+)', 'authored'),
            (r'(\w+)\s+created\s+by\s+(\w+(?:\s+\w+)?)', 'created_reverse'),
            (r'(\w+(?:\s+\w+)?)\s+developed\s+(?:the\s+)?(\w+)', 'developed')
        ]

        for pattern, pattern_type in author_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                if pattern_type == 'created_reverse':
                    creation = match.group(1)
                    author = match.group(2)
                else:
                    author = match.group(1)
                    creation = match.group(2)

                # Add author as PERSON (or ORGANIZATION if detected)
                entities.append(Entity(
                    text=author,
                    label='PERSON',  # Will be refined later if it's an org
                    start_pos=match.start(
                        2 if pattern_type == 'created_reverse' else 1),
                    end_pos=match.end(2 if pattern_type ==
                                      'created_reverse' else 1),
                    confidence=0.92,
                    metadata={'role': 'author', 'pattern': pattern_type}
                ))

                # Add creation as MALWARE
                entities.append(Entity(
                    text=creation,
                    label='MALWARE',
                    start_pos=match.start(
                        1 if pattern_type == 'created_reverse' else 2),
                    end_pos=match.end(1 if pattern_type ==
                                      'created_reverse' else 2),
                    confidence=0.92,
                    metadata={'created_by': author, 'pattern': pattern_type}
                ))

        # 2. Extract malware by keyword matching
        for token in doc:
            token_lower = token.text.lower()
            if token_lower in self.malware_keywords:
                # Try to get full name (may be multi-word)
                start = token.idx
                end = token.idx + len(token.text)

                # Extend to include adjacent capitalized words or version numbers
                full_name = self._get_full_entity_name(text, start, end, doc)

                entity = Entity(
                    text=full_name,
                    label='MALWARE',
                    start_pos=start,
                    end_pos=start + len(full_name),
                    confidence=0.88,
                    metadata={'extraction_method': 'keyword_match'}
                )
                entities.append(entity)

        # 3. Extract attack patterns
        for keyword in self.attack_keywords:
            for match in re.finditer(re.escape(keyword), text_lower):
                entity = Entity(
                    text=text[match.start():match.end()],
                    label='ATTACK_PATTERN',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.85,
                    metadata={'extraction_method': 'keyword_match'}
                )
                entities.append(entity)

        # 4. Extract systems/software
        for keyword in self.system_keywords:
            for match in re.finditer(r'\b' + re.escape(keyword) + r'\b', text_lower):
                # Get actual text (preserve capitalization)
                actual_text = text[match.start():match.end()]

                entity = Entity(
                    text=actual_text,
                    label='SYSTEM',
                    start_pos=match.start(),
                    end_pos=match.end(),
                    confidence=0.82,
                    metadata={'extraction_method': 'keyword_match'}
                )
                entities.append(entity)

        return entities
    
    def _get_full_entity_name(self, text: str, start: int, end: int, doc) -> str:
        """Extend entity span to capture full name (e.g., 'GinWui rootkit')"""
        extended_text = text[start:end]

        # Look ahead for related words
        pos = end
        while pos < len(text) and text[pos].isspace():
            pos += 1

        # Capture next word if it's capitalized, a number, or common suffix
        while pos < len(text):
            word_end = pos
            while word_end < len(text) and (text[word_end].isalnum() or text[word_end] in '.-_'):
                word_end += 1

            next_word = text[pos:word_end].lower()

            # Common malware/software suffixes
            if next_word in ['trojan', 'backdoor', 'rootkit', 'worm', 'dropper', 'loader',
                             'version', 'v', 'beta', 'attack', 'malware']:
                extended_text = text[start:word_end]
                pos = word_end
                while pos < len(text) and text[pos].isspace():
                    pos += 1
            else:
                break

        return extended_text.strip()

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
            # Create key based on normalized text and position
            key = (entity.text.lower().strip(),
                   entity.start_pos, entity.end_pos)

            if key not in seen or entity.confidence > seen[key].confidence:
                seen[key] = entity

        return list(seen.values())

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations using dependency parsing and patterns"""
        relations = []

        relation_patterns = {
            'CREATED_BY': {
                'patterns': [
                    'is the author of',
                    'authored',
                    'created by',
                    'developed by',
                    'responsible for development',
                    'author of'
                ],
                'entity_pairs': [
                    ('MALWARE', 'PERSON'),
                    ('MALWARE', 'ORGANIZATION')
                ],
                'bidirectional': True
            },
            'EXPLOITS': {
                'patterns': [
                    'exploits',
                    'targets',
                    'attacks',
                    'compromises',
                    'abuses',
                    'exploit',
                    'targeting'
                ],
                'entity_pairs': [
                    ('MALWARE', 'CVE'),
                    ('MALWARE', 'VULNERABILITY'),
                    ('MALWARE', 'SYSTEM'),
                    ('MALWARE', 'MS_VULN'),
                    ('ATTACK_PATTERN', 'SYSTEM'),
                    ('ORGANIZATION', 'SYSTEM')
                ],
                'bidirectional': False
            },
            'USES': {
                'patterns': [
                    'uses',
                    'utilizes',
                    'employs',
                    'leverages',
                    'deploys',
                    'use'
                ],
                'entity_pairs': [
                    ('MALWARE', 'SYSTEM'),
                    ('ORGANIZATION', 'MALWARE'),
                    ('PERSON', 'MALWARE'),
                    ('ATTACK_PATTERN', 'MALWARE')
                ],
                'bidirectional': False
            },
            'AFFECTS': {
                'patterns': [
                    'affects',
                    'impacts',
                    'influences',
                    'damages',
                    'affect'
                ],
                'entity_pairs': [
                    ('CVE', 'SYSTEM'),
                    ('VULNERABILITY', 'SYSTEM'),
                    ('MALWARE', 'SYSTEM'),
                    ('MS_VULN', 'SYSTEM')
                ],
                'bidirectional': False
            },
            'MEMBER_OF': {
                'patterns': [
                    'member of',
                    'part of',
                    'belongs to',
                    'affiliated with',
                    'member'
                ],
                'entity_pairs': [
                    ('PERSON', 'ORGANIZATION')
                ],
                'bidirectional': False
            },
            'OCCURRED_ON': {
                'patterns': [
                    'on',
                    'began on',
                    'started on',
                    'commenced on',
                    'took place',
                    'occurred'
                ],
                'entity_pairs': [
                    ('MALWARE', 'DATE'),
                    ('ATTACK_PATTERN', 'DATE'),
                    ('CVE', 'DATE')
                ],
                'bidirectional': False
            }
        }

        # Check each entity pair
        for i, ent1 in enumerate(entities):
            for ent2 in entities[i+1:]:
                # Skip if entities are too far apart (>300 chars)
                if abs(ent1.start_pos - ent2.start_pos) > 300:
                    continue

                # Get context between entities
                start = min(ent1.start_pos, ent2.start_pos)
                end = max(ent1.end_pos, ent2.end_pos)

                # Expand context window
                context_start = max(0, start - 50)
                context_end = min(len(text), end + 50)
                context = text[context_start:context_end].lower()

                # Try to find matching relation
                for rel_type, rel_info in relation_patterns.items():
                    # Check if entity pair is valid for this relation
                    entity_pair = (ent1.label, ent2.label)
                    reverse_pair = (ent2.label, ent1.label)

                    is_valid_pair = entity_pair in rel_info['entity_pairs']
                    is_valid_reverse = rel_info.get(
                        'bidirectional', False) and reverse_pair in rel_info['entity_pairs']

                    if not (is_valid_pair or is_valid_reverse):
                        continue

                    # Check for relation patterns in context
                    for pattern in rel_info['patterns']:
                        if pattern in context:
                            # Determine direction
                            if is_valid_pair:
                                head, tail = ent1, ent2
                            else:
                                head, tail = ent2, ent1

                            relation = Relation(
                                head_entity=head,
                                tail_entity=tail,
                                relation_type=rel_type,
                                confidence=0.85,
                                context=context[:200],
                                metadata={'pattern_matched': pattern}
                            )
                            relations.append(relation)
                            break  # Found a match, no need to check more patterns

        self._logger.info(f"Extracted {len(relations)} relations")
        return relations
