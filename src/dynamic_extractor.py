import re
import json
from typing import List, Dict, Any

import spacy

from src.utils.base_extractor import BaseExtractor
from src.utils.base_extractor import Entity, Relation
from src.utils.constants import SPACY_MODEL


class GenericEntityExtractor(BaseExtractor):
    """Domain-adaptable entity extractor, LLM-augmented"""

    def __init__(self, config: Dict[str, Any], domain_schema: Dict[str, Any] = None, llm=None):
        super().__init__(config)
        self.domain_schema = domain_schema or {}
        self.llm = llm
        self._initialize_spacy()
        self._compile_patterns()

    def _initialize_spacy(self):
        model_name = self.config.get('spacy_model', SPACY_MODEL)
        self.nlp = spacy.load(model_name)

    def _compile_patterns(self):
        # Accepts {"ENTITY_TYPE": [pattern1, pattern2, ...], ...}
        self.patterns = {}
        for ent_type, pats in self.domain_schema.get("entity_patterns", {}).items():
            self.patterns[ent_type] = [
                re.compile(p, re.IGNORECASE) for p in pats]

    def extract_entities(self, text: str) -> List[Entity]:
        doc = self.nlp(text)
        entities = []

        # 1. spaCy NER
        for ent in doc.ents:
            mapped = self.domain_schema.get(
                "spacy_label_map", {}).get(ent.label_, ent.label_)
            entities.append(Entity(text=ent.text, label=mapped,
                                   start_pos=ent.start_char, end_pos=ent.end_char,
                                   confidence=0.85))

        # 2. Pattern-based (regex/keyword) extraction
        for label, regexes in self.patterns.items():
            for regex in regexes:
                for match in regex.finditer(text):
                    entities.append(Entity(text=match.group(), label=label,
                                           start_pos=match.start(), end_pos=match.end(),
                                           confidence=0.95))

        # 3. LLM-augmented/zero-shot NER (if enabled)
        if self.llm and self.config.get("use_llm_entities", False):
            prompt = (
                f"You are an expert information extractor. "
                f"Given the domain schema: {list(self.domain_schema.get('entity_types', []))}. "
                "Identify entities and their types in this passage:\n\n"
                f"{text}\n\n"
                "Return as a list of JSON objects with 'text', 'label', and 'span':"
            )
            llm_response = self.llm(prompt)
            try:
                llm_entities = json.loads(llm_response)
                for e in llm_entities:
                    entities.append(
                        Entity(**e, confidence=0.80, metadata={"llm": True}))
            except Exception:
                self._logger.warning("LLM entity parsing failed.")

        return self._deduplicate_entities(entities)

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        if self.llm and self.config.get("use_llm_relations", False):
            entity_list = "\n".join(
                [f"{e.text} [{e.label}]" for e in entities])
            rel_types = list(self.domain_schema.get("relation_types", []))
            prompt = (
                "You are an expert in information extraction.\n"
                f"Entity list: {entity_list}\n"
                f"Relation types: {rel_types}\n"
                f"Text:\n{text}\n"
                "For every pair of entities, if a relation exists, output a JSON object: {\"head\": str, \"tail\": str, \"type\": str}."
            )
            response = self.llm(prompt)
            try:
                pairs = json.loads(response)
                relations = [Relation(head_entity=next(e for e in entities if e.text == r['head']),
                                        tail_entity=next(
                                            e for e in entities if e.text == r['tail']),
                                        relation_type=r['type'],
                                        confidence=0.80) for r in pairs]
                return relations
            except Exception:
                self._logger.warning("LLM relation extraction failed.")