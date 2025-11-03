import re
import json
from typing import List, Dict, Any, Optional
from src.utils.base_extractor import BaseExtractor, Entity, Relation
from src.utils.schema_loader import DomainSchema


class HybridEntityExtractor(BaseExtractor):
    """
    Modular, schema-driven, hybrid extractor supporting pattern+LLM extraction for entities and relations.
    """

    def __init__(self, config: Dict[str, Any], domain_schema: DomainSchema, llm: Optional[Any] = None):
        super().__init__(config)
        self.schema = domain_schema
        self.llm = llm
        self.use_llm_entities = config.get("use_llm_entities", False)
        self.use_llm_relations = config.get("use_llm_relations", False)
        self._initialize_nlp()
        self.entity_patterns = self._compile_entity_patterns()
        self.relation_patterns = self._compile_relation_patterns()
        self.label_map = self.schema.get_label_map()

    def _initialize_nlp(self):
        model_name = self.config.get("spacy_model", "en_core_web_lg")
        import spacy
        self.nlp = spacy.load(model_name)

    def _compile_entity_patterns(self):
        ent_patterns = {}
        for ent, val in self.schema.get_entity_types().items():
            patterns = val.get("patterns", [])
            ent_patterns[ent] = [re.compile(p, re.IGNORECASE)
                                 for p in patterns]
        return ent_patterns

    def _compile_relation_patterns(self):
        rel_patterns = {}
        for rel, val in self.schema.get_relation_types().items():
            patterns = val.get("patterns", [])
            rel_patterns[rel] = [re.compile(p, re.IGNORECASE)
                                 for p in patterns]
        return rel_patterns

    def extract_entities(self, text: str) -> List[Entity]:
        entities = []

        # 1. spaCy NER + domain schema mapping
        doc = self.nlp(text)
        for ent in doc.ents:
            mapped_label = self.label_map.get(ent.label_, ent.label_)
            if mapped_label in self.schema.get_entity_types():
                entities.append(Entity(
                    text=ent.text, label=mapped_label,
                    start_pos=ent.start_char, end_pos=ent.end_char, confidence=0.85
                ))

        # 2. Regex/Pattern-based extraction from schema
        for ent_type, regexes in self.entity_patterns.items():
            for regex in regexes:
                for match in regex.finditer(text):
                    entities.append(Entity(
                        text=match.group(), label=ent_type,
                        start_pos=match.start(), end_pos=match.end(),
                        confidence=0.95, metadata={"from": "pattern"}
                    ))

        # 3. LLM-assisted extraction (optional, plug in your LLM here)
        if self.llm and self.use_llm_entities:
            prompt = (
                "You are a cybersecurity information extraction assistant.\n"
                f"Given the domain entity types: {list(self.schema.get_entity_types().keys())}\n"
                "Extract entities from the following text and returning a list of "
                "JSONs with keys: text, label, start_pos, end_pos. "
                "Only give the JSON in your response:\n\n"
                f"{text}\n"
            )
            llm_response = self.llm(prompt)
            self._logger.info(f"LLM response:\n{llm_response}")
            # Clean and parse LLM response
            try:
                llm_entities = json.loads(llm_response)
                for e in llm_entities:
                    # Defensive conversion
                    entities.append(Entity(
                        text=e["text"],
                        label=e["label"],
                        start_pos=int(e.get("start_pos", 0)),
                        end_pos=int(e.get("end_pos", 0)),
                        confidence=0.80,
                        metadata={"from": "llm"}
                    ))
            except Exception as e:
                self._logger.warning(f"LLM entity parsing failed: {e}")

        return self._deduplicate_entities(entities)

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        relations = []

        # 1. Pattern-based schema relation extraction
        for rel_type, rel_def in self.schema.get_relation_types().items():
            for (head_label, tail_label) in rel_def.get("entity_pairs", []):
                head_ents = [e for e in entities if e.label == head_label]
                tail_ents = [e for e in entities if e.label == tail_label]
                for head in head_ents:
                    for tail in tail_ents:
                        if head is tail:
                            continue
                        start = min(head.start_pos, tail.start_pos)
                        end = max(head.end_pos, tail.end_pos)
                        context = text[max(0, start-50)
                                           :min(len(text), end+50)].lower()
                        for regex in self.relation_patterns[rel_type]:
                            if regex.search(context):
                                relations.append(Relation(
                                    head_entity=head,
                                    tail_entity=tail,
                                    relation_type=rel_type,
                                    confidence=0.9,
                                    context=context[:200],
                                    metadata={"from": "pattern"}
                                ))
                                break

        # 2. LLM relation extraction (optional, plug in your LLM chain here)
        if self.llm and self.use_llm_relations:
            entity_mentions = "\n".join(
                [f"{e.text} ({e.label})" for e in entities])
            rel_types = list(self.schema.get_relation_types().keys())
            prompt = (
                "You are a cybersecurity information extraction assistant.\n"
                "Given the following entity list and text, extract all relations.\n"
                f"Entity list: {entity_mentions}\n"
                f"Relation types: {rel_types}\n"
                f"Text:\n{text}\n"
                "Output a JSON list of objects: "
                '{"head": "text", "tail": "text", "type": "relation_type"}'
                "Only give the JSON in your response:\n\n"
            )
            llm_response = self.llm(prompt)
            self._logger.info(f"LLM relation response:\n{llm_response}")
            try:
                llm_rels = json.loads(llm_response)
                for r in llm_rels:
                    head = next(
                        (e for e in entities if e.text == r["head"]), None)
                    tail = next(
                        (e for e in entities if e.text == r["tail"]), None)
                    if head and tail:
                        relations.append(Relation(
                            head_entity=head,
                            tail_entity=tail,
                            relation_type=r["type"],
                            confidence=0.8,
                            context="",
                            metadata={"from": "llm"}
                        ))
            except Exception:
                self._logger.warning("LLM relation extraction failed.")

        return relations

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        unique = {}
        for e in entities:
            key = (e.text.lower(), e.label)
            if key not in unique or e.confidence > unique[key].confidence:
                unique[key] = e
        return list(unique.values())
