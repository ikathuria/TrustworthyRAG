import json
import yaml
from typing import Dict, Any
from pathlib import Path


class DomainSchema:
    """
    Handles loading and validating the domain extraction schema,
    including entities, relations, label mapping, and patterns.
    """

    def __init__(self, schema_file: str = None, schema_dict: Dict[str, Any] = None):
        self.schema_data = {}
        self.load_successful = False

        if schema_dict:
            self.schema_data = schema_dict
            self.load_successful = True
        elif schema_file:
            self.load_successful = self.load_schema_from_file(schema_file)
        else:
            raise ValueError(
                "Either schema_file or schema_dict must be provided.")

        if self.load_successful:
            self._validate_schema()
        else:
            raise RuntimeError("Failed to load domain schema.")

    def load_schema_from_file(self, filepath: str) -> bool:
        path = Path(filepath)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Schema file not found: {filepath}")

        try:
            if filepath.endswith(".json"):
                with open(filepath, "r") as f:
                    self.schema_data = json.load(f)
            elif filepath.endswith((".yaml", ".yml")):
                with open(filepath, "r") as f:
                    self.schema_data = yaml.safe_load(f)
            else:
                raise ValueError(
                    "Unsupported schema file format (expected .json, .yaml or .yml)")

            return True
        except Exception as e:
            print(f"Failed to load schema file: {e}")
            return False

    def _validate_schema(self):
        # Check required keys and value types

        required_keys = ["entity_types", "relation_types"]

        for key in required_keys:
            if key not in self.schema_data:
                raise ValueError(
                    f"Domain schema missing required key: '{key}'")

        # entity_types should be dict with keys as entity labels
        if not isinstance(self.schema_data.get("entity_types"), dict):
            raise ValueError("Field 'entity_types' must be a dictionary")

        # relation_types should be dict with keys as relation names
        if not isinstance(self.schema_data.get("relation_types"), dict):
            raise ValueError("Field 'relation_types' must be a dictionary")

        # ner_label_map is optional but if present must be dict
        if "ner_label_map" in self.schema_data and not isinstance(self.schema_data["ner_label_map"], dict):
            raise ValueError("Field 'ner_label_map' must be a dictionary")

        # Each entity_type may contain 'patterns' list
        for ent, val in self.schema_data["entity_types"].items():
            if not isinstance(val, dict):
                raise ValueError(
                    f"Entity type '{ent}' value must be a dictionary")
            patterns = val.get("patterns", [])
            if not isinstance(patterns, list):
                raise ValueError(
                    f"'patterns' of entity type '{ent}' must be a list")

        # Each relation_type may contain 'patterns' and 'entity_pairs'
        for rel, val in self.schema_data["relation_types"].items():
            if not isinstance(val, dict):
                raise ValueError(
                    f"Relation type '{rel}' value must be a dictionary")
            patterns = val.get("patterns", [])
            if not isinstance(patterns, list):
                raise ValueError(
                    f"'patterns' of relation type '{rel}' must be a list")
            entity_pairs = val.get("entity_pairs", [])
            if not isinstance(entity_pairs, list):
                raise ValueError(
                    f"'entity_pairs' of relation type '{rel}' must be a list")

    def get_entity_types(self) -> Dict[str, Dict[str, Any]]:
        return self.schema_data.get("entity_types", {})

    def get_relation_types(self) -> Dict[str, Dict[str, Any]]:
        return self.schema_data.get("relation_types", {})

    def get_label_map(self) -> Dict[str, str]:
        return self.schema_data.get("ner_label_map", {})

    def get_entity_patterns(self) -> Dict[str, list]:
        patterns = {}
        for ent, defn in self.schema_data.get("entity_types", {}).items():
            pats = defn.get("patterns", [])
            patterns[ent] = pats
        return patterns

    def get_relation_patterns(self) -> Dict[str, list]:
        patterns = {}
        for rel, defn in self.schema_data.get("relation_types", {}).items():
            pats = defn.get("patterns", [])
            patterns[rel] = pats
        return patterns
