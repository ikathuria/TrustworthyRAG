import os
from typing import List, Dict, Tuple
from PIL import Image
import torch
import open_clip
from src.utils.base_extractor import Entity
from src.neo4j_manager import Neo4jManager
from sentence_transformers import util


class MultimodalGrounderOpenCLIP:
    def __init__(self,
                 neo4j_manager: Neo4jManager,
                 device: str = None,
                 similarity_threshold: float = 0.75):
        self.neo4j_manager = neo4j_manager
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.similarity_threshold = similarity_threshold

        # Initialize OpenCLIP model and preprocess
        self.model, self.preprocess = self._load_openclip_model()
        self.model.eval()

    def _load_openclip_model(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k"
        )
        model = model.to(self.device)
        return model, preprocess

    def enrich_entity_text(self, entity: Entity, context_text: str, window: int = 100) -> str:
        """Add context window around entity mention in the document text"""
        start = max(0, entity.start_pos - window)
        end = min(len(context_text), entity.end_pos + window)
        enriched_text = context_text[start:entity.start_pos] + \
            entity.text + context_text[entity.end_pos:end]
        return enriched_text.replace('\n', ' ')

    def embed_text_entities(self, entities: List[Entity], full_text: str) -> Dict[int, torch.Tensor]:
        enriched_texts = [self.enrich_entity_text(e, full_text) for e in entities]
        print(f"Encriched texts for embedding: {enriched_texts}")
        tokenizer = open_clip.get_tokenizer("ViT-B-32")
        tokenized = tokenizer(enriched_texts).to(self.device)
        with torch.no_grad(), torch.autocast(self.device):
            embeddings = self.model.encode_text(tokenized)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
        return {i: embeddings[i] for i in range(len(entities))}

    def embed_images(self, image_paths: List[str]) -> Dict[str, torch.Tensor]:
        image_embeds = {}
        for path in image_paths:
            try:
                image = Image.open(path).convert("RGB")
                input_tensor = self.preprocess(
                    image).unsqueeze(0).to(self.device)
                with torch.no_grad(), torch.autocast(self.device):
                    embed = self.model.encode_image(input_tensor)
                    embed /= embed.norm(dim=-1, keepdim=True)
                    image_embeds[path] = embed
            except Exception as e:
                print(f"Failed to embed image {path}: {e}")
        return image_embeds

    def link_entities_to_images(
        self, entities: List[Entity], entity_embeds: Dict[int, torch.Tensor], image_embeds: Dict[str, torch.Tensor]
    ) -> List[Tuple[Entity, str, float]]:
        links = []
        for idx, entity_embed in entity_embeds.items():
            for img_path, img_embed in image_embeds.items():
                sim = util.cos_sim(entity_embed, img_embed).item()
                if sim >= self.similarity_threshold:
                    links.append((entities[idx], img_path, sim))
        return links

    def ingest_links_to_graph(self, links: List[Tuple[Entity, str, float]], caption_map: Dict[str, str]) -> None:
        for entity, img_path, score in links:
            caption = caption_map.get(img_path, "")
            cypher_query = """
                MERGE (e:Entity {id: $entity_id})
                MERGE (img:Image {path: $img_path})
                SET img.caption = $caption
                MERGE (e)-[r:DEPICTED_IN]->(img)
                SET r.similarity = $score, r.ingested_at = datetime()
            """
            params = {
                "entity_id": f"{entity.label}_{entity.text.lower().strip()}",
                "img_path": img_path,
                "caption": caption,
                "score": float(score)
            }
            try:
                self.neo4j_manager.query_graph(cypher_query, params)
            except Exception as e:
                print(
                    f"Error ingesting link for {entity.text} and {img_path}: {e}")

    def process(self, entities: List[Entity], image_paths: List[str], caption_map: Dict[str, str], full_text: str) -> None:
        entity_embeds = self.embed_text_entities(entities, full_text)
        image_embeds = self.embed_images(image_paths)
        links = self.link_entities_to_images(
            entities, entity_embeds, image_embeds)

        print(
            f"Found {len(links)} entity-image links with similarity > {self.similarity_threshold}")
        self.ingest_links_to_graph(links, caption_map)
