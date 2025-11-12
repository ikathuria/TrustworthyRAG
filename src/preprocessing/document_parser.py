from pathlib import Path
from typing import List, Dict, Any
from PIL import Image
from pdf2image import convert_from_path
import traceback

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from mineru_vl_utils import MinerUClient

from src.utils.base_parser import BaseParser, ParsedContent
import src.utils.constants as C


class DocumentParser(BaseParser):
    def __init__(self, config: Dict[str, Any], load_from_file: str = False, filepath: str = ""):
        super().__init__(config)
        if load_from_file and filepath:
            self.load_parsed_content([filepath])
        self._initialize_model()

    def _initialize_model(self):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            C.MINERU_MODEL_NAME,
            dtype=self.config.get("dtype", "auto"),
            device_map=self.config.get("device_map", "auto")
        )
        self.processor = AutoProcessor.from_pretrained(
            C.MINERU_MODEL_NAME,
            use_fast=True
        )
        self.client = MinerUClient(
            backend="transformers",
            model=self.model,
            processor=self.processor
        )

    def load_parsed_content(self, parsed_pkl_paths: List[str]) -> List[ParsedContent]:
        """Load previously parsed content from pickle files"""
        self.parsed_content = []
        for pkl_path in parsed_pkl_paths:
            try:
                parsed_content = ParsedContent.from_pkl(pkl_path)
                self.parsed_content.append(parsed_content)
                print(f"Loaded: {pkl_path}")
            except Exception as e:
                print(f"Failed to load {pkl_path}: {str(e)}")
        return self.parsed_content

    def parse(self, file_path: str) -> ParsedContent:
        path = Path(file_path)
        try:
            if path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']:
                image = Image.open(file_path)
                blocks = self.client.two_step_extract(image)
                page_images = [image]
            elif path.suffix.lower() == '.pdf':
                page_images = convert_from_path(file_path)
                blocks = []
                for page_num, img in enumerate(page_images):
                    page_blocks = self.client.two_step_extract(img)
                    for block in page_blocks:
                        block['page_num'] = page_num + 1
                    blocks.extend(page_blocks)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

            self.parsed_content = self._process_blocks(
                blocks, str(file_path), page_images
            )

            # Save extracted content
            self._save_parsed_content(self.parsed_content)

            return self.parsed_content

        except Exception as e:
            self._logger.error(f"Error parsing {file_path}: {str(e)}")
            if self.parsed_content:
                return self.parsed_content
            traceback.print_exc()
            raise

    def _process_blocks(self, blocks: List[Dict], source_file: str, page_images: List[Image.Image]) -> ParsedContent:
        """Processes extracted blocks into ParsedContent.

        For each block, depending on its type, we extract text, tables, and images.
        Image blocks will include bounding box metadata even if no image content is extracted.

        Args:
            blocks (List[Dict]): Extracted blocks from MinerU2.5.
            source_file (str): The source file path.

        Returns:
            ParsedContent: The structured parsed content.        
        """
        text_content = []
        tables = []
        images = []

        for block in blocks:
            block_type = block.get('type', 'text')
            content = block.get('content')
            bbox = block.get('bbox', [])
            page_num = block.get('page_num', 1)

            if block_type == 'text':
                if content:
                    text_content.append(content)

            elif block_type == 'title':
                if content:
                    text_content.append(f"\n## {content}\n")

            elif block_type == 'table':
                if content:
                    tables.append({
                        'content': content,
                        'bbox': bbox,
                        'page': page_num
                    })

            elif block_type == 'image':
                image_data = {
                    'bbox': bbox,
                    'page': page_num,
                    'type': 'image_region',
                    'content': content
                }

                self._logger.debug(f"Processing image block on page {page_num} with bbox {bbox}")

                if page_images and bbox and len(bbox) == 4:
                    try:
                        extracted_img = self._extract_image_from_bbox(
                            page_images[page_num - 1],
                            bbox
                        )
                        image_data['image_data'] = extracted_img
                        image_data['has_pixels'] = True
                    except Exception as e:
                        self._logger.warning(
                            f"Failed to extract image pixels: {e}"
                        )
                        image_data['has_pixels'] = False
                else:
                    image_data['has_pixels'] = False

                images.append(image_data)

            elif block_type == 'image_caption':
                if images and content:
                    images[-1]['caption'] = content
                    text_content.append(f"[Image: {content}]")

        return ParsedContent(
            text={"content": "\n".join(text_content)},
            tables=tables,
            images=images,
            metadata={
                'extraction_method': 'MinerU2.5',
                'total_blocks': len(blocks),
                'image_regions_found': len([b for b in blocks if b.get('type') == 'image']),
                'images_with_pixels': len([i for i in images if i.get('has_pixels')])
            },
            source_file=source_file
        )
    
    def _extract_image_from_bbox(
        self,
        page_image: Image.Image,
        bbox: List[float]
    ) -> Image.Image:
        """
        Extract image region from page using normalized bbox coordinates
        
        Args:
            page_image: PIL Image of the PDF page
            bbox: Normalized bounding box [x0, y0, x1, y1] where values are 0-1
        
        Returns:
            Cropped PIL Image
        """
        # Bbox is normalized (0-1), convert to pixel coordinates
        width, height = page_image.size
        x0, y0, x1, y1 = bbox

        # Convert normalized coordinates to pixels
        pixel_bbox = (
            int(x0 * width),
            int(y0 * height),
            int(x1 * width),
            int(y1 * height)
        )

        # Crop the image
        cropped_image = page_image.crop(pixel_bbox)

        return cropped_image

    def _save_extracted_images(
        self,
        parsed_content: ParsedContent,
        output_dir: str = C.EXTRACTED_IMAGES_DIR
    ) -> List[str]:
        """
        Save all extracted images to disk and update their paths in ParsedContent.
        
        Args:
            parsed_content: Parsed document content
            output_dir: Directory to save images
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        doc_name = Path(parsed_content.source_file).stem

        for idx, img_data in enumerate(parsed_content.images):
            if img_data.get('has_pixels') and 'image_data' in img_data:
                # Create filename
                caption = img_data.get('caption', f'image_{idx}')
                # Sanitize caption for filename
                safe_caption = "".join(
                    c for c in caption if c.isalnum() or c in (' ', '-', '_')).strip()
                safe_caption = safe_caption[:50]  # Limit length

                filename = f"{doc_name}_page{img_data['page']}_{idx}_{safe_caption}.png"
                filepath = output_path / filename

                # Save image
                img_data['image_data'].save(filepath)
                img_data['path'] = str(filepath)

                self._logger.info(
                    f"Saved image: {filepath} for {parsed_content.source_file}")

    def _save_extracted_text(
        self,
        parsed_content: ParsedContent,
        output_dir: str = C.EXTRACTED_TEXT_DIR
    ) -> List[str]:
        """
        Save all extracted text to disk and update their paths in ParsedContent.

        Args:
            parsed_content: Parsed document content
            output_dir: Directory to save text

        Returns:
            List of saved text paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        doc_name = Path(parsed_content.source_file).stem
        filename = f"{doc_name}.txt"
        filepath = output_path / filename

        text_data = parsed_content.text
        if text_data["content"]:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_data['content'])
        text_data['path'] = str(filepath)

        self._logger.info(f"Saved text: {filepath} for {parsed_content.source_file}")

    def _save_parsed_content(
        self, parsed_content: ParsedContent,
        output_dir: str = C.PROCESSED_DATA_DIR
    ) -> str:
        """Save the parsed content to disk as pkl."""

        self._save_extracted_images(parsed_content)
        self._save_extracted_text(parsed_content)

        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)

        doc_name = Path(parsed_content.source_file).stem
        filename = f"{doc_name}_parsed.pkl"
        filepath = output_path / filename

        parsed_content.to_pkl(str(filepath))

        self._logger.info(f"Saved parsed content: {filepath} for {parsed_content.source_file}")
        return str(filepath)

    def parse_batch(self, file_paths: List[str]) -> List[ParsedContent]:
        """Parse multiple documents in a batch.
        
        Args:
            file_paths (List[str]): List of file paths to parse.
        
        Returns:
            List[ParsedContent]: List of parsed content objects.
        """
        results = []
        for file_path in file_paths:
            try:
                results.append(self.parse(file_path))
                self._logger.info(f"Parsed: {file_path}")
            except Exception as e:
                self._logger.error(f"Failed: {file_path} - {str(e)}")
                traceback.print_exc()
        return results
