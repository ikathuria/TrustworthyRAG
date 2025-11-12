import os
import camelot
from PIL import Image
from sentence_transformers import SentenceTransformer
import torch
import open_clip
import pypdf

# CONFIG
DATA_DIR = "./docs"
TEXT_MODEL = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- TEXT CHUNKING ---
text_model = SentenceTransformer(TEXT_MODEL)


def chunk_text(text, max_len=500, overlap=50):
    chunks = []
    for i in range(0, len(text), max_len-overlap):
        chunks.append(text[i:i+max_len])
    return chunks


# --- IMAGE EMBEDDING & CAPTIONING ---
image_model, _, image_preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer("ViT-B-32")


def embed_and_caption_image(image_path):
    image = image_preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    # Use zero-shot text as caption (could use BLIP for more accurate captions)
    # Example prompt for simplicity:
    prompts = [
        "photo of a cybersecurity diagram",
        "screenshot of a warning page",
        "table of numbers"
    ]
    text_tokens = tokenizer(prompts).to(device)

    with torch.no_grad(), torch.autocast(device):
        image_features = image_model.encode_image(image)
        text_features = image_model.encode_text(text_tokens)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

    caption = prompts[int(text_probs.cpu().numpy()[0].argmax())]
    return image_features.cpu().numpy(), caption

# --- TABLE EXTRACTION ---


def extract_tables_from_pdf(pdf_path):
    tables = camelot.read_pdf(pdf_path, pages="all")
    return [table.df for table in tables]


# --- MAIN INGESTION LOOP ---
for filename in os.listdir(DATA_DIR):
    fpath = os.path.join(DATA_DIR, filename)
    if filename.endswith(".txt"):
        with open(fpath, "r", encoding="utf-8") as fin:
            text = fin.read()
        chunks = chunk_text(text)
        text_embs = text_model.encode(chunks)
        # Store chunks and embeddings... (write code to ingest into Neo4j/vector DB)
    elif filename.endswith(".pdf"):
        # Example: parse text pages for chunking
        reader = pypdf.PdfReader(fpath)
        all_text = " ".join(
            [page.extract_text() or '' for page in reader.pages])
        chunks = chunk_text(all_text)
        text_embs = text_model.encode(chunks)
        # TABLES
        tables = extract_tables_from_pdf(fpath)
        # Process tables: For now, convert to text summaries if needed
        table_summaries = [df.head().to_string() for df in tables]
        table_embs = text_model.encode(table_summaries)
        # Store tables and embeddings...
    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        image_emb, image_caption = embed_and_caption_image(fpath)
        # Store image_emb and image_caption...
