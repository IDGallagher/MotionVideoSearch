# nodes.py
import os
import torch
import faiss
import logging
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from .database import get_connection  # Ensure accessible to ComfyUI
from torchvision import transforms

logger = logging.getLogger(__name__)

# Define URLs for data.bin and embeddings.db on HuggingFace
DATA_BIN_URL = "https://huggingface.co/iggy101/MotionVideoSearch/raw/main/index.faiss"
EMBEDDINGS_DB_URL = "https://huggingface.co/iggy101/MotionVideoSearch/raw/main/data.sqlite"

# Directory to store downloaded files
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

INDEX_PATH = DATA_DIR / "index.faiss"
EMBEDDINGS_DB_PATH = DATA_DIR / "data.sqlite"
EMBEDDING_DIM = 768

_dinov2_vitb14_reg = None
_faiss_index = None

_transform = transforms.Compose([
    transforms.ToTensor(),
])

def download_file(url, dest_path):
    """
    Downloads a file from the specified URL to the destination path with a progress bar.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(dest_path, 'wb') as f, tqdm(
            desc=f"Downloading {dest_path.name}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
        logger.info(f"Downloaded {dest_path.name} successfully.")
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        raise

def ensure_file_exists(file_path, url):
    """
    Ensures that the file exists locally, and uses ETag to detect if it needs updating.
    If ETag indicates the remote file changed, download the new version.
    """
    local_etag_path = file_path.with_suffix(file_path.suffix + ".etag")

    # Attempt to retrieve the remote ETag
    remote_etag = None
    try:
        head_resp = requests.head(url, allow_redirects=True)
        head_resp.raise_for_status()
        remote_etag = head_resp.headers.get("ETag")
    except requests.exceptions.RequestException as e:
        logger.warning(f"Failed to retrieve ETag for {url}: {e}")

    # If the file does not exist locally, always download
    if not file_path.exists():
        logger.info(f"{file_path.name} not found. Downloading from HuggingFace...")
        download_file(url, file_path)
        if remote_etag:
            local_etag_path.write_text(remote_etag)
        return

    # If we have a remote ETag, compare to local ETag
    if remote_etag:
        if local_etag_path.exists():
            local_etag = local_etag_path.read_text().strip()
            # If they match, do nothing
            if local_etag == remote_etag:
                logger.info(f"{file_path.name} is already up to date (ETag match).")
                return
        # Otherwise, re-download
        logger.info(f"{file_path.name} is outdated. Downloading new version from HuggingFace...")
        download_file(url, file_path)
        local_etag_path.write_text(remote_etag)
    else:
        # If no remote ETag is available, default to the old logic
        logger.info(f"{file_path.name} already exists (no ETag to compare).")

def load_model_and_index():
    """
    Loads DINOv2 model and the FAISS index (if not already loaded).
    Downloads required files if they are missing or outdated.
    """
    global _dinov2_vitb14_reg, _faiss_index

    # Ensure that data.bin and embeddings.db are present/updated
    ensure_file_exists(INDEX_PATH, DATA_BIN_URL)
    ensure_file_exists(EMBEDDINGS_DB_PATH, EMBEDDINGS_DB_URL)

    # Load the model lazily
    if _dinov2_vitb14_reg is None:
        logger.info("Loading DINOv2 model...")
        _dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dinov2_vitb14_reg.to(device)
        _dinov2_vitb14_reg.eval()

    # Load or initialize the FAISS index
    if _faiss_index is None:
        if not INDEX_PATH.exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run the 'store' command first.")
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        _faiss_index = faiss.read_index(str(INDEX_PATH))

    return _dinov2_vitb14_reg, _faiss_index

class IG_MotionVideoSearch:
    """
    A ComfyUI node that accepts a ComfyUI image and
    returns the top 5 search results from the FAISS index.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI "IMAGE" type
                "top_k": ("INT", {"default": 5, "min": 1, "max": 20, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("url_1", "url_2", "url_3", "url_4", "url_5", "status")
    FUNCTION = "search"

    CATEGORY = "Motion Video DB"  # Appears in ComfyUI under this category in the node menu

    def search(self, image, top_k):
        """
        Perform the search using the loaded FAISS index and DINOv2 model.

        :param image: A torch.Tensor, shape [batch_size, C, H, W]
        :param top_k: Number of top results to retrieve
        :return: 5 separate URLs for the search results and a status string with scores
        """
        # Log input details for debugging
        logger.debug(f"Image type: {type(image)}")
        logger.debug(f"Image shape: {image.shape}")
        logger.debug(f"Image dtype: {image.dtype}")

        # 1. Load model and index if needed
        model, index = load_model_and_index()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Convert ComfyUI image (torch tensor) into a PIL Image
        c_img = image
        if c_img.ndim == 4:
            # Assume shape is [batch_size, C, H, W]
            if c_img.size(0) > 1:
                logger.warning("Received batch size > 1. Only processing the first image in the batch.")
                c_img = c_img[0]
            else:
                c_img = c_img.squeeze(0)  # Remove the batch dimension
        elif c_img.ndim != 3:
            raise ValueError(f"Expected image tensor to have 3 or 4 dimensions, got {c_img.ndim}")

        # Ensure values are in [0, 1] range
        c_img = c_img.clamp(0, 1)

        # Convert to uint8 and numpy array
        np_img = (c_img * 255.0).byte().cpu().numpy()
        # Convert to PIL image
        pil_img = Image.fromarray(np_img, mode='RGB')

        # 3. Apply the same resizing logic as your main.py does (multiple of 14, etc.)
        with torch.no_grad():
            tensor_img = _transform(pil_img).unsqueeze(0).to(device)  # shape [1, C, H, W]
            _, _, h, w = tensor_img.shape
            new_h = (h // 14) * 14
            new_w = (w // 14) * 14
            h_start = (h - new_h) // 2
            w_start = (w - new_w) // 2
            tensor_img = tensor_img[:, :, h_start: h_start + new_h, w_start: w_start + new_w]

            # 4. Get the embedding
            embedding = model(tensor_img).cpu().numpy().astype("float32")

        # 5. Search in FAISS
        distances, ids = index.search(embedding, top_k)

        # Handle edge cases
        if ids.size == 0 or (ids.size == 1 and ids[0][0] == -1):
            return ("No embeddings found in the FAISS index.", "", "", "", "", "No scores available.")

        # 6. Retrieve metadata from SQLite
        conn = get_connection()
        cursor = conn.cursor()

        urls = [""] * 5  # Initialize list of 5 URL strings
        results_str = []

        for rank, (dist, uid) in enumerate(zip(distances[0], ids[0]), start=1):
            if rank > 5:
                break
            if uid == -1:
                results_str.append(f"{rank}. [No valid ID] - Distance: {dist:.4f}")
                continue

            cursor.execute(
                """
                SELECT videos.url
                FROM embeddings
                JOIN videos ON embeddings.video_id = videos.id
                WHERE embeddings.id = ?
                """,
                (int(uid),),
            )
            row = cursor.fetchone()
            if row:
                urls[rank - 1] = row[0]
                results_str.append(f"{rank}. Distance: {dist:.4f}")
            else:
                results_str.append(f"{rank}. [Missing DB row for ID {uid}], Distance: {dist:.4f}")

        conn.close()

        status = "\n".join(results_str)
        return (*urls, status)
