# nodes.py
import os
import torch
import faiss
import logging
import numpy as np

from PIL import Image
from pathlib import Path
from .database import get_connection  # Make sure this is accessible to ComfyUI
from torchvision import transforms

logger = logging.getLogger(__name__)

# Global references to avoid re-loading model/index each time
INDEX_PATH = "data.bin"
EMBEDDING_DIM = 768

_dinov2_vitb14_reg = None
_faiss_index = None

_transform = transforms.Compose([
    transforms.ToTensor(),
])

def load_model_and_index():
    """
    Loads DINOv2 model and the FAISS index (if not already loaded).
    """
    global _dinov2_vitb14_reg, _faiss_index

    # Load the model lazily
    if _dinov2_vitb14_reg is None:
        logger.info("Loading DINOv2 model...")
        _dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _dinov2_vitb14_reg.to(device)
        _dinov2_vitb14_reg.eval()

    # Load or initialize the FAISS index
    if _faiss_index is None:
        if not Path(INDEX_PATH).exists():
            raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Please run the 'store' command first.")
        logger.info(f"Loading FAISS index from {INDEX_PATH}...")
        _faiss_index = faiss.read_index(INDEX_PATH)

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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("search_results",)
    FUNCTION = "search"

    CATEGORY = "Motion Video DB"  # Appears in ComfyUI under this category in the node menu
    @classmethod
    def IS_CHANGED(cls):
        """
        If you want caching behavior or re-run logic, adjust here.
        Returning True means ComfyUI won't try to cache results from previous runs.
        """
        return True

    def search(self, image, top_k):
        """
        Perform the search using the loaded FAISS index and DINOv2 model.
        
        :param image: A ComfyUI image dictionary
        :param top_k: Number of top results to retrieve
        :return: (string,) with the search results
        """
        # 1. Load model and index if needed
        model, index = load_model_and_index()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 2. Convert ComfyUI image (torch tensor) into a PIL Image
        #    By default, ComfyUI images are float16 or float32, shape [C, H, W], range 0..1
        c_img = image["image"]  # This is a torch.Tensor
        # Ensure 0..1 range, convert to uint8, and create PIL
        c_img = c_img.squeeze(0).clamp(0, 1)  # Remove extra batch dim if present
        np_img = (c_img * 255.0).byte().cpu().numpy()
        if np_img.shape[0] == 3:
            # shape: (C, H, W) => (H, W, C)
            np_img = np.transpose(np_img, (1, 2, 0))
        pil_img = Image.fromarray(np_img, mode='RGB')

        # 3. Apply the same resizing logic as your main.py does (multiple of 14, etc.)
        #    We'll replicate that as best we can
        with torch.no_grad():
            # Transform and then adjust dimension to multiple-of-14 if needed
            tensor_img = _transform(pil_img).unsqueeze(0).to(device)  # shape [1, C, H, W]
            _, _, h, w = tensor_img.shape
            new_h = (h // 14) * 14
            new_w = (w // 14) * 14
            h_start = (h - new_h) // 2
            w_start = (w - new_w) // 2
            tensor_img = tensor_img[:, :, h_start : h_start + new_h, w_start : w_start + new_w]

            # 4. Get the embedding
            embedding = model(tensor_img).cpu().numpy().astype("float32")

        # 5. Search in FAISS
        distances, ids = index.search(embedding, top_k)

        # Handle edge cases
        if ids.size == 0 or (ids.size == 1 and ids[0][0] == -1):
            return ("No embeddings found in the FAISS index.",)

        # 6. Retrieve metadata from SQLite
        conn = get_connection()
        cursor = conn.cursor()

        # 7. Format the results as a single string
        results_str = []
        for rank, (dist, uid) in enumerate(zip(distances[0], ids[0]), start=1):
            if uid == -1:
                results_str.append(f"{rank}. [No valid ID] - Distance: {dist:.4f}")
                continue

            cursor.execute(
                """
                SELECT videos.url, videos.description, embeddings.start_time
                FROM embeddings
                JOIN videos ON embeddings.video_id = videos.id
                WHERE embeddings.id = ?
                """,
                (int(uid),),
            )
            row = cursor.fetchone()
            if row:
                url, description, start_time = row
                results_str.append(
                    f"{rank}. URL: {url}, Desc: {description}, Timestamp: {start_time}s, Dist: {dist:.4f}"
                )
            else:
                results_str.append(f"{rank}. [Missing DB row for ID {uid}], Dist: {dist:.4f}")

        conn.close()

        # Combine all lines into one output
        final_text = "\n".join(results_str)
        return (final_text,)