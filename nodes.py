# nodes.py
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import faiss
import logging
import numpy as np
import requests
from tqdm import tqdm
from PIL import Image
from pathlib import Path
import torch.nn.functional as F

# Make sure this import can find the database.py in your package
from .database import get_connection  
from torchvision import transforms

# Import your get_dot_frame function
from .dot_functions import get_dot_frame
from .functions import download_checkpoints

logger = logging.getLogger(__name__)

# Define URLs for data.bin and embeddings.db on HuggingFace
DATA_BIN_URL = "https://huggingface.co/iggy101/MotionVideoSearch/resolve/main/index.faiss"
EMBEDDINGS_DB_URL = "https://huggingface.co/iggy101/MotionVideoSearch/resolve/main/data.sqlite"

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
    returns 5 ranked search results from the FAISS index based on a given starting rank.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),  # ComfyUI "IMAGE" type
                "starting_rank": ("INT", {"default": 1, "min": 1, "max": 9999999, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("url_1", "url_2", "url_3", "url_4", "url_5", "status")
    FUNCTION = "search"

    CATEGORY = "🐓 IG Motion Search Nodes"

    def search(self, image, starting_rank):
        """
        Perform the search using the loaded FAISS index and DINOv2 model.

        :param image: A torch.Tensor, shape [batch_size, C, H, W]
        :param starting_rank: The starting rank for the search results (defaults to 1).
                              The node will return this rank plus the next 4 lower-ranked results.
        :return: 5 separate URLs for the search results and a status string with scores
        """
        logger.debug(f"Image type: {type(image)}")
        logger.debug(f"Image shape: {image.shape}")
        logger.debug(f"Image dtype: {image.dtype}")
        logger.debug(f"Starting rank: {starting_rank}")

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
                c_img = c_img.squeeze(0)
        elif c_img.ndim != 3:
            raise ValueError(f"Expected image tensor to have 3 or 4 dimensions, got {c_img.ndim}")

        # Ensure values are in [0, 1] range
        c_img = c_img.clamp(0, 1)

        # Convert to uint8 and numpy array
        np_img = (c_img * 255.0).byte().cpu().numpy()
        # Convert to PIL image
        pil_img = Image.fromarray(np_img, mode='RGB')

        # 3. Apply the same resizing logic if desired
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

        # We want top_k = starting_rank + 4
        top_k = starting_rank + 4
        distances, ids = index.search(embedding, top_k)

        if ids.size == 0 or (ids.size == 1 and ids[0][0] == -1):
            return (
                "No embeddings found in the FAISS index.", 
                "", 
                "", 
                "", 
                "", 
                "No scores available."
            )

        # Slice out the 5 results we actually want
        selected_distances = distances[0][starting_rank - 1 : starting_rank - 1 + 5]
        selected_ids = ids[0][starting_rank - 1 : starting_rank - 1 + 5]

        # 6. Retrieve metadata from SQLite
        conn = get_connection()
        cursor = conn.cursor()

        urls = [""] * 5
        results_str = []

        for offset, (dist, uid) in enumerate(zip(selected_distances, selected_ids)):
            rank = starting_rank + offset
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
                urls[offset] = row[0]
                results_str.append(f"{rank}. Distance: {dist:.4f}")
            else:
                results_str.append(f"{rank}. [Missing DB row for ID {uid}], Distance: {dist:.4f}")

        conn.close()
        status = "\n".join(results_str)

        return (*urls, status)


###############################################################################
# New Node: IG_MotionVideoDotFrame
###############################################################################
class IG_MotionVideoFrame:
    """
    A ComfyUI node that takes a batch of frames from a video, ensures we have at least
    24 frames, trims (cuts off) the video to exactly 24 frames, then calls get_dot_frame
    to produce a colorized motion image. Finally, it returns that as a ComfyUI image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("colored_motion_image",)
    FUNCTION = "apply"
    CATEGORY = "🐓 IG Motion Search Nodes"

    def apply(self, video_frames):
        """
        :param video_frames: A 4D torch.Tensor of shape [N, C, H, W], with N >= 24
        :return: A single colorized motion image (torch.Tensor of shape [1, C, H, W])
        """
        
        if not isinstance(video_frames, torch.Tensor):
            raise TypeError("video_frames must be a torch.Tensor.")

        if video_frames.ndim != 4:
            raise ValueError(
                f"Expected a 4D tensor [N, C, H, W], but got shape {video_frames.shape}."
            )
        
        # AI is convinced that Comfy images are B, C, H, W but they're actually B, H, W, C
        video_frames = video_frames.permute(0, 3, 1, 2)
        
        num_frames, channels, height, width = video_frames.shape
        if num_frames < 24:
            raise ValueError(
                f"Video must have at least 24 frames, but got {num_frames}."
            )

        # 1) Trim the video to 24 frames
        video_frames = video_frames[:24]  # shape: [24, C, H, W]

        # 2) Scale so the shorter side is 'fit_to':
        fit_to = 336
        _, _, H, W = video_frames.shape
        
        # If width < height, we set width to 336 and scale height accordingly.
        # Otherwise, we set height to 336 and scale width accordingly.
        if W < H:
            new_W = fit_to
            new_H = int(round(H * fit_to / W))
        else:
            new_H = fit_to
            new_W = int(round(W * fit_to / H))

        # Interpolate all frames to this new size
        video_frames = F.interpolate(
            video_frames,
            size=(new_H, new_W),
            mode='bilinear',
            align_corners=False
        )

        # 3) Crop so the height and width are multiples of 8
        new_H_aligned = (new_H // 8) * 8
        new_W_aligned = (new_W // 8) * 8

        h_start = (new_H - new_H_aligned) // 2
        w_start = (new_W - new_W_aligned) // 2

        video_frames = video_frames[
            :,
            :,
            h_start : h_start + new_H_aligned,
            w_start : w_start + new_W_aligned
        ]

        # get_dot_frame typically expects the shape [N, C, H, W].
        # This function returns a single frame of shape [C, H, W].
        download_checkpoints()
        frame = get_dot_frame(video_frames)

        # Convert to ComfyUI's "IMAGE" format: [1, H, W, C]
        frame = frame.unsqueeze(0)  # shape = [1, C, H, W]
        frame = frame.permute(0, 2, 3, 1)

        return (frame,)
