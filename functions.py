# functions.py
import gc
import logging
import os
from pathlib import Path
import tempfile
import imageio

import numpy as np
import requests
import torch
import torchaudio
from torchvision import transforms
import torch.nn.functional as F
import torchvision.utils as vutils

from dot_functions import get_dot_frame
from database import (
    add_video,
    add_embedding,
    update_video_saved_up_to,
    update_video_duration
)

logger = logging.getLogger(__name__)

# Try to import the watermark removal utilities. If they don't exist,
# just set a flag to False so we can skip calls to them later.
try:
    from WatermarkRemoval.find_watermarks import find_watermark_tensor, remove_watermark_batch
    WATERMARK_REMOVAL_AVAILABLE = True
except ImportError:
    WATERMARK_REMOVAL_AVAILABLE = False

def save_pixel_values_as_video(pixel_values, output_path):
    """
    Save pixel values as a video file with a target FPS of 24.

    Parameters:
    pixel_values (Tensor): Tensor containing pixel values with shape (N, C, H, W),
                           where N is the number of frames, C is the number of channels,
                           H is the height, and W is the width.
    output_path (str): The output path for the video file.
    """
    target_fps = 24
    writer = imageio.get_writer(output_path, fps=target_fps)

    try:
        for frame in pixel_values.cpu().numpy():
            # Convert the frame from (C, H, W) to (H, W, C)
            frame = np.transpose(frame, (1, 2, 0))
            writer.append_data((frame * 255).astype(np.uint8))
    finally:
        writer.close()

    logger.info(f"Video saved as {output_path}")

def store_chunk(video, start_time, end_time, video_metadata, embedding_model, index):
    """
    Processes a video chunk, extracts and stores its embedding, and updates the FAISS index and SQLite database.
    """
    frame = get_dot_frame(video)
    # Pad to make h and w divisible by 14
    c, h, w = frame.shape
    new_h = (h // 14) * 14  # Floor division to get nearest smaller multiple of 14
    new_w = (w // 14) * 14
    h_start = (h - new_h) // 2
    w_start = (w - new_w) // 2
    frame = frame[:, h_start:h_start+new_h, w_start:w_start+new_w]

    logger.info(f"Processing video URL {video_metadata['url']} at {start_time}s with frame shape {frame.shape}")

    # Generate embedding
    with torch.no_grad():
        embeddings = embedding_model(frame.unsqueeze(0))
        embedding = embeddings[0].cpu().numpy().astype('float32')  # Ensure dtype is float32 for FAISS

    log_gpu_memory("after embedding")

    logger.debug(f"Generated embedding with shape {embedding.shape}")

    # Save the processed frame image for reference
    path = Path(video_metadata['url'])
    output_image_path = Path('debug') / f'track_{path.stem}_{start_time}.png'
    output_image_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    vutils.save_image(frame, str(output_image_path))
    logger.debug(f"Saved processed frame to {output_image_path}")

    # Insert embedding into FAISS index and SQLite
    video_id = video_metadata['db_id']
    embedding_db_id = add_embedding(video_id, start_time)

    # Add the new embedding to the FAISS index with the unique ID
    index.add_with_ids(embedding.reshape(1, -1), np.array([embedding_db_id], dtype='int64'))
    logger.info(f"Added embedding for video ID {video_id} at {start_time}s to FAISS index with embedding ID {embedding_db_id}")

    # Update the video's saved_up_to time
    if end_time > video_metadata.get('saved_up_to', 0.0):
        update_video_saved_up_to(video_id, end_time)

    # Explicitly delete tensors to free GPU memory
    del frame, embeddings, embedding
    torch.cuda.empty_cache()
    log_gpu_memory("after deletion")

def log_gpu_memory(stage=""):
    """
    Logs the current GPU memory usage.

    Args:
        stage (str): Description of the current stage for logging purposes.
    """
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MB
        logger.info(f"GPU Memory {stage}: Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
    else:
        logger.info("CUDA is not available.")

def store_video(video_metadata, embedding_model, index, max_time=1.0):
    """
    Processes and stores video embeddings up to a specified maximum time.
    """
    url = video_metadata['url']
    duration_seconds = video_metadata.get('duration_seconds')
    description = video_metadata.get('description')

    # Add or retrieve video metadata from the database
    db_video_metadata = add_video(url, duration_seconds, description)
    
    # Update the video_metadata dictionary with database values
    video_metadata.update({
        'db_id': db_video_metadata['id'],
        'duration_seconds': db_video_metadata['duration'],
        'description': db_video_metadata.get('description'),
        'saved_up_to': db_video_metadata.get('saved_up_to', 0.0)
    })

    fit_to = 336
    sample_n_frames = 24
    target_fps = 24

    if video_metadata['saved_up_to'] >= max_time:
        logger.info("Video already saved")
        return False

    stream_reader = None
    frames = None
    temp_file_path = None  # Initialize to handle scope in finally
    try:
        response = requests.get(url, stream=True)
        if response.status_code != 200:
            raise ValueError(f"Failed to download video from {url}")

        temp_file_path = os.path.join(tempfile.gettempdir(), os.path.basename(url))
        with open(temp_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        video_path = temp_file_path

        logger.info(f"Initializing StreamReader for video URL: {video_path}")
        stream_reader = torchaudio.io.StreamReader(video_path)
        stream_reader.add_video_stream(
            frames_per_chunk=sample_n_frames,
            filter_desc=(
                f"fps={target_fps},"
                f"scale='if(gt(iw,ih),{fit_to}*iw/ih,{fit_to}):if(gt(iw,ih),{fit_to},{fit_to}*ih/iw)',"
                "crop=w=floor(iw/8)*8:h=floor(ih/8)*8:x=(iw-floor(iw/8)*8)/2:y=(ih-floor(ih/8)*8)/2,"
                "format=pix_fmts=rgb24"
            )
        )

        metadata = stream_reader.get_src_stream_info(0)
        logger.info(f"metadata {metadata}")
        original_fps = metadata.frame_rate
        video_length = metadata.num_frames

        if video_length == 0:
            raise ValueError("Empty video file")

        duration = video_length / original_fps
        logger.info(f"Video duration: {duration}s")

        # Update the duration in the database if it has changed
        if duration != video_metadata['duration_seconds']:
            update_video_duration(db_video_metadata['id'], duration)
            video_metadata['duration_seconds'] = duration

        sampling_interval = 1.0
        # Use the configurable max_time instead of hardcoded 1.0
        max_time_to_process = min(max_time, duration)

        for start_time in np.arange(0, max_time_to_process, sampling_interval):
            logger.info(f"saved up to {video_metadata.get('saved_up_to', 0.0)}")
            if start_time < video_metadata.get('saved_up_to', 0.0):
                continue

            stream_reader.seek(start_time)
            stream_reader.fill_buffer()
            (frames,) = stream_reader.pop_chunks()
            pixel_values = frames.float() / 255.0

            if WATERMARK_REMOVAL_AVAILABLE:
                res = find_watermark_tensor(pixel_values)
                if not res:
                    continue
                final_x, final_y, _ = res
                pixel_values = remove_watermark_batch(pixel_values, final_x, final_y)
            
            # Save the processed video chunk
            output_video_path = Path('debug') / f'test_vid_{db_video_metadata["id"]}_{start_time}.mp4'
            save_pixel_values_as_video(pixel_values, output_video_path)
            logger.info(f"Saved pixel values as video: {output_video_path}")
            
            # Store the embedding chunk
            end_time = start_time + sampling_interval
            store_chunk(pixel_values, start_time, end_time, video_metadata, embedding_model, index)

            # Explicitly delete frames and pixel_values to free GPU memory
            del frames, pixel_values
            frames = None
            torch.cuda.empty_cache()
            log_gpu_memory(f"after processing chunk at {start_time}s")

    except KeyError as e:
        logger.error(f"Missing key in video metadata: {e}")
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        logger.error(f"Error processing video {url}: {e}")
    finally:
        if stream_reader is not None:
            stream_reader.remove_stream(0)
            del stream_reader

        if frames is not None:
            del frames

        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

        gc.collect()
        torch.cuda.empty_cache()
        log_gpu_memory("after final cleanup")
    return True