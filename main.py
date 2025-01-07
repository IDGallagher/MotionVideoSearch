# main.py
import os
import torch
import typer
import logging
import faiss
from rich.logging import RichHandler
from rich.console import Console
from pathlib import Path
from typing import Annotated, Optional
from functions import store_video, download_checkpoints
from PIL import Image
import torchvision.transforms as transforms
import csv
import torch.nn.functional as F

from database import initialize_db, get_connection, set_db_path 

cli: typer.Typer = typer.Typer(
    context_settings=dict(help_option_names=["-h", "--help"]),
    rich_markup_mode="rich",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)

console = Console(highlight=True)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        RichHandler(
            console=console,
            rich_tracebacks=True,
            omit_repeated_times=False,
        ),
    ],
    force=True,
)
logger = logging.getLogger(__name__)

transform = transforms.Compose([
    transforms.ToTensor(),
])

# FAISS index
EMBEDDING_DIM = 768

# Paths
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_PATH = DATA_DIR / "index.faiss"

logger.info(f"Index path {INDEX_PATH}")

def get_next_available_directory(base_dir: Path) -> Path:
    """
    Returns a new subdirectory named with the next integer, starting from 0.
    If 0 already exists, it tries 1, etc.
    """
    existing_indices = []
    for child in base_dir.iterdir():
        if child.is_dir() and child.name.isdigit():
            existing_indices.append(int(child.name))
    if not existing_indices:
        next_index = 0
    else:
        next_index = max(existing_indices) + 1

    new_subdir = base_dir / str(next_index)
    new_subdir.mkdir(parents=True, exist_ok=True)
    return new_subdir

@cli.command()
def store(
    dir: Annotated[
            Path,
            typer.Option(
                "--dir",
                "-d",
                path_type=Path,
                exists=True,
                readable=True,
                dir_okay=True,
                help="Video directory for testing",
            ),
    ] = None,
    csv_file: Annotated[
        Path,
        typer.Option(
            "--csv",
            "-c",
            path_type=Path,
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            help="Path to the CSV file containing video metadata",
        ),
    ] = None,
    max_time: Annotated[
        float,
        typer.Option(
            "--max-time",
            "-m",
            help="Maximum time (in seconds) to process each video for chunking",
        ),
    ] = 1.0,
    start_entry: Annotated[
        int,
        typer.Option(
            "--start-entry",
            "-s",
            show_default=True,
            help="The CSV entry number to start processing from (1-based index)",
        ),
    ] = 1,  # Added start_entry option
    concurrent_store: Annotated[
        bool,
        typer.Option(
            "--concurrent-store",
            "-cs",
            help="If passed, store data.sqlite and index.faiss in a numbered subdirectory under ./data",
        ),
    ] = False,
    concurrent_index: Annotated[
        Optional[int],
        typer.Option(
            "--concurrent-index",
            "-ci",
            help="Integer subdirectory for concurrent mode. If not specified, a new one will be created automatically.",
        ),
    ] = None,
    debug: Annotated[
        bool,
        typer.Option(
            "--debug",
            help="Enable debug mode to save frames and videos to the debug folder.",
        ),
    ] = False,
):
    """
    Store videos from a directory and/or a CSV file into the vector DB.
    """

    download_checkpoints()

    # 1. Handle concurrency: Decide if we override DB path and index path
    global INDEX_PATH  # We'll reassign if needed

    if debug:
        if not os.path.exists('debug'):
            os.makedirs('debug')
        logger.info("Debug mode is ON. Images and videos will be saved in the debug folder.")
    else:
        logger.info("Debug mode is OFF. Skipping saving debug frames and videos.")

    if concurrent_store:
        logger.info("[CONCURRENT] concurrent_store mode is ON.")

        if concurrent_index is not None:
            # Subdirectory based on user request
            subdir = DATA_DIR / str(concurrent_index)
            subdir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CONCURRENT] Using user-specified subdir -> {subdir}")
        else:
            # Automatically find next subdirectory
            subdir = get_next_available_directory(DATA_DIR)
            logger.info(f"[CONCURRENT] Created new subdir -> {subdir}")

        # Now override paths in memory
        # new_db_path = subdir / "data.sqlite"
        new_index_path = subdir / "index.faiss"

        # set_db_path(new_db_path)  # Override EMBEDDINGS_DB_PATH
        INDEX_PATH = new_index_path  # We'll rely on the global index path variable

        # logger.info(f"[CONCURRENT] Overriding EMBEDDINGS_DB_PATH -> {new_db_path}")
        logger.info(f"[CONCURRENT] Overriding INDEX_PATH -> {new_index_path}")
    else:
        logger.info("Concurrent store mode not activated. Using default data/ directory.")

    # 2. Validate start_entry
    if start_entry < 1:
        logger.error("start_entry must be a positive integer starting from 1.")
        raise typer.Exit(code=1)

    # 3. Initialize the (potentially new) database
    initialize_db()
    conn = get_connection()
    conn.close()

    # 4. Load embedding model
    dinov2_vitb14_reg = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14_reg")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dinov2_vitb14_reg.to(device)
    dinov2_vitb14_reg.eval()

    # 5. Load or initialize FAISS index from the correct path
    if INDEX_PATH.exists():
        logger.info(f"Loading existing FAISS index from '{INDEX_PATH}'...")
        index = faiss.read_index(str(INDEX_PATH))
    else:
        logger.info("Initializing new FAISS index...")
        base_index = faiss.IndexFlatL2(EMBEDDING_DIM)
        index = faiss.IndexIDMap(base_index)

    # Process videos from the directory
    if dir:
        logger.info(f"Processing videos in directory: {dir}")
        video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
        with torch.no_grad():
            for video_path in dir.glob('*'):
                if video_path.suffix.lower() in video_extensions:
                    # Prepare video metadata for local videos
                    video_metadata = {
                        'url': str(video_path.resolve()),  # Use absolute path
                        'duration_seconds': 0.0,  # Placeholder or implement duration extraction
                        'description': video_path.name,
                        'db_id': None,  # Will be set in store_video
                        'saved_up_to': 0.0,
                    }
                    store_video(video_metadata, dinov2_vitb14_reg, index, max_time, debug=debug)
                    # Save updated FAISS index
                    faiss.write_index(index, INDEX_PATH)
        logger.info("Storage complete. FAISS index saved.")

    # Process videos from the CSV file
    elif csv_file:
        logger.info(f"Processing videos from CSV file: {csv_file}")

        processed_entries = 0

        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            # We keep track of the current row_number ourselves
            row_number = 0

            for row in reader:
                row_number += 1

                # If row_number is before our start_entry, skip it
                if row_number < start_entry:
                    continue
                
                logger.info(f"Procesing row {row_number}")

                # From here on, we process the row
                try:
                    duration_seconds = parse_duration(row.get('duration', 'PT0S'))
                except Exception as e:
                    logger.warning(f"Failed to parse duration for video {row.get('url') or row.get('contentUrl')}: {e}")
                    duration_seconds = 0.0

                # Build the video metadata
                video_metadata = {
                    'url': row.get('contentUrl') or row.get('url'),
                    'duration_seconds': duration_seconds,
                    'description': row.get('name') or row.get('description'),
                    'db_id': None,  # Will be set in store_video
                    'row': row_number,
                    'saved_up_to': 0.0,
                }

                success = store_video(video_metadata, dinov2_vitb14_reg, index, max_time, debug=debug)
                if success:
                    processed_entries += 1
                    # Save updated FAISS index periodically or after each successful insertion
                    faiss.write_index(index, str(INDEX_PATH))

                # Optional: if you need to stop after a certain number of lines,
                # you can break here or you can keep going until the file ends.

        logger.info(f"Storage complete. {processed_entries} entries processed and FAISS index saved.")


    else:
        logger.error("Specify dir or csv!")
        raise typer.Exit(code=1)

@cli.command()
def combine():
    """
    Combine multiple FAISS indexes from numbered subdirectories under ./data
    into one index.faiss in the main ./data folder.
    """
    # 1) Gather subdirectories that contain "index.faiss"
    subdirs = []
    for item in DATA_DIR.iterdir():
        if item.is_dir() and item.name.isdigit():
            sub_index = item / "index.faiss"
            if sub_index.exists():
                subdirs.append(sub_index)

    if not subdirs:
        logger.error("No subdirectory indexes found. Nothing to combine.")
        raise typer.Exit(code=1)

    logger.info(f"Found {len(subdirs)} indexes to combine: {subdirs}")

    # 2) Read the first index to serve as a "base"
    combined_index = faiss.read_index(str(subdirs[0]))
    logger.info(f"Loaded first index from: {subdirs[0]}")

    # 3) For the remaining sub-indexes, merge them into the base
    for sub_index_path in subdirs[1:]:
        idx = faiss.read_index(str(sub_index_path))
        combined_index.merge_from(idx)
        logger.info(f"Merged index from {sub_index_path}")

    # 4) Write out the resulting combined index
    faiss.write_index(combined_index, str(INDEX_PATH))
    logger.info(f"Combined index saved to: {INDEX_PATH}")


@cli.command()
def search(
    image_path: Annotated[
        Path,
        typer.Option(
            "--image",
            "-i",
            exists=True,
            readable=True,
            file_okay=True,
            dir_okay=False,
            help="Path to the motion image for searching",
        ),
    ] = Path("./query.jpg"),
    top_k: Annotated[
        int,
        typer.Option(
            "--top_k",
            "-k",
            help="Number of top results to return",
        ),
    ] = 5,
):
    """
    Search the index for videos similar to the given motion image.
    """
    # Ensure the database and FAISS index exist
    if not Path(INDEX_PATH).exists():
        logger.error("FAISS index not found. Please run the 'store' command first.")
        raise typer.Exit(code=1)

    # Load FAISS index
    logger.info(f"Loading FAISS index from '{INDEX_PATH}'...")
    try:
        index = faiss.read_index(str(INDEX_PATH))
    except Exception as e:
        logger.error(f"Failed to load FAISS index: {e}")
        raise typer.Exit(code=1)

    # Load embedding model
    logger.info(f"Loading embedding model...")
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vitb14_reg.to(device)
    dinov2_vitb14_reg.eval()

    # Process the input image
    logger.info(f"Processing the input image: {image_path}")
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise typer.Exit(code=1)

    image_tensor = transform(image).unsqueeze(0).to(device)
    b, c, h, w = image_tensor.shape

    new_h = (h // 14) * 14  # Floor division to get nearest smaller multiple of 14
    new_w = (w // 14) * 14
    h_start = (h - new_h) // 2
    w_start = (w - new_w) // 2
    image_tensor = image_tensor[:, :, h_start:h_start+new_h, w_start:w_start+new_w]

    with torch.no_grad():
        embedding = dinov2_vitb14_reg(image_tensor).cpu().numpy().astype('float32')

    # Search FAISS index
    logger.info(f"Searching for the top {top_k} similar videos...")
    distances, ids = index.search(embedding, top_k)

    logger.info(f"Retrieved IDs: {ids}")

    if ids.size == 0 or (ids.size == 1 and ids[0][0] == -1):
        logger.error("No embeddings found in the FAISS index.")
        raise typer.Exit(code=1)

    # Retrieve metadata from SQLite based on FAISS IDs
    conn = get_connection()
    cursor = conn.cursor()
    logger.info("Search Results:")
    for rank, (dist, uid) in enumerate(zip(distances[0], ids[0]), start=1):
        if uid == -1:
            console.print(f"[bold green]{rank}.[/bold green] Unknown video - Distance: {dist:.4f}")
            continue
        cursor.execute("""
            SELECT videos.url, videos.description, embeddings.start_time 
            FROM embeddings 
            JOIN videos ON embeddings.video_id = videos.id 
            WHERE embeddings.id = ?
        """, (int(uid),))
        result = cursor.fetchone()
        if result:
            url, description, start_time = result
            console.print(
                f"[bold green]{rank}.[/bold green] "
                f"URL: {url}, "
                f"Description: {description}, "
                f"Timestamp: {start_time}s - Distance: {dist:.4f}"
            )
        else:
            console.print(f"[bold green]{rank}.[/bold green] Unknown video ID {uid} - Distance: {dist:.4f}")
    conn.close()

def parse_duration(duration_str):
    """
    Parses an ISO 8601 duration string and returns the duration in seconds.

    Args:
        duration_str (str): Duration string (e.g., "PT00H00M30S")

    Returns:
        float: Duration in seconds
    """
    import re
    pattern = re.compile(
        r'PT'
        r'(?:(?P<hours>\d+)H)?'
        r'(?:(?P<minutes>\d+)M)?'
        r'(?:(?P<seconds>\d+)S)?'
    )
    match = pattern.match(duration_str)
    if not match:
        logger.warning(f"Unable to parse duration string: {duration_str}")
        return 0.0
    hours = int(match.group('hours') or 0)
    minutes = int(match.group('minutes') or 0)
    seconds = int(match.group('seconds') or 0)
    return hours * 3600 + minutes * 60 + seconds

@cli.command()
def test():
    logger.info("TEST")

if __name__ == "__main__":
    cli()
    logger.info("Done.")