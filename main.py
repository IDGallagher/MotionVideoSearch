# main.py
import torch
import typer
import logging
import faiss
from rich.logging import RichHandler
from rich.console import Console
from pathlib import Path
from typing import Annotated
from functions import store_video
from PIL import Image
import torchvision.transforms as transforms
import csv
import torch.nn.functional as F

from database import initialize_db, get_connection  # Import updated database functions

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
INDEX_PATH = DATA_DIR / "data.bin"

logger.info(f"Index path {INDEX_PATH}")

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
):
    """
    Store videos from a directory and/or a CSV file into the vector DB.
    """
    # Validate start_entry
    if start_entry < 1:
        logger.error("start_entry must be a positive integer starting from 1.")
        raise typer.Exit(code=1)

    # Initialize the database
    initialize_db()
    conn = get_connection()
    conn.close()

    # Embedding model
    dinov2_vitb14_reg = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    dinov2_vitb14_reg.to(device)
    dinov2_vitb14_reg.eval()  # Set model to evaluation mode

    # Load or initialize FAISS index
    if Path(INDEX_PATH).exists():
        logger.info("Loading existing FAISS index...")
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
                    store_video(video_metadata, dinov2_vitb14_reg, index, max_time)
                    # Save updated FAISS index
                    faiss.write_index(index, INDEX_PATH)
        logger.info("Storage complete. FAISS index saved.")

    # Process videos from the CSV file
    elif csv_file:
        logger.info(f"Processing videos from CSV file: {csv_file}")

        total_entries = 0
        processed_entries = 0

        with open(csv_file, 'r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            # Convert reader to list to count total entries (optional)
            rows = list(reader)
            total_entries = len(rows)
            if start_entry > total_entries:
                logger.error(f"start_entry {start_entry} exceeds the total number of CSV entries ({total_entries}).")
                raise typer.Exit(code=1)

            # Determine the range of rows to process
            rows_to_process = rows[start_entry - 1:]  # 1-based index

            logger.info(f"Starting processing from entry {start_entry} out of {total_entries} total entries.")

            for row_number, row in enumerate(rows_to_process, start=start_entry):
                try:
                    duration_seconds = parse_duration(row['duration'])
                except Exception as e:
                    logger.warning(f"Failed to parse duration for video {row.get('url') or row.get('contentUrl')}: {e}")
                    duration_seconds = 0.0

                video_metadata = {
                    'url': row.get('contentUrl') or row.get('url'),  # Handle different possible keys
                    'duration_seconds': duration_seconds,
                    'description': row.get('name') or row.get('description'),  # Handle different possible keys
                    'db_id': None,  # Will be set in store_video
                    'saved_up_to': 0.0,
                }

                success = store_video(video_metadata, dinov2_vitb14_reg, index, max_time)
                if success:
                    processed_entries += 1
                    # Save updated FAISS index periodically or after each successful insertion
                    faiss.write_index(index, str(INDEX_PATH))

                # Optional: Log progress every N entries
                if processed_entries % 100 == 0:
                    logger.info(f"Processed {processed_entries} entries from CSV.")

        logger.info(f"Storage complete. {processed_entries} entries processed and FAISS index saved.")

    else:
       logger.error("Specify dir or csv!")



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