# MotionVideoSearch

**MotionVideoSearch** is a Python project for extracting motion-based embeddings from videos and storing them in a vector database (FAISS). You can then quickly **search** those embeddings with an input image, retrieving matching segments of the indexed videos. It offers:

- A command-line interface (CLI) built with [Typer](https://typer.tiangolo.com/).
- Integration with PyTorch for embedding extraction.
- Video sampling, embedding, and database storage with SQLite + FAISS.
- Optional watermark detection and removal (via an external repo).
- Utility nodes for [ComfyUI](https://github.com/comfyanonymous/ComfyUI) to create and search motion images within ComfyUI’s node graph environment.

## Features

- **Video Ingestion**  
  Slice videos into small segments, generate motion-based embeddings for each segment, and store them in FAISS for fast similarity search.

- **Search**  
  Query the index with a single motion image to find relevant video segments, returning top matches ranked by distance in vector space.

- **Concurrency Support**  
  Organize indexing jobs in different subdirectories, then merge (combine) them later into one big index.

- **Watermark Removal** (Optional)  
  If you clone [l-comm/WatermarkRemoval](https://github.com/l-comm/WatermarkRemoval.git), you can automatically detect and remove watermarks from video frames during ingestion.

- **ComfyUI Nodes**  
  Includes two custom nodes:
  1. `IG_MotionVideoSearch`: Takes an image input and returns the top 5 ranked URLs from the FAISS index.
  2. `IG_MotionVideoFrame`: Converts 24 consecutive frames into a single “dot frame,” which encodes motion features in a color-coded image.

## ComfyUI
  We have some ComfyUI nodes ready to be used with a pre-existing video database consisting of roughly 100,000 videos. You can get started using the following workflow:
  [MotionSearch](workflows/MotionSearch.json)
  
  ![MotionSearch](https://github.com/user-attachments/assets/f1c1acb5-6c29-415e-9c68-86da0486ed75)

## Installation

### 1. Set Up Environment
Clone the repository:

```bash
git clone https://github.com/IDGallagher/MotionVideoSearch
cd MotionVideoSearch
```

(Optional) Create a conda environment:

```bash
conda create -n mvs python=3.9
conda activate mvs
```

### 2. Install Dependencies
Install [PyTorch and TorchVision](https://pytorch.org/get-started/locally/) compatible with your CUDA setup (example below uses CUDA 11.8):

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Install **ffmpeg 6.x**:

```bash
conda install -c conda-forge ffmpeg=6.*
```

Install other dependencies:

```bash
pip install tqdm matplotlib einops einshape scipy timm lmdb av mediapy typer imageio imageio-ffmpeg requests opencv-python
```

Install **Faiss** (version example below):

```bash
conda install -c pytorch faiss-cpu=1.9.0
```

(Optional) Download the watermark removal repo if you want that functionality:

```bash
git clone https://github.com/l-comm/WatermarkRemoval.git
```

## Usage

After installing, you can run the CLI commands via \`python main.py [COMMAND] [OPTIONS]\`.

### 1. Storing Embeddings
Use the \`store\` command to process videos and store their embeddings in the FAISS index:

```bash
python main.py store \
  --dir /path/to/videos \
  --max-time 1.0 \
  --debug
```
- \`--dir\`: Directory with video files (e.g., \`.mp4\`, \`.avi\`, etc.).
- \`--max-time\`: How many seconds to process from each video (default 1 second).
- \`--debug\`: Enable debug mode to save intermediate frames and videos in a \`debug\` folder.

You can also process a CSV file of URLs:

```bash
python main.py store \
  --csv path/to/videos.csv \
  --max-time 1.0 \
  --start-entry 1
```
- \`--csv\`: CSV with columns including \`contentUrl\`, \`duration\`, and \`name/description\`.
- \`--start-entry\`: Start ingesting from a particular row number in the CSV.

### 2. Searching
Once you have stored some embeddings, you can search with:

```bash
python main.py search \
  --image ./query.jpg \
  --top_k 5
```

This compares the motion image \`query.jpg\` against the FAISS index and returns the top 5 matches with their URLs, metadata, and distance.

### 3. Combining Multiple Indices
If you used the concurrency feature (\`--concurrent-store\`) to generate multiple subindexes, you can combine them all:

```bash
python main.py combine
```
This searches for numbered directories under \`./data\` that each contain \`index.faiss\`, merges them, and writes a final combined \`index.faiss\` in \`./data\`.

## Concurrency Mode
To create multiple indexes in subdirectories, use:
```bash
python main.py store --concurrent-store
```
Each run will create (or use) a new numbered subdirectory under \`./data\`. You can specify a subdirectory with \`--concurrent-index 3\` to store in \`./data/3/\`, for example. Later, use \`combine\` to merge them.

## ComfyUI Integration
If you want to use the motion search in [ComfyUI](https://github.com/comfyanonymous/ComfyUI), simply:
1. Place or symlink this repo in ComfyUI’s \`custom_nodes\` folder.
2. Restart ComfyUI.
3. You should see two new nodes:
   - **IG_MotionVideoSearch**: Takes an image and returns URLs + ranks.
   - **IG_MotionVideoFrame**: Takes a stack of 24 frames and returns a single “dot frame.”

## Contributing
Contributions and suggestions are welcome! Feel free to open issues or pull requests for enhancements or bug fixes.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Please see the [LICENSE file](LICENSE) for details.
