# SD-XS Local Image Generator

A minimal web application for generating images locally using Stable Diffusion XS models from HuggingFace.

## Features

- 🚀 Download and load SD-XS models directly from HuggingFace
- 🎨 Generate images locally using native TensorFlow/PyTorch
- 💾 Save generated images to local storage
- 🖥️ Clean, minimal web interface
- ⚡ Fast inference with optimized settings

## Tech Stack

- **Backend**: FastAPI (Python) with PyTorch + Diffusers
- **Frontend**: React with Tailwind CSS + shadcn/ui components
- **Inference**: PyTorch with diffusers library
- **Storage**: Local file system

## Installation & Setup

### Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Git** - for cloning the repository
- **Python 3.11+** - for running the backend
- **Node.js 18+** and **npm** or **yarn** - for running the frontend
- **MongoDB** - for the database (optional for this app, but required by the stack)

### 1. Clone the Repository

```bash
git clone <your-repository-url>
cd <repository-name>
```

### 2. Backend Setup

#### Install Python Dependencies

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

#### Configure Environment Variables

Create or update `backend/.env`:

```bash
MONGO_URL="mongodb://localhost:27017"
DB_NAME="sdxs_database"
CORS_ORIGINS="*"
```

#### Start the Backend Server

```bash
# From the backend directory
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

The backend API will be available at `http://localhost:8001`

### 3. Frontend Setup

#### Install Node Dependencies

```bash
cd ../frontend

# Using yarn (recommended)
yarn install

# Or using npm
npm install
```

#### Configure Environment Variables

Create or update `frontend/.env`:

```bash
REACT_APP_BACKEND_URL=http://localhost:8001
WDS_SOCKET_PORT=3000
```

#### Start the Frontend Development Server

```bash
# Using yarn
yarn start

# Or using npm
npm start
```

The frontend will be available at `http://localhost:3000`

### 4. First Run

1. Open your browser and navigate to `http://localhost:3000`
2. Enter the model URL: `https://huggingface.co/IDKiro/sdxs-512-0.9`
3. Click **"Fetch & Load Model"** (this will download ~2GB, takes a few minutes)
4. Once loaded, enter a text prompt (e.g., "A beautiful sunset over mountains")
5. Click **"Generate"** and wait 30-60 seconds
6. Your generated image will appear below!

## Project Structure

```
/app/
├── backend/
│   ├── server.py              # Main FastAPI application
│   ├── services/
│   │   ├── hf_downloader.py   # HuggingFace model downloader
│   │   ├── model_loader.py    # Model loading and caching
│   │   └── pipeline.py        # Image generation pipeline
│   ├── models/                # Downloaded models (created at runtime)
│   ├── data/
│   │   └── images/            # Generated images (created at runtime)
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Backend environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styles
│   │   └── components/ui/     # Shadcn UI components
│   ├── package.json           # Node dependencies
│   └── .env                   # Frontend environment variables
└── README.md
```

## How It Works

### 1. Model Loading

When you click "Fetch & Load Model":

1. The app parses the HuggingFace model URL to extract the repo ID
2. Downloads the model files using `huggingface_hub.snapshot_download()`
3. Loads the model into memory using the `diffusers` library
4. Caches the pipeline for fast subsequent generations

### 2. Image Generation

When you click "Generate":

1. Takes your text prompt
2. Runs the SD-XS diffusion pipeline (default: 8 steps, 512x512)
3. Saves the generated PNG to `./backend/data/images/`
4. Displays the image in the UI

## Supported Models

The app works with SD-XS (Stable Diffusion XS) models that are compatible with the `diffusers` library.

### Tested Models

- **IDKiro/sdxs-512-0.9** - SD-XS 512x512 model (recommended for testing)

## API Endpoints

### `POST /api/model/prepare`
Loads a model from HuggingFace.

**Request:**
```json
{
  "modelCardUrl": "https://huggingface.co/IDKiro/sdxs-512-0.9"
}
```

**Response:**
```json
{
  "ok": true,
  "repoId": "IDKiro/sdxs-512-0.9",
  "message": "Model IDKiro/sdxs-512-0.9 loaded successfully"
}
```

### `POST /api/generate`
Generates an image from a text prompt.

**Request:**
```json
{
  "prompt": "A beautiful sunset over mountains",
  "size": "512x512",
  "steps": 8,
  "guidance": 4.0
}
```

**Response:**
```json
{
  "ok": true,
  "imagePath": "/api/images/uuid.png",
  "filename": "uuid.png"
}
```

### `GET /api/images/{filename}`
Serves a generated image file.

## Configuration

### Generation Parameters

- **size**: Image dimensions (default: `512x512`)
- **steps**: Number of inference steps (default: `8`)
- **guidance**: Guidance scale (default: `4.0`)
- **seed**: Random seed for reproducibility (optional)

### Performance

- **CPU Mode**: Works but slower (~30-60s per image)
- **GPU Mode**: Much faster (~2-5s per image) if CUDA is available

The app automatically detects and uses GPU if available.

## Troubleshooting

### Model Download Issues

If model download fails:
- Check your internet connection
- Ensure you have enough disk space (~2GB per model)
- Try accessing the HuggingFace URL in your browser first

### Generation Timeout

If image generation times out:
- The frontend is set to wait 120 seconds
- CPU generation takes 30-60 seconds for 8 steps
- Consider reducing steps to 4 for faster generation (lower quality)

### Memory Issues

If you run out of memory:
- Close other applications
- Try using a smaller model
- Reduce the number of inference steps
- Consider using a machine with more RAM (8GB+ recommended)

## Development

### Running Tests

Backend tests can be run with:
```bash
cd backend
python -m pytest
```

### Code Quality

```bash
# Python linting
cd backend
black .
flake8 .

# JavaScript linting
cd frontend
npm run lint
```

## Notes

- All inference is local - no cloud API calls except initial HuggingFace download
- Generated images are saved permanently in `./backend/data/images/`
- Models are cached in `./backend/models/` directory
- First model load takes time (~2-5 minutes depending on connection)
- Subsequent runs are much faster as the model is cached

## License

This project is for demonstration purposes.
