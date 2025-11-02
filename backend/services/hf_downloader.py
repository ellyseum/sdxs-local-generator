import os
import re
from pathlib import Path
from typing import Optional
import logging
from huggingface_hub import snapshot_download, hf_hub_url, list_repo_files

logger = logging.getLogger(__name__)

class HFDownloader:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
    
    def parse_repo_id(self, model_card_url: str) -> str:
        """Extract repo_id from HuggingFace URL."""
        # Handle various URL formats
        patterns = [
            r'huggingface\.co/([^/]+/[^/]+)',
            r'^([^/]+/[^/]+)$'  # Direct repo_id
        ]
        
        for pattern in patterns:
            match = re.search(pattern, model_card_url)
            if match:
                return match.group(1)
        
        raise ValueError(f"Could not parse repo_id from: {model_card_url}")
    
    async def download_model(self, repo_id: str) -> Path:
        """Download model from HuggingFace."""
        try:
            model_path = self.models_dir / repo_id.replace('/', '_')
            
            # Check if already downloaded
            if model_path.exists():
                logger.info(f"Model {repo_id} already downloaded at {model_path}")
                return model_path
            
            logger.info(f"Downloading model {repo_id} from HuggingFace...")
            
            # Download the entire repository
            download_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(model_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Model downloaded successfully to {download_path}")
            return Path(download_path)
            
        except Exception as e:
            logger.error(f"Error downloading model: {e}")
            raise Exception(f"Failed to download model {repo_id}: {str(e)}")
