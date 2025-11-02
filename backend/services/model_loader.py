import logging
from pathlib import Path
from typing import Optional
import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers.schedulers import LCMScheduler

logger = logging.getLogger(__name__)

class ModelLoader:
    def __init__(self):
        self.pipeline: Optional[StableDiffusionPipeline] = None
        self.repo_id: Optional[str] = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
    
    async def load_model(self, repo_id: str, model_path: Path):
        """Load SD-XS model using diffusers."""
        try:
            logger.info(f"Loading model from {model_path}...")
            
            # Try to load as a complete pipeline first
            try:
                self.pipeline = DiffusionPipeline.from_pretrained(
                    str(model_path),
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    use_safetensors=True
                )
                logger.info("Loaded as complete pipeline")
            except Exception as e:
                logger.warning(f"Could not load as pipeline: {e}")
                # Fall back to loading from HuggingFace directly
                logger.info("Loading from HuggingFace directly...")
                self.pipeline = DiffusionPipeline.from_pretrained(
                    repo_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    safety_checker=None,
                    use_safetensors=True
                )
            
            # Configure scheduler for fast inference
            try:
                self.pipeline.scheduler = LCMScheduler.from_config(
                    self.pipeline.scheduler.config
                )
            except:
                logger.warning("Could not set LCM scheduler, using default")
            
            self.pipeline = self.pipeline.to(self.device)
            
            # Enable optimizations
            if self.device == "cuda":
                try:
                    self.pipeline.enable_attention_slicing()
                except:
                    pass
            
            self.repo_id = repo_id
            logger.info(f"Model {repo_id} loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise Exception(f"Failed to load model: {str(e)}")
    
    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.pipeline is not None
    
    def get_pipeline(self) -> StableDiffusionPipeline:
        """Get the loaded pipeline."""
        if not self.is_loaded():
            raise Exception("No model loaded")
        return self.pipeline
