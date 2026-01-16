"""
Artifact loading utilities for EHR-FM evaluation scripts.

Supports loading model checkpoints from:
- Local filesystem paths
- S3-compatible storage (AWS S3, Tigris, MinIO, etc.)

Environment variables used:
- AWS_ENDPOINT_URL_S3 / MLFLOW_S3_ENDPOINT_URL: S3 endpoint URL
- AWS_ACCESS_KEY_ID: AWS access key
- AWS_SECRET_ACCESS_KEY: AWS secret key
- AWS_REGION: AWS region (optional)
"""

import os
import hashlib
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from loguru import logger


def load_env(env_file: str | Path | None = None) -> None:
    """
    Load environment variables from .env file.
    
    Searches for .env file in the following order:
    1. Provided env_file path
    2. Current working directory
    3. EHR_FM project root (parent of src/)
    
    Args:
        env_file: Optional path to .env file
    """
    if env_file is not None:
        env_path = Path(env_file)
        if env_path.exists():
            load_dotenv(env_path)
            logger.info(f"Loaded environment from: {env_path}")
            return
        else:
            logger.warning(f"Specified .env file not found: {env_path}")
    
    # Try current directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        load_dotenv(cwd_env)
        logger.info(f"Loaded environment from: {cwd_env}")
        return
    
    # Try project root (EHR_FM directory)
    project_root = Path(__file__).parent.parent.parent  # src/utils -> src -> EHR_FM
    project_env = project_root / ".env"
    if project_env.exists():
        load_dotenv(project_env)
        logger.info(f"Loaded environment from: {project_env}")
        return
    
    logger.warning("No .env file found. Using system environment variables only.")


def _get_s3_client():
    """Create boto3 S3 client with configured endpoint."""
    import boto3
    
    # Get endpoint URL (prefer AWS_ENDPOINT_URL_S3, fall back to MLFLOW_S3_ENDPOINT_URL)
    endpoint_url = os.getenv("AWS_ENDPOINT_URL_S3") or os.getenv("MLFLOW_S3_ENDPOINT_URL")
    region = os.getenv("AWS_REGION", "auto")
    
    if not endpoint_url:
        # Standard AWS S3
        return boto3.client("s3", region_name=region if region != "auto" else None)
    
    # S3-compatible endpoint (Tigris, MinIO, etc.)
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        region_name=region if region != "auto" else None,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


def _get_cache_path(uri: str, cache_dir: Path) -> Path:
    """Generate a unique cache path for an S3 URI."""
    # Create a hash of the URI for uniqueness
    uri_hash = hashlib.md5(uri.encode()).hexdigest()[:12]
    
    # Extract filename from URI
    parsed = urlparse(uri)
    filename = Path(parsed.path).name
    
    # Create cache path: cache_dir/hash_filename
    return cache_dir / f"{uri_hash}_{filename}"


def download_checkpoint(
    artifact_uri: str, 
    cache_dir: str | Path = ".cache/checkpoints",
    force_download: bool = False
) -> Path:
    """
    Download checkpoint from S3 artifact URI to local cache.
    
    Args:
        artifact_uri: S3 URI (s3://bucket/path/to/checkpoint.pt)
        cache_dir: Local directory for caching downloaded files
        force_download: If True, re-download even if cached
        
    Returns:
        Path to the downloaded checkpoint file
        
    Raises:
        ValueError: If URI is not a valid S3 URI
        FileNotFoundError: If download fails
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse S3 URI
    parsed = urlparse(artifact_uri)
    if parsed.scheme != "s3":
        raise ValueError(f"Expected s3:// URI, got: {artifact_uri}")
    
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    
    if not bucket or not key:
        raise ValueError(f"Invalid S3 URI: {artifact_uri}")
    
    # Check cache
    cache_path = _get_cache_path(artifact_uri, cache_dir)
    
    if cache_path.exists() and not force_download:
        logger.info(f"Using cached checkpoint: {cache_path}")
        return cache_path
    
    # Download from S3
    logger.info(f"Downloading checkpoint from: {artifact_uri}")
    logger.info(f"  Bucket: {bucket}, Key: {key}")
    
    s3_client = _get_s3_client()
    
    try:
        s3_client.download_file(bucket, key, str(cache_path))
        logger.info(f"Downloaded to: {cache_path}")
        return cache_path
    except Exception as e:
        # Clean up partial download
        if cache_path.exists():
            cache_path.unlink()
        raise FileNotFoundError(f"Failed to download {artifact_uri}: {e}") from e


def resolve_model_path(
    model_spec: str, 
    cache_dir: str | Path = ".cache/checkpoints"
) -> Path:
    """
    Resolve model specification to a local filesystem path.
    
    Handles:
    - S3 URIs (s3://bucket/path) -> downloads to cache and returns cache path
    - Local paths -> returns as-is if exists
    
    Args:
        model_spec: Either an S3 URI or a local filesystem path
        cache_dir: Cache directory for S3 downloads
        
    Returns:
        Path to the model checkpoint file
        
    Raises:
        FileNotFoundError: If model cannot be found or downloaded
    """
    # Check if it's an S3 URI
    if model_spec.startswith("s3://"):
        return download_checkpoint(model_spec, cache_dir=cache_dir)
    
    # Treat as local path
    local_path = Path(model_spec)
    
    if local_path.exists():
        logger.info(f"Using local checkpoint: {local_path}")
        return local_path
    
    raise FileNotFoundError(
        f"Model not found: {model_spec}\n"
        f"  - Not a valid S3 URI (doesn't start with s3://)\n"
        f"  - Local path does not exist"
    )

