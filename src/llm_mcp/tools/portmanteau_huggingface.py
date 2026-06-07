"""Hugging Face Portmanteau tool for Local LLM MCP server.

This tool consolidates all Hugging Face operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 10+ separate Hugging Face tools (one per operation), this tool consolidates
related operations into a single interface. Prevents tool explosion (10+ tools → 1 tool) while maintaining
full functionality and improving discoverability. Follows FastMCP 2.13+ best practices.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)


class HuggingFaceConfig(BaseModel):
    """Configuration for Hugging Face operations.

    Supports both HF_TOKEN and HUGGINGFACE_TOKEN environment variables for maximum compatibility.
    """

    api_token: str | None = Field(None, description="Hugging Face API token", alias="token")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")
    default_author: str | None = Field(None, description="Default author for filtering")

    # Download settings
    use_auth_token: bool = Field(True, description="Whether to use auth token for downloads")

    model_config = ConfigDict(env_prefix="HUGGINGFACE_", populate_by_name=True, extra="ignore")

    @classmethod
    def from_env(cls) -> "HuggingFaceConfig":
        """Load configuration from environment variables."""
        # Check for both HF_TOKEN and HUGGINGFACE_TOKEN
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        config_data = {"api_token": token} if token else {}

        # Load other config from environment
        return cls(**config_data)


# Import FastMCP components
try:
    from fastmcp import FastMCP
    from fastmcp.tools import Tool

    FASTMCP_AVAILABLE = True
except ImportError:
    logger.error("FastMCP not available - portmanteau tools require FastMCP >= 2.12.0")
    FASTMCP_AVAILABLE = False

# Hugging Face dependencies
try:
    import huggingface_hub
    from huggingface_hub import HfApi, HfFolder, login, logout, whoami
    from huggingface_hub.utils import HfHubHTTPError

    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install with: pip install huggingface-hub")


async def llm_huggingface(
    operation: str,
    # Model operations
    model_id: str | None = None,
    # Dataset operations
    dataset_id: str | None = None,
    # Repository operations
    repo_id: str | None = None,
    repo_type: str = "model",
    # File operations
    local_path: str | None = None,
    filename: str | None = None,
    # Search and discovery
    query: str | None = None,
    author: str | None = None,
    # Upload operations
    path_in_repo: str | None = None,
    commit_message: str = "Upload via Local LLM MCP",
    # Token operations
    token: str | None = None,
) -> dict[str, Any]:
    """Comprehensive Hugging Face management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 15+ Hugging Face operations into one tool.

    CONFIGURATION:
    - Environment Variables: HUGGINGFACE_TOKEN or HF_TOKEN for authentication
    - Automatic token detection for gated models (FLUX, etc.)
    - Integrated with main config system for consistency

    SUPPORTED OPERATIONS:
    - login: Authenticate with Hugging Face (token parameter or env var)
    - logout: Sign out from Hugging Face
    - whoami: Check current user authentication
    - list_models: List available models with filtering
    - list_datasets: List available datasets with filtering
    - get_model_info: Get detailed model information
    - download_model: Download models (supports gated models like FLUX)
    - download_dataset: Download datasets (supports private datasets)
    - upload_file: Upload files to repositories
    - create_repo: Create new repositories
    - delete_repo: Delete repositories
    - list_user_repos: List user's repositories
    - search_models: Search for models by query
    - search_datasets: Search for datasets by query

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_id: Hugging Face model ID (e.g., "microsoft/DialoGPT-medium")
        dataset_id: Hugging Face dataset ID
        repo_id: Repository ID for operations
        repo_type: Type of repository ("model", "dataset", "space")
        local_path: Local file path for upload/download
        filename: Filename in repository
        query: Search query for discovery operations
        author: Author filter for listings
        path_in_repo: Path within repository for uploads
        commit_message: Commit message for uploads
        token: Hugging Face API token (optional if HUGGINGFACE_TOKEN/HF_TOKEN env var set)

    Returns:
        Operation-specific result dictionary
    """
    if not HF_AVAILABLE:
        return {"error": "Hugging Face Hub not available. Install with: pip install huggingface-hub"}

    try:
        # Load configuration from environment
        config = HuggingFaceConfig.from_env()
        api = HfApi(token=config.api_token)

        if operation == "login":
            # Use provided token or config token
            auth_token = token or config.api_token
            if not auth_token:
                return {
                    "error": (
                        "token required for login operation. Set HUGGINGFACE_TOKEN or "
                        "HF_TOKEN environment variable, or pass token parameter"
                    )
                }
            try:
                login(token=auth_token)
                user_info = whoami()
                return {
                    "success": True,
                    "message": "Successfully logged in to Hugging Face",
                    "user": user_info.get("name"),
                    "token_source": "parameter" if token else "environment",
                }
            except Exception as e:
                return {"error": f"Login failed: {e!s}"}

        elif operation == "logout":
            try:
                logout()
                return {"success": True, "message": "Successfully logged out from Hugging Face"}
            except Exception as e:
                return {"error": f"Logout failed: {e!s}"}

        elif operation == "whoami":
            try:
                # Check if we have a token available
                auth_token = config.api_token
                if not auth_token:
                    return {
                        "error": (
                            "No authentication token available. Set HUGGINGFACE_TOKEN or "
                            "HF_TOKEN environment variable, or use login operation first"
                        )
                    }

                user_info = whoami(token=auth_token)
                return {
                    "success": True,
                    "user": {
                        "name": user_info.get("name"),
                        "fullname": user_info.get("fullname"),
                        "email": user_info.get("email"),
                        "organization": user_info.get("orgs", []),
                        "is_pro": user_info.get("isPro", False),
                    },
                    "token_source": "environment",
                }
            except Exception as e:
                return {"error": f"Authentication check failed: {e!s}. Make sure you're logged in with a valid token."}

        elif operation == "list_models":
            try:
                models = api.list_models(author=author, limit=50)
                return {
                    "success": True,
                    "models": [
                        {
                            "id": model.id,
                            "author": model.author,
                            "downloads": model.downloads,
                            "likes": model.likes,
                            "tags": model.tags,
                            "pipeline_tag": model.pipeline_tag,
                            "created_at": model.created_at.isoformat() if model.created_at else None,
                        }
                        for model in models
                    ],
                    "count": len(list(models)),
                }
            except Exception as e:
                return {"error": f"Failed to list models: {e!s}"}

        elif operation == "list_datasets":
            try:
                datasets = api.list_datasets(author=author, limit=50)
                return {
                    "success": True,
                    "datasets": [
                        {
                            "id": dataset.id,
                            "author": dataset.author,
                            "downloads": dataset.downloads,
                            "likes": dataset.likes,
                            "tags": dataset.tags,
                            "description": dataset.description,
                            "created_at": dataset.created_at.isoformat() if dataset.created_at else None,
                        }
                        for dataset in datasets
                    ],
                    "count": len(list(datasets)),
                }
            except Exception as e:
                return {"error": f"Failed to list datasets: {e!s}"}

        elif operation == "get_model_info":
            if not model_id:
                return {"error": "model_id required for get_model_info operation"}
            try:
                model_info = api.model_info(model_id)
                return {
                    "success": True,
                    "model": {
                        "id": model_info.id,
                        "author": model_info.author,
                        "downloads": model_info.downloads,
                        "likes": model_info.likes,
                        "tags": model_info.tags,
                        "pipeline_tag": model_info.pipeline_tag,
                        "license": model_info.license,
                        "description": model_info.description,
                        "card_data": model_info.card_data,
                        "created_at": model_info.created_at.isoformat() if model_info.created_at else None,
                        "last_modified": model_info.last_modified.isoformat() if model_info.last_modified else None,
                        "siblings": [
                            {"rfilename": sibling.rfilename, "size": sibling.size, "blob_id": sibling.blob_id}
                            for sibling in model_info.siblings
                        ],
                    },
                }
            except Exception as e:
                return {"error": f"Failed to get model info: {e!s}"}

        elif operation == "download_model":
            if not model_id:
                return {"error": "model_id required for download_model operation"}
            if not local_path:
                return {"error": "local_path required for download_model operation"}
            try:
                from huggingface_hub import snapshot_download

                download_path = snapshot_download(
                    repo_id=model_id,
                    local_dir=local_path,
                    repo_type="model",
                    token=config.api_token if config.use_auth_token else None,
                )
                return {
                    "success": True,
                    "model_id": model_id,
                    "local_path": download_path,
                    "message": f"Model downloaded to {download_path}",
                    "auth_used": config.api_token is not None and config.use_auth_token,
                }
            except Exception as e:
                error_msg = f"Failed to download model: {e!s}"
                if "gated" in str(e).lower() or "private" in str(e).lower():
                    error_msg += (
                        ". This may be a gated model requiring authentication. "
                        "Ensure HUGGINGFACE_TOKEN or HF_TOKEN is set."
                    )
                return {"error": error_msg}

        elif operation == "download_dataset":
            if not dataset_id:
                return {"error": "dataset_id required for download_dataset operation"}
            if not local_path:
                return {"error": "local_path required for download_dataset operation"}
            try:
                from huggingface_hub import snapshot_download

                download_path = snapshot_download(
                    repo_id=dataset_id,
                    local_dir=local_path,
                    repo_type="dataset",
                    token=config.api_token if config.use_auth_token else None,
                )
                return {
                    "success": True,
                    "dataset_id": dataset_id,
                    "local_path": download_path,
                    "message": f"Dataset downloaded to {download_path}",
                    "auth_used": config.api_token is not None and config.use_auth_token,
                }
            except Exception as e:
                error_msg = f"Failed to download dataset: {e!s}"
                if "gated" in str(e).lower() or "private" in str(e).lower():
                    error_msg += (
                        ". This may be a gated dataset requiring authentication. "
                        "Ensure HUGGINGFACE_TOKEN or HF_TOKEN is set."
                    )
                return {"error": error_msg}

        elif operation == "upload_file":
            if not repo_id:
                return {"error": "repo_id required for upload_file operation"}
            if not local_path:
                return {"error": "local_path required for upload_file operation"}
            if not path_in_repo:
                return {"error": "path_in_repo required for upload_file operation"}
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type=repo_type,
                    commit_message=commit_message,
                )
                return {
                    "success": True,
                    "repo_id": repo_id,
                    "uploaded_file": path_in_repo,
                    "commit_message": commit_message,
                    "message": f"File uploaded to {repo_id}/{path_in_repo}",
                }
            except Exception as e:
                return {"error": f"Failed to upload file: {e!s}"}

        elif operation == "create_repo":
            if not repo_id:
                return {"error": "repo_id required for create_repo operation"}
            try:
                api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
                return {
                    "success": True,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "url": f"https://huggingface.co/{repo_id}",
                    "message": f"Repository created: https://huggingface.co/{repo_id}",
                }
            except Exception as e:
                return {"error": f"Failed to create repository: {e!s}"}

        elif operation == "delete_repo":
            if not repo_id:
                return {"error": "repo_id required for delete_repo operation"}
            try:
                api.delete_repo(repo_id=repo_id, repo_type=repo_type)
                return {
                    "success": True,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "message": f"Repository deleted: {repo_id}",
                }
            except Exception as e:
                return {"error": f"Failed to delete repository: {e!s}"}

        elif operation == "list_user_repos":
            try:
                repos = api.list_repos()
                return {
                    "success": True,
                    "repositories": [
                        {
                            "id": repo.id,
                            "type": repo.type,
                            "private": repo.private,
                            "downloads": repo.downloads,
                            "likes": repo.likes,
                            "created_at": repo.created_at.isoformat() if repo.created_at else None,
                        }
                        for repo in repos
                    ],
                    "count": len(list(repos)),
                }
            except Exception as e:
                return {"error": f"Failed to list user repositories: {e!s}"}

        elif operation == "search_models":
            if not query:
                return {"error": "query required for search_models operation"}
            try:
                models = api.list_models(search=query, limit=20)
                return {
                    "success": True,
                    "query": query,
                    "models": [
                        {
                            "id": model.id,
                            "author": model.author,
                            "downloads": model.downloads,
                            "likes": model.likes,
                            "tags": model.tags[:5] if model.tags else [],  # Limit tags
                            "pipeline_tag": model.pipeline_tag,
                        }
                        for model in models
                    ],
                    "count": len(list(models)),
                }
            except Exception as e:
                return {"error": f"Failed to search models: {e!s}"}

        elif operation == "search_datasets":
            if not query:
                return {"error": "query required for search_datasets operation"}
            try:
                datasets = api.list_datasets(search=query, limit=20)
                return {
                    "success": True,
                    "query": query,
                    "datasets": [
                        {
                            "id": dataset.id,
                            "author": dataset.author,
                            "downloads": dataset.downloads,
                            "likes": dataset.likes,
                            "tags": dataset.tags[:5] if dataset.tags else [],
                            "description": dataset.description[:200] if dataset.description else None,
                        }
                        for dataset in datasets
                    ],
                    "count": len(list(datasets)),
                }
            except Exception as e:
                return {"error": f"Failed to search datasets: {e!s}"}

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "login",
                    "logout",
                    "whoami",
                    "list_models",
                    "list_datasets",
                    "get_model_info",
                    "download_model",
                    "download_dataset",
                    "upload_file",
                    "create_repo",
                    "delete_repo",
                    "list_user_repos",
                    "search_models",
                    "search_datasets",
                ],
            }

    except Exception as e:
        logger.error(f"Error in llm_huggingface operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {e!s}", "operation": operation}


def register_llm_huggingface_tools(mcp):
    """Register the Hugging Face Portmanteau tool with the MCP server."""
    if not FASTMCP_AVAILABLE:
        logger.error("Cannot register Hugging Face tools - FastMCP not available")
        return mcp

    @mcp.tool()
    async def llm_huggingface_tool(
        operation: str,
        model_id: str | None = None,
        dataset_id: str | None = None,
        repo_id: str | None = None,
        repo_type: str = "model",
        local_path: str | None = None,
        filename: str | None = None,
        query: str | None = None,
        author: str | None = None,
        path_in_repo: str | None = None,
        commit_message: str = "Upload via Local LLM MCP",
        token: str | None = None,
    ) -> dict[str, Any]:
        """Hugging Face Portmanteau Tool - Consolidated Hugging Face operations.

        This tool consolidates all Hugging Face operations into a single interface,
        reducing the number of MCP tools while maintaining full functionality.

        Use the 'operation' parameter to specify what you want to do:
        - login: Authenticate with Hugging Face (requires token)
        - logout: Sign out from Hugging Face
        - whoami: Check current user authentication
        - list_models: List available models (optional author filter)
        - list_datasets: List available datasets (optional author filter)
        - get_model_info: Get detailed model information (requires model_id)
        - download_model: Download a model (requires model_id, local_path)
        - download_dataset: Download a dataset (requires dataset_id, local_path)
        - upload_file: Upload file to repository (requires repo_id, local_path, path_in_repo)
        - create_repo: Create new repository (requires repo_id)
        - delete_repo: Delete repository (requires repo_id)
        - list_user_repos: List user's repositories
        - search_models: Search models by query (requires query)
        - search_datasets: Search datasets by query (requires query)
        """
        return await llm_huggingface(
            operation=operation,
            model_id=model_id,
            dataset_id=dataset_id,
            repo_id=repo_id,
            repo_type=repo_type,
            local_path=local_path,
            filename=filename,
            query=query,
            author=author,
            path_in_repo=path_in_repo,
            commit_message=commit_message,
            token=token,
        )

    logger.info("Registered Hugging Face portmanteau tool")
    return mcp
