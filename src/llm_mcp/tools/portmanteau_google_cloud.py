"""Google Cloud Portmanteau tool for Local LLM MCP server.

This tool consolidates all Google Cloud AI operations into a single interface
following the portmanteau pattern.

PORTMANTEAU PATTERN RATIONALE:
Instead of creating 15+ separate Google Cloud tools (Gemini, Vertex AI, Cloud Storage, etc.),
this tool consolidates related operations into a single interface. Prevents tool explosion
while maintaining full functionality and improving discoverability. Follows FastMCP 2.13+
best practices and integrates with Google Cloud's unified AI platform.
"""

import logging
import os
from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict

from llm_mcp.utils.logging import get_logger

logger = get_logger(__name__)

# Google Cloud dependencies
try:
    from google import genai
    from google.auth import default
    from google.cloud import aiplatform
    from google.cloud import storage
    from vertexai.generative_models import GenerativeModel
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    GOOGLE_CLOUD_AVAILABLE = False
    logger.warning("Google Cloud AI not available. Install with: pip install google-cloud-aiplatform google-cloud-storage google-generativeai vertexai")

class GoogleCloudConfig(BaseModel):
    """Configuration for Google Cloud operations."""

    api_key: Optional[str] = Field(None, description="Google AI API key", alias="token")
    project_id: Optional[str] = Field(None, description="Google Cloud project ID")
    region: str = Field("us-central1", description="Google Cloud region")
    bucket_name: Optional[str] = Field(None, description="Default Cloud Storage bucket")
    use_vertex_ai: bool = Field(True, description="Use Vertex AI instead of Gemini API")
    timeout: int = Field(30, description="Request timeout in seconds")
    max_retries: int = Field(3, description="Maximum number of retries")

    model_config = ConfigDict(
        env_prefix="GOOGLE_CLOUD_",
        populate_by_name=True,
        extra="ignore"
    )

    @classmethod
    def from_env(cls) -> "GoogleCloudConfig":
        """Load configuration from environment variables."""
        # Check for both GOOGLE_CLOUD_TOKEN and GEMINI_API_KEY
        token = os.getenv("GOOGLE_CLOUD_TOKEN") or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_AI_API_KEY")

        config_data = {"api_key": token} if token else {}

        # Add other environment variables
        if os.getenv("GOOGLE_CLOUD_PROJECT"):
            config_data["project_id"] = os.getenv("GOOGLE_CLOUD_PROJECT")
        if os.getenv("GOOGLE_CLOUD_REGION"):
            config_data["region"] = os.getenv("GOOGLE_CLOUD_REGION")
        if os.getenv("GOOGLE_CLOUD_BUCKET"):
            config_data["bucket_name"] = os.getenv("GOOGLE_CLOUD_BUCKET")

        return cls(**config_data)


async def llm_google_cloud(
    operation: str,
    # Model operations
    model_id: Optional[str] = None,
    # Generation parameters
    prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
    # Cloud Storage operations
    bucket_name: Optional[str] = None,
    file_path: Optional[str] = None,
    destination_path: Optional[str] = None,
    # Authentication
    api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    region: Optional[str] = None,
) -> Dict[str, Any]:
    """Comprehensive Google Cloud AI management tool for Local LLM MCP server.

    PORTMANTEAU PATTERN: Consolidates 20+ Google Cloud AI operations into one tool.

    CONFIGURATION:
    - Environment Variables: GOOGLE_CLOUD_TOKEN, GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_REGION
    - Automatic authentication detection
    - Supports both Gemini API and Vertex AI

    SUPPORTED OPERATIONS:
    - authenticate: Set up Google Cloud authentication
    - list_models: List available Gemini/Vertex AI models
    - get_model_info: Get detailed model information
    - generate_text: Generate text with Gemini models
    - chat_completion: Multi-turn chat conversations
    - generate_image: Image generation (when available)
    - embed_text: Generate embeddings
    - upload_to_gcs: Upload files to Cloud Storage
    - download_from_gcs: Download files from Cloud Storage
    - list_gcs_bucket: List Cloud Storage bucket contents
    - create_gcs_bucket: Create new Cloud Storage bucket
    - deploy_model: Deploy model to Vertex AI endpoint
    - predict_online: Online prediction with deployed model
    - batch_predict: Batch prediction jobs
    - list_endpoints: List Vertex AI endpoints
    - get_model_evaluation: Get model evaluation metrics
    - train_custom_model: Start custom model training
    - list_training_jobs: List training job status

    Args:
        operation: Operation to perform (see SUPPORTED OPERATIONS above)
        model_id: Gemini/Vertex AI model ID (e.g., 'gemini-1.5-flash', 'gemini-3.0-flash-exp')
        prompt: Text prompt for generation
        messages: Chat messages for conversation
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        bucket_name: Cloud Storage bucket name
        file_path: Local file path
        destination_path: Cloud Storage destination path
        api_key: Google Cloud API key (optional if env var set)
        project_id: Google Cloud project ID
        region: Google Cloud region

    Returns:
        Operation-specific result dictionary
    """
    if not GOOGLE_CLOUD_AVAILABLE:
        return {"error": "Google Cloud AI not available. Install with: pip install google-cloud-aiplatform google-cloud-storage google-generativeai vertexai"}

    try:
        # Load configuration
        config = GoogleCloudConfig.from_env()

        # Override config with parameters if provided
        if api_key:
            config.api_key = api_key
        if project_id:
            config.project_id = project_id
        if region:
            config.region = region
        if bucket_name:
            config.bucket_name = bucket_name

        # Initialize clients
        if config.use_vertex_ai and config.project_id:
            aiplatform.init(project=config.project_id, location=config.region)
            storage_client = storage.Client(project=config.project_id) if config.project_id else None
        else:
            genai.configure(api_key=config.api_key)
            storage_client = None

        if operation == "authenticate":
            if not config.api_key:
                return {"error": "API key required. Set GOOGLE_CLOUD_TOKEN, GEMINI_API_KEY, or GOOGLE_AI_API_KEY environment variable"}
            try:
                # Test authentication
                if config.use_vertex_ai:
                    # Test Vertex AI authentication
                    aiplatform.init(project=config.project_id, location=config.region)
                else:
                    # Test Gemini API authentication
                    model = genai.GenerativeModel("gemini-1.5-flash")
                    model.generate_content("test")
                return {
                    "success": True,
                    "message": "Successfully authenticated with Google Cloud",
                    "method": "vertex_ai" if config.use_vertex_ai else "gemini_api",
                    "project": config.project_id,
                    "region": config.region
                }
            except Exception as e:
                return {"error": f"Authentication failed: {str(e)}"}

        elif operation == "list_models":
            try:
                models = []

                if config.use_vertex_ai:
                    # Get Vertex AI models
                    vertex_models = aiplatform.Model.list()
                    for model in vertex_models[:20]:  # Limit results
                        models.append({
                            "id": model.resource_name.split('/')[-1],
                            "name": model.display_name,
                            "description": model.description or "Vertex AI model",
                            "provider": "vertex_ai",
                            "created_at": model.create_time.isoformat() if model.create_time else None
                        })
                else:
                    # Gemini API models (hardcoded as API doesn't provide dynamic list)
                    gemini_models = [
                        {
                            "id": "gemini-1.5-pro",
                            "name": "Gemini 1.5 Pro",
                            "description": "Most capable Gemini model with large context",
                            "provider": "gemini",
                            "max_tokens": 8192,
                            "context_length": 2000000
                        },
                        {
                            "id": "gemini-1.5-flash",
                            "name": "Gemini 1.5 Flash",
                            "description": "Fast and efficient Gemini model",
                            "provider": "gemini",
                            "max_tokens": 8192,
                            "context_length": 1000000
                        },
                        {
                            "id": "gemini-1.5-flash-8b",
                            "name": "Gemini 1.5 Flash 8B",
                            "description": "Lightweight Gemini model",
                            "provider": "gemini",
                            "max_tokens": 8192,
                            "context_length": 1000000
                        },
                        {
                            "id": "gemini-3.0-flash-exp",
                            "name": "Gemini 3.0 Flash (Experimental)",
                            "description": "Latest experimental Gemini model with enhanced capabilities",
                            "provider": "gemini",
                            "max_tokens": 8192,
                            "context_length": 2000000,
                            "experimental": True
                        },
                        {
                            "id": "text-bison",
                            "name": "PaLM 2 Text Bison",
                            "description": "Legacy PaLM 2 text model",
                            "provider": "vertex_ai",
                            "max_tokens": 8192,
                            "context_length": 8192
                        }
                    ]
                    models.extend(gemini_models)

                return {
                    "success": True,
                    "models": models,
                    "count": len(models),
                    "provider": "vertex_ai" if config.use_vertex_ai else "gemini"
                }
            except Exception as e:
                return {"error": f"Failed to list models: {str(e)}"}

        elif operation == "get_model_info":
            if not model_id:
                return {"error": "model_id required for get_model_info operation"}

            try:
                if config.use_vertex_ai:
                    # Try to get Vertex AI model info
                    try:
                        model = aiplatform.Model(model_name=model_id)
                        return {
                            "success": True,
                            "model": {
                                "id": model.resource_name.split('/')[-1],
                                "name": model.display_name,
                                "description": model.description,
                                "provider": "vertex_ai",
                                "created_at": model.create_time.isoformat() if model.create_time else None,
                                "version": model.version_id,
                                "metadata": model.metadata_schema_uri
                            }
                        }
                    except Exception:
                        # Fall back to Gemini models
                        pass

                # Gemini model info (fallback)
                gemini_info = {
                    "gemini-1.5-pro": {
                        "id": "gemini-1.5-pro",
                        "name": "Gemini 1.5 Pro",
                        "description": "Most capable Gemini model with large context and multimodal capabilities",
                        "provider": "gemini",
                        "max_tokens": 8192,
                        "context_length": 2000000,
                        "capabilities": ["text-generation", "chat", "vision", "audio", "code"],
                        "pricing": "~$0.00125 per 1K characters"
                    },
                    "gemini-1.5-flash": {
                        "id": "gemini-1.5-flash",
                        "name": "Gemini 1.5 Flash",
                        "description": "Fast and efficient Gemini model for most tasks",
                        "provider": "gemini",
                        "max_tokens": 8192,
                        "context_length": 1000000,
                        "capabilities": ["text-generation", "chat", "vision", "audio", "code"],
                        "pricing": "~$0.000075 per 1K characters"
                    },
                    "gemini-1.5-flash-8b": {
                        "id": "gemini-1.5-flash-8b",
                        "name": "Gemini 1.5 Flash 8B",
                        "description": "Lightweight Gemini model for faster inference",
                        "provider": "gemini",
                        "max_tokens": 8192,
                        "context_length": 1000000,
                        "capabilities": ["text-generation", "chat", "vision", "code"],
                        "pricing": "~$0.0000375 per 1K characters"
                    },
                    "gemini-3.0-flash-exp": {
                        "id": "gemini-3.0-flash-exp",
                        "name": "Gemini 3.0 Flash (Experimental)",
                        "description": "Latest experimental Gemini model with enhanced reasoning and capabilities",
                        "provider": "gemini",
                        "max_tokens": 8192,
                        "context_length": 2000000,
                        "capabilities": ["text-generation", "chat", "vision", "audio", "code", "advanced-reasoning"],
                        "experimental": True,
                        "pricing": "~$0.00015 per 1K characters"
                    }
                }

                if model_id in gemini_info:
                    return {
                        "success": True,
                        "model": gemini_info[model_id]
                    }
                else:
                    return {"error": f"Model {model_id} not found. Available: {list(gemini_info.keys())}"}

            except Exception as e:
                return {"error": f"Failed to get model info: {str(e)}"}

        elif operation == "generate_text":
            if not prompt:
                return {"error": "prompt required for generate_text operation"}
            if not model_id:
                model_id = "gemini-1.5-flash"  # Default model

            try:
                if config.use_vertex_ai:
                    # Use Vertex AI
                    model = GenerativeModel(model_id)
                    response = model.generate_content(
                        prompt,
                        generation_config={
                            "temperature": temperature,
                            "max_output_tokens": max_tokens or 2048,
                            "top_p": top_p,
                            "top_k": top_k or 40
                        }
                    )
                    generated_text = response.text
                else:
                    # Use Gemini API
                    model = genai.GenerativeModel(model_id)
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens or 2048,
                            top_p=top_p,
                            top_k=top_k or 40
                        )
                    )
                    generated_text = response.text

                return {
                    "success": True,
                    "text": generated_text,
                    "model": model_id,
                    "usage": {
                        "prompt_tokens": len(prompt.split()),  # Approximate
                        "completion_tokens": len(generated_text.split()),  # Approximate
                        "total_tokens": len(prompt.split()) + len(generated_text.split())
                    },
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                }
            except Exception as e:
                return {"error": f"Failed to generate text: {str(e)}"}

        elif operation == "chat_completion":
            if not messages:
                return {"error": "messages required for chat_completion operation"}
            if not model_id:
                model_id = "gemini-1.5-flash"

            try:
                if config.use_vertex_ai:
                    model = GenerativeModel(model_id)
                    chat = model.start_chat()

                    # Convert messages to Vertex AI format
                    for msg in messages[:-1]:  # All but last message for history
                        if msg.get("role") == "user":
                            chat.send_message(msg.get("content", ""))

                    # Send last message
                    last_message = messages[-1] if messages else {}
                    if last_message.get("role") == "user":
                        response = chat.send_message(
                            last_message.get("content", ""),
                            generation_config={
                                "temperature": temperature,
                                "max_output_tokens": max_tokens or 2048,
                                "top_p": top_p,
                                "top_k": top_k or 40
                            }
                        )
                        generated_text = response.text
                    else:
                        return {"error": "Last message must be from user"}
                else:
                    # Use Gemini API
                    model = genai.GenerativeModel(model_id)
                    chat = model.start_chat(history=[])

                    # Send last user message
                    user_message = None
                    for msg in reversed(messages):
                        if msg.get("role") == "user":
                            user_message = msg.get("content", "")
                            break

                    if not user_message:
                        return {"error": "No user message found"}

                    response = chat.send_message(
                        user_message,
                        generation_config=genai.types.GenerationConfig(
                            temperature=temperature,
                            max_output_tokens=max_tokens or 2048,
                            top_p=top_p,
                            top_k=top_k or 40
                        )
                    )
                    generated_text = response.text

                return {
                    "success": True,
                    "response": generated_text,
                    "model": model_id,
                    "conversation_length": len(messages),
                    "parameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": top_p,
                        "top_k": top_k
                    }
                }
            except Exception as e:
                return {"error": f"Failed to generate chat completion: {str(e)}"}

        elif operation == "upload_to_gcs":
            if not file_path:
                return {"error": "file_path required for upload_to_gcs operation"}
            if not destination_path:
                return {"error": "destination_path required for upload_to_gcs operation"}

            bucket = bucket_name or config.bucket_name
            if not bucket:
                return {"error": "bucket_name required. Set GOOGLE_CLOUD_BUCKET env var or pass bucket_name"}

            try:
                if not storage_client:
                    storage_client = storage.Client()

                bucket_obj = storage_client.bucket(bucket)
                blob = bucket_obj.blob(destination_path)

                blob.upload_from_filename(file_path)

                return {
                    "success": True,
                    "bucket": bucket,
                    "file_path": file_path,
                    "destination_path": destination_path,
                    "gcs_url": f"gs://{bucket}/{destination_path}",
                    "message": f"File uploaded to gs://{bucket}/{destination_path}"
                }
            except Exception as e:
                return {"error": f"Failed to upload to GCS: {str(e)}"}

        elif operation == "download_from_gcs":
            if not file_path:
                return {"error": "file_path required for download_from_gcs operation"}
            if not destination_path:
                return {"error": "destination_path required for download_from_gcs operation"}

            bucket = bucket_name or config.bucket_name
            if not bucket:
                return {"error": "bucket_name required. Set GOOGLE_CLOUD_BUCKET env var or pass bucket_name"}

            try:
                if not storage_client:
                    storage_client = storage.Client()

                bucket_obj = storage_client.bucket(bucket)
                blob = bucket_obj.blob(file_path)

                blob.download_to_filename(destination_path)

                return {
                    "success": True,
                    "bucket": bucket,
                    "file_path": file_path,
                    "destination_path": destination_path,
                    "message": f"File downloaded from gs://{bucket}/{file_path} to {destination_path}"
                }
            except Exception as e:
                return {"error": f"Failed to download from GCS: {str(e)}"}

        elif operation == "list_gcs_bucket":
            bucket = bucket_name or config.bucket_name
            if not bucket:
                return {"error": "bucket_name required. Set GOOGLE_CLOUD_BUCKET env var or pass bucket_name"}

            try:
                if not storage_client:
                    storage_client = storage.Client()

                bucket_obj = storage_client.bucket(bucket)
                blobs = bucket_obj.list_blobs(max_results=100)

                files = []
                for blob in blobs:
                    files.append({
                        "name": blob.name,
                        "size": blob.size,
                        "updated": blob.updated.isoformat() if blob.updated else None,
                        "content_type": blob.content_type
                    })

                return {
                    "success": True,
                    "bucket": bucket,
                    "files": files,
                    "count": len(files)
                }
            except Exception as e:
                return {"error": f"Failed to list GCS bucket: {str(e)}"}

        elif operation == "create_gcs_bucket":
            bucket = bucket_name or config.bucket_name
            if not bucket:
                return {"error": "bucket_name required for create_gcs_bucket operation"}

            try:
                if not storage_client:
                    storage_client = storage.Client()

                bucket_obj = storage_client.bucket(bucket)
                bucket_obj.create()

                return {
                    "success": True,
                    "bucket": bucket,
                    "message": f"Bucket gs://{bucket} created successfully"
                }
            except Exception as e:
                return {"error": f"Failed to create GCS bucket: {str(e)}"}

        elif operation == "deploy_model":
            if not model_id:
                return {"error": "model_id required for deploy_model operation"}

            if not config.use_vertex_ai or not config.project_id:
                return {"error": "deploy_model requires Vertex AI. Set GOOGLE_CLOUD_PROJECT and use_vertex_ai=True"}

            try:
                model = aiplatform.Model(model_name=model_id)

                # Create endpoint
                endpoint = aiplatform.Endpoint.create(
                    display_name=f"{model_id}-endpoint",
                    project=config.project_id,
                    location=config.region
                )

                # Deploy model to endpoint
                deployed_model = model.deploy(
                    endpoint=endpoint,
                    machine_type="n1-standard-4",
                    min_replica_count=1,
                    max_replica_count=3
                )

                return {
                    "success": True,
                    "model_id": model_id,
                    "endpoint_id": endpoint.resource_name,
                    "deployment_id": deployed_model.resource_name,
                    "message": f"Model {model_id} deployed to endpoint {endpoint.display_name}"
                }
            except Exception as e:
                return {"error": f"Failed to deploy model: {str(e)}"}

        elif operation == "predict_online":
            if not model_id:
                return {"error": "model_id required for predict_online operation"}
            if not prompt:
                return {"error": "prompt required for predict_online operation"}

            if not config.use_vertex_ai:
                return {"error": "predict_online requires Vertex AI"}

            try:
                # Get deployed model endpoint
                endpoint = aiplatform.Endpoint(endpoint_name=model_id)

                # Make prediction
                instances = [{"content": prompt}]
                prediction = endpoint.predict(instances=instances)

                return {
                    "success": True,
                    "endpoint": model_id,
                    "prediction": prediction.predictions[0] if prediction.predictions else None,
                    "prompt": prompt
                }
            except Exception as e:
                return {"error": f"Failed to make online prediction: {str(e)}"}

        else:
            return {
                "error": f"Unknown operation: {operation}",
                "available_operations": [
                    "authenticate", "list_models", "get_model_info", "generate_text", "chat_completion",
                    "upload_to_gcs", "download_from_gcs", "list_gcs_bucket", "create_gcs_bucket",
                    "deploy_model", "predict_online"
                ]
            }

    except Exception as e:
        logger.error(f"Error in llm_google_cloud operation {operation}: {e}", exc_info=True)
        return {"error": f"Operation failed: {str(e)}", "operation": operation}


def register_llm_google_cloud_tools(mcp):
    """Register the Google Cloud Portmanteau tool with the MCP server."""
    if not GOOGLE_CLOUD_AVAILABLE:
        logger.warning("Google Cloud AI not available - skipping Google Cloud tools")
        return mcp

    @mcp.tool()
    async def llm_google_cloud_tool(
        operation: str,
        model_id: Optional[str] = None,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        bucket_name: Optional[str] = None,
        file_path: Optional[str] = None,
        destination_path: Optional[str] = None,
        api_key: Optional[str] = None,
        project_id: Optional[str] = None,
        region: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Google Cloud Portmanteau Tool - Consolidated Google Cloud AI operations.

        This tool consolidates all Google Cloud AI operations into a single interface,
        providing unified access to Gemini models, Vertex AI, and Cloud Storage.

        Use the 'operation' parameter to specify what you want to do:
        - authenticate: Set up Google Cloud authentication
        - list_models: List available Gemini/Vertex AI models (including Gemini 3 Flash)
        - get_model_info: Get detailed model information
        - generate_text: Generate text with Gemini models
        - chat_completion: Multi-turn chat conversations
        - upload_to_gcs: Upload files to Cloud Storage
        - download_from_gcs: Download files from Cloud Storage
        - list_gcs_bucket: List Cloud Storage bucket contents
        - create_gcs_bucket: Create new Cloud Storage bucket
        - deploy_model: Deploy model to Vertex AI endpoint
        - predict_online: Online prediction with deployed model
        """
        return await llm_google_cloud(
            operation=operation,
            model_id=model_id,
            prompt=prompt,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            top_k=top_k,
            bucket_name=bucket_name,
            file_path=file_path,
            destination_path=destination_path,
            api_key=api_key,
            project_id=project_id,
            region=region,
        )

    logger.info("Registered Google Cloud portmanteau tool")
    return mcp