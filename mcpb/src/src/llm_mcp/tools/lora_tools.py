"""
LoRA Tools for LLM MCP

This module provides tools for managing LoRA (Low-Rank Adaptation) adapters
for language models, including loading, unloading, and listing available adapters.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import json

# Try to import PEFT (for LoRA support)
try:
    from peft import LoraConfig, get_peft_model, PeftModel, PeftConfig
    from peft.utils import WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)

# Default directory for LoRA adapters
DEFAULT_LORA_DIR = os.path.expanduser("~/.cache/llm-mcp/loras")

class LoraManager:
    """Manages LoRA adapters for language models."""
    
    def __init__(self, base_model=None, lora_dir: str = None):
        """Initialize the LoRA manager.
        
        Args:
            base_model: The base model to apply adapters to
            lora_dir: Directory containing LoRA adapters
        """
        self.base_model = base_model
        self.lora_dir = Path(lora_dir) if lora_dir else Path(DEFAULT_LORA_DIR)
        self.lora_dir.mkdir(parents=True, exist_ok=True)
        self.loaded_adapters: Dict[str, Any] = {}
        
        # Create default config if it doesn't exist
        self.config_file = self.lora_dir / "config.json"
        if not self.config_file.exists():
            self._save_config()
    
    def _save_config(self):
        """Save the current configuration to disk."""
        config = {
            "default_lora_dir": str(self.lora_dir),
            "loaded_adapters": {
                name: {
                    "base_model": adapter.get("base_model"),
                    "path": str(adapter.get("path")),
                    "config": adapter.get("config", {})
                }
                for name, adapter in self.loaded_adapters.items()
            }
        }
        
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2)
    
    def _load_config(self):
        """Load configuration from disk."""
        if not self.config_file.exists():
            return
            
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
                
            # Update loaded adapters
            self.loaded_adapters = config.get("loaded_adapters", {})
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA config: {e}")
    
    def set_base_model(self, model):
        """Set the base model for applying adapters.
        
        Args:
            model: The base model to apply adapters to
        """
        self.base_model = model
        return {"status": "success", "message": "Base model set"}
    
    def list_available_adapters(self) -> List[Dict[str, Any]]:
        """List all available LoRA adapters.
        
        Returns:
            List of adapter information dictionaries
        """
        if not self.lora_dir.exists():
            return []
            
        adapters = []
        
        # Look for adapter directories
        for adapter_dir in self.lora_dir.iterdir():
            if not adapter_dir.is_dir():
                continue
                
            # Check for adapter config
            config_path = adapter_dir / "adapter_config.json"
            if not config_path.exists():
                continue
                
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    
                # Get adapter files
                adapter_files = list(adapter_dir.glob(f"*.bin")) + list(adapter_dir.glob(f"*.safetensors"))
                
                adapters.append({
                    "name": adapter_dir.name,
                    "path": str(adapter_dir),
                    "base_model": config.get("base_model_name_or_path", "unknown"),
                    "r": config.get("r", 0),
                    "alpha": config.get("lora_alpha", 0),
                    "target_modules": config.get("target_modules", []),
                    "files": [str(f.name) for f in adapter_files]
                })
                
            except Exception as e:
                logger.warning(f"Error loading adapter {adapter_dir.name}: {e}")
        
        return adapters
    
    def load_adapter(
        self,
        adapter_name: str,
        adapter_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter to load
            adapter_path: Path to the adapter directory (default: lora_dir/adapter_name)
            **kwargs: Additional arguments for PeftModel.from_pretrained
            
        Returns:
            Dictionary with status and adapter information
        """
        if not PEFT_AVAILABLE:
            return {
                "status": "error",
                "message": "PEFT library is not installed. Install with: pip install peft"
            }
            
        if not self.base_model:
            return {
                "status": "error",
                "message": "No base model set. Call set_base_model() first."
            }
            
        try:
            # If no path provided, use the default directory
            if adapter_path is None:
                adapter_path = str(self.lora_dir / adapter_name)
            
            # Load the adapter
            model = PeftModel.from_pretrained(
                self.base_model,
                adapter_path,
                adapter_name=adapter_name,
                **kwargs
            )
            
            # Store adapter info
            config = PeftConfig.from_pretrained(adapter_path)
            self.loaded_adapters[adapter_name] = {
                "model": model,
                "config": config.to_dict(),
                "path": adapter_path
            }
            
            # Save updated config
            self._save_config()
            
            return {
                "status": "success",
                "adapter": adapter_name,
                "config": config.to_dict()
            }
            
        except Exception as e:
            logger.error(f"Failed to load adapter {adapter_name}: {e}")
            return {
                "status": "error",
                "message": f"Failed to load adapter: {str(e)}"
            }
    
    def unload_adapter(self, adapter_name: str) -> Dict[str, Any]:
        """Unload a LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter to unload
            
        Returns:
            Status dictionary
        """
        if adapter_name in self.loaded_adapters:
            del self.loaded_adapters[adapter_name]
            self._save_config()
            return {"status": "success", "message": f"Adapter {adapter_name} unloaded"}
        return {"status": "error", "message": f"Adapter {adapter_name} not found"}
    
    def get_loaded_adapters(self) -> Dict[str, Any]:
        """Get information about loaded adapters.
        
        Returns:
            Dictionary of loaded adapters and their configurations
        """
        return {
            name: {
                "base_model": adapter.get("base_model"),
                "path": adapter.get("path"),
                "config": adapter.get("config", {})
            }
            for name, adapter in self.loaded_adapters.items()
        }

# Global instance (internal use only)
_lora_manager = LoraManager()

# Implementation functions

async def _lora_list_adapters_impl(adapter_dir: str = None) -> List[Dict[str, Any]]:
    """Implementation of lora_list_adapters.
    
    Args:
        adapter_dir: Directory containing LoRA adapters (default: ~/.cache/llm-mcp/loras)
        
    Returns:
        List of adapter information dictionaries
    """
    if adapter_dir:
        _lora_manager.lora_dir = Path(adapter_dir)
    return _lora_manager.list_available_adapters()


async def _lora_load_adapter_impl(
    adapter_name: str,
    adapter_path: str = None,
    base_model: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Implementation of lora_load_adapter.
    
    Args:
        adapter_name: Name to give to the loaded adapter
        adapter_path: Path to the adapter directory (default: lora_dir/adapter_name)
        base_model: Base model name or path (if not already set)
        **kwargs: Additional arguments for PeftModel.from_pretrained
        
    Returns:
        Status dictionary with adapter information
    """
    if base_model:
        _lora_manager.set_base_model(base_model)
    return _lora_manager.load_adapter(adapter_name, adapter_path, **kwargs)


async def _lora_unload_adapter_impl(adapter_name: str) -> Dict[str, Any]:
    """Implementation of lora_unload_adapter.
    
    Args:
        adapter_name: Name of the adapter to unload
        
    Returns:
        Status dictionary
    """
    return _lora_manager.unload_adapter(adapter_name)


async def _lora_list_loaded_impl() -> Dict[str, Any]:
    """Implementation of lora_list_loaded.
    
    Returns:
        Dictionary of loaded adapters and their configurations
    """
    return _lora_manager.get_loaded_adapters()


def register_lora_tools(mcp):
    """Register all LoRA-related tools with the MCP server.
    
    Args:
        mcp: The MCP server instance with tool decorator
        
    Returns:
        The MCP server instance with LoRA tools registered
    """
    if not PEFT_AVAILABLE:
        logger.warning("PEFT is not installed. LoRA tools will not be available.")
        return mcp
    
    @mcp.tool()
    async def lora_list_adapters(adapter_dir: str = None) -> List[Dict[str, Any]]:
        """List all available LoRA adapters.
        
        Args:
            adapter_dir: Directory containing LoRA adapters (default: ~/.cache/llm-mcp/loras)
            
        Returns:
            List of adapter information dictionaries
        """
        return await _lora_list_adapters_impl(adapter_dir)
    
    @mcp.tool()
    async def lora_load_adapter(
        adapter_name: str,
        adapter_path: str = None,
        base_model: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Load a LoRA adapter for the current model.
        
        Args:
            adapter_name: Name to give to the loaded adapter
            adapter_path: Path to the adapter directory (default: lora_dir/adapter_name)
            base_model: Base model name or path (if not already set)
            **kwargs: Additional arguments for PeftModel.from_pretrained
            
        Returns:
            Status dictionary with adapter information
        """
        return await _lora_load_adapter_impl(adapter_name, adapter_path, base_model, **kwargs)
    
    @mcp.tool()
    async def lora_unload_adapter(adapter_name: str) -> Dict[str, Any]:
        """Unload a LoRA adapter.
        
        Args:
            adapter_name: Name of the adapter to unload
            
        Returns:
            Status dictionary
        """
        return await _lora_unload_adapter_impl(adapter_name)
    
    @mcp.tool()
    async def lora_list_loaded() -> Dict[str, Any]:
        """List all currently loaded LoRA adapters.
        
        Returns:
            Dictionary of loaded adapters and their configurations
        """
        return await _lora_list_loaded_impl()
    
    return mcp
