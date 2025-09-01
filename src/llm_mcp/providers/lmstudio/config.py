"""Configuration for LM Studio provider."""
from typing import Dict, Any, Optional
from pydantic import Field

from ...models.base import BaseProviderConfig

class LMStudioConfig(BaseProviderConfig):
    """Configuration for LM Studio provider.
    
    Attributes:
        base_url: Base URL for the LM Studio API (default: http://localhost:1234)
        timeout: Request timeout in seconds (default: 60)
    """
    base_url: str = Field(
        default="http://localhost:1234",
        description="Base URL for the LM Studio API"
    )
    timeout: int = Field(
        default=60,
        description="Request timeout in seconds",
        ge=1,
        le=300
    )
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LMStudioConfig':
        """Create an LMStudioConfig from a dictionary.
        
        Args:
            config_dict: Dictionary containing configuration
            
        Returns:
            LMStudioConfig instance
        """
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the config to a dictionary.
        
        Returns:
            Dictionary representation of the config
        """
        return self.dict(exclude_unset=True, exclude_none=True)
