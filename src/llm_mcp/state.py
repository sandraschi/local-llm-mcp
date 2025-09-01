"""Enhanced state management for LLM MCP server - FIXED with persistence and cleanup.

This module implements comprehensive state management with session persistence,
cleanup mechanisms, and monitoring capabilities for FastMCP 2.12+.
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging
import time
import json
import asyncio
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)

@dataclass
class ModelState:
    """Represents the state of a loaded model with enhanced tracking."""
    model_name: str
    model_type: str
    provider: str
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    usage_count: int = 0
    total_tokens_generated: int = 0
    average_tokens_per_second: float = 0.0
    memory_usage_mb: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update_usage(self, tokens_generated: int, duration_seconds: float):
        """Update usage statistics."""
        self.usage_count += 1
        self.total_tokens_generated += tokens_generated
        self.last_used = time.time()
        
        if duration_seconds > 0:
            current_tps = tokens_generated / duration_seconds
            # Exponential moving average for TPS
            if self.average_tokens_per_second == 0:
                self.average_tokens_per_second = current_tps
            else:
                alpha = 0.3  # Smoothing factor
                self.average_tokens_per_second = (
                    alpha * current_tps + (1 - alpha) * self.average_tokens_per_second
                )

@dataclass
class SessionState:
    """Enhanced session state with conversation history and context."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    context: Dict[str, Any] = field(default_factory=dict)
    messages: List[Dict[str, Any]] = field(default_factory=list)
    total_tokens_used: int = 0
    model_preference: Optional[str] = None
    
    def add_message(self, role: str, content: str, model: str = None, tokens: int = 0, **metadata):
        """Add a message to the conversation history."""
        message = {
            'role': role,
            'content': content,
            'timestamp': time.time(),
            'model': model,
            'tokens': tokens,
            **metadata
        }
        self.messages.append(message)
        self.total_tokens_used += tokens
        self.last_active = time.time()
        
        # Keep only last 50 messages to prevent memory bloat
        if len(self.messages) > 50:
            self.messages = self.messages[-50:]
    
    def get_context_summary(self) -> Dict[str, Any]:
        """Get a summary of the session context."""
        return {
            'session_id': self.session_id,
            'message_count': len(self.messages),
            'total_tokens': self.total_tokens_used,
            'duration_minutes': (time.time() - self.created_at) / 60,
            'last_active': self.last_active,
            'model_preference': self.model_preference
        }

class StateManager:
    """Enhanced state manager with persistence and monitoring."""
    
    def __init__(self, persistence_dir: Optional[Path] = None):
        self.models: Dict[str, ModelState] = {}
        self.sessions: Dict[str, SessionState] = {}
        
        # Persistence configuration
        self.persistence_dir = persistence_dir or Path("data/state")
        self.persistence_dir.mkdir(parents=True, exist_ok=True)
        self.models_file = self.persistence_dir / "models.json"
        self.sessions_file = self.persistence_dir / "sessions.json"
        
        # Cleanup configuration
        self.max_inactive_session_hours = 24
        self.max_sessions = 1000
        self.cleanup_interval_minutes = 60
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start_background_tasks(self):
        """Start background cleanup and monitoring tasks."""
        if not self._running:
            self._running = True
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
            logger.info("State manager background tasks started")
    
    async def stop_background_tasks(self):
        """Stop background tasks and save state."""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        await self.save_state()
        logger.info("State manager background tasks stopped")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of inactive sessions and old data."""
        while self._running:
            try:
                await asyncio.sleep(self.cleanup_interval_minutes * 60)
                
                # Cleanup inactive sessions
                cleaned_sessions = self.cleanup_inactive_sessions(
                    max_inactive_seconds=self.max_inactive_session_hours * 3600
                )
                
                # Limit total sessions
                if len(self.sessions) > self.max_sessions:
                    oldest_sessions = sorted(
                        self.sessions.items(), 
                        key=lambda x: x[1].last_active
                    )
                    to_remove = len(self.sessions) - self.max_sessions
                    for session_id, _ in oldest_sessions[:to_remove]:
                        del self.sessions[session_id]
                    logger.info("Removed old sessions", count=to_remove)
                
                # Save state periodically
                await self.save_state()
                
                if cleaned_sessions > 0:
                    logger.info("Periodic cleanup completed", cleaned_sessions=cleaned_sessions)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in periodic cleanup", error=str(e))
    
    async def load_state(self) -> bool:
        """Load state from persistent storage."""
        try:
            # Load models
            if self.models_file.exists():
                with open(self.models_file, 'r') as f:
                    models_data = json.load(f)
                    for model_id, data in models_data.items():
                        self.models[model_id] = ModelState(**data)
                logger.info("Loaded model state", count=len(self.models))
            
            # Load sessions (only recent ones)
            if self.sessions_file.exists():
                with open(self.sessions_file, 'r') as f:
                    sessions_data = json.load(f)
                    cutoff_time = time.time() - (24 * 3600)  # 24 hours ago
                    
                    for session_id, data in sessions_data.items():
                        if data.get('last_active', 0) > cutoff_time:
                            self.sessions[session_id] = SessionState(**data)
                
                logger.info("Loaded session state", count=len(self.sessions))
            
            return True
            
        except Exception as e:
            logger.error("Failed to load state", error=str(e))
            return False
    
    async def save_state(self) -> bool:
        """Save current state to persistent storage."""
        try:
            # Save models
            models_data = {}
            for model_id, model_state in self.models.items():
                models_data[model_id] = {
                    'model_name': model_state.model_name,
                    'model_type': model_state.model_type,
                    'provider': model_state.provider,
                    'loaded_at': model_state.loaded_at,
                    'last_used': model_state.last_used,
                    'usage_count': model_state.usage_count,
                    'total_tokens_generated': model_state.total_tokens_generated,
                    'average_tokens_per_second': model_state.average_tokens_per_second,
                    'memory_usage_mb': model_state.memory_usage_mb,
                    'metadata': model_state.metadata
                }
            
            with open(self.models_file, 'w') as f:
                json.dump(models_data, f, indent=2)
            
            # Save sessions (only active ones)
            sessions_data = {}
            cutoff_time = time.time() - (24 * 3600)  # Keep 24 hours of sessions
            
            for session_id, session_state in self.sessions.items():
                if session_state.last_active > cutoff_time:
                    sessions_data[session_id] = {
                        'session_id': session_state.session_id,
                        'created_at': session_state.created_at,
                        'last_active': session_state.last_active,
                        'context': session_state.context,
                        'messages': session_state.messages[-10:],  # Keep last 10 messages
                        'total_tokens_used': session_state.total_tokens_used,
                        'model_preference': session_state.model_preference
                    }
            
            with open(self.sessions_file, 'w') as f:
                json.dump(sessions_data, f, indent=2)
            
            logger.debug("State saved to disk", models=len(models_data), sessions=len(sessions_data))
            return True
            
        except Exception as e:
            logger.error("Failed to save state", error=str(e))
            return False
        
    def register_model(self, model_name: str, model_type: str, provider: str = "unknown", **metadata) -> ModelState:
        """Register a new model in the state manager."""
        model_id = f"{provider}:{model_name}"
        
        if model_id in self.models:
            logger.warning("Model already registered", model_id=model_id)
            return self.models[model_id]
            
        model_state = ModelState(
            model_name=model_name,
            model_type=model_type,
            provider=provider,
            metadata=metadata
        )
        self.models[model_id] = model_state
        logger.info("Model registered", model_id=model_id, model_type=model_type, provider=provider)
        return model_state
        
    def unregister_model(self, model_name: str, provider: str = "unknown") -> bool:
        """Unregister a model from the state manager."""
        model_id = f"{provider}:{model_name}"
        if model_id in self.models:
            del self.models[model_id]
            logger.info("Model unregistered", model_id=model_id)
            return True
        return False
        
    def get_model_state(self, model_name: str, provider: str = "unknown") -> Optional[ModelState]:
        """Get the state of a registered model."""
        model_id = f"{provider}:{model_name}"
        if model_id in self.models:
            model_state = self.models[model_id]
            model_state.last_used = time.time()
            return model_state
        return None
    
    def update_model_usage(self, model_name: str, provider: str, tokens_generated: int, duration_seconds: float):
        """Update model usage statistics."""
        model_state = self.get_model_state(model_name, provider)
        if model_state:
            model_state.update_usage(tokens_generated, duration_seconds)
        
    def create_session(self, session_id: str, **context) -> SessionState:
        """Create a new user session."""
        if session_id in self.sessions:
            logger.warning("Session already exists", session_id=session_id)
            return self.sessions[session_id]
            
        session = SessionState(session_id=session_id, context=context)
        self.sessions[session_id] = session
        logger.info("Session created", session_id=session_id)
        return session
        
    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Get an existing session."""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.last_active = time.time()
            return session
        return None
        
    def end_session(self, session_id: str) -> bool:
        """End a user session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("Session ended", session_id=session_id)
            return True
        return False
        
    def cleanup_inactive_sessions(self, max_inactive_seconds: int = 3600) -> int:
        """Clean up sessions that have been inactive for too long."""
        now = time.time()
        inactive_sessions = []
        
        for session_id, session in self.sessions.items():
            if now - session.last_active > max_inactive_seconds:
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.sessions[session_id]
            
        if inactive_sessions:
            logger.info("Cleaned up inactive sessions", count=len(inactive_sessions))
            
        return len(inactive_sessions)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        now = time.time()
        
        # Model statistics
        model_stats = {}
        total_tokens = 0
        total_requests = 0
        
        for model_id, model_state in self.models.items():
            total_tokens += model_state.total_tokens_generated
            total_requests += model_state.usage_count
            
            model_stats[model_id] = {
                'usage_count': model_state.usage_count,
                'total_tokens': model_state.total_tokens_generated,
                'avg_tokens_per_second': model_state.average_tokens_per_second,
                'last_used': model_state.last_used,
                'uptime_hours': (now - model_state.loaded_at) / 3600
            }
        
        # Session statistics
        active_sessions = len([s for s in self.sessions.values() if now - s.last_active < 3600])
        total_session_tokens = sum(s.total_tokens_used for s in self.sessions.values())
        
        return {
            'models': {
                'total_models': len(self.models),
                'total_requests': total_requests,
                'total_tokens_generated': total_tokens,
                'model_details': model_stats
            },
            'sessions': {
                'total_sessions': len(self.sessions),
                'active_sessions': active_sessions,
                'total_session_tokens': total_session_tokens,
            },
            'system': {
                'uptime_seconds': now - getattr(self, '_start_time', now),
                'last_cleanup': getattr(self, '_last_cleanup', None)
            }
        }
    
    async def cleanup(self):
        """Cleanup method for graceful shutdown."""
        await self.stop_background_tasks()
        logger.info("State manager cleanup completed")

# Global state manager instance
state_manager = StateManager()
