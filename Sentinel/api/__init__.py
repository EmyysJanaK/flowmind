"""
API module for Sentinel multi-agent system.
Provides REST and async interfaces to interact with the system.
"""

from .server import create_app

__all__ = ["create_app"]
