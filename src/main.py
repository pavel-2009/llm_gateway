"""Compatibility module for running the ASGI app from the source tree."""

from llm_gateway.app import app, gateway

__all__ = ["app", "gateway"]
