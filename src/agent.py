"""ADK discovery entrypoint for the `src` app."""

from src.adk_agent.web_agent import build_root_agent

root_agent = build_root_agent()
