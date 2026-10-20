"""Shared Playwright runtime used by ADK tools.

The runtime is intentionally module-global so tool calls in the same process
can share a browser context and current page.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from playwright.async_api import Browser, BrowserContext, Page, async_playwright


@dataclass
class BrowserRuntime:
    browser: Optional[Browser] = None
    context: Optional[BrowserContext] = None
    page: Optional[Page] = None
    pw = None
    last_elements: list[dict] = None  # type: ignore[assignment]
    capture_index: int = 0
    last_screenshot_path: Optional[Path] = None
    action_history: list[str] = None  # type: ignore[assignment]
    visited_urls: set[str] = None  # type: ignore[assignment]
    observation_streak: int = 0
    no_progress_streak: int = 0
    last_action_name: str = ""
    search_loop_streak: int = 0
    last_search_query: str = ""
    answer_locked: bool = False
    recent_actions: list[str] = None  # type: ignore[assignment]
    churn_break_count: int = 0
    session_done: bool = False
    required_domains: set[str] = None  # type: ignore[assignment]
    focus_domain: str = ""
    wrong_domain_detour_count: int = 0  # times agent landed on non-required domain
    task_constraints: dict = None  # type: ignore[assignment]
    task_constraints_meta: dict = None  # type: ignore[assignment]
    _last_user_task_fp: Optional[str] = None

    async def start(self) -> None:
        if self.page:
            return
        self.pw = await async_playwright().start()
        headless = os.getenv("BROWSER_HEADLESS", "false").lower() == "true"
        self.browser = await self.pw.chromium.launch(headless=headless)
        self.context = await self.browser.new_context(
            viewport={"width": 1440, "height": 900},
            locale="en-US",
            timezone_id="America/New_York",
            extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
        )
        self.page = await self.context.new_page()
        if self.action_history is None:
            self.action_history = []
        if self.visited_urls is None:
            self.visited_urls = set()
        if self.task_constraints is None:
            self.task_constraints = {}
        if self.task_constraints_meta is None:
            self.task_constraints_meta = {}

    async def ensure_page(self) -> Page:
        if not self.page:
            await self.start()
        assert self.page is not None
        return self.page

    def screenshot_path(self) -> Path:
        output_dir = Path("artifacts")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.capture_index += 1
        path = output_dir / f"page_{self.capture_index}.png"
        self.last_screenshot_path = path
        return path

    async def close(self) -> None:
        if self.context:
            await self.context.close()
        if self.browser:
            await self.browser.close()
        if self.pw:
            await self.pw.stop()
        self.browser = None
        self.context = None
        self.page = None
        self.pw = None
        self.last_elements = None
        self.capture_index = 0
        self.last_screenshot_path = None
        self.action_history = []
        self.visited_urls = set()
        self.observation_streak = 0
        self.no_progress_streak = 0
        self.last_action_name = ""
        self.search_loop_streak = 0
        self.last_search_query = ""
        self.answer_locked = False
        self.recent_actions = []
        self.churn_break_count = 0
        self.session_done = False
        self.required_domains = set()
        self.focus_domain = ""
        self.wrong_domain_detour_count = 0
        self.task_constraints = {}
        self.task_constraints_meta = {}
        self._last_observation_key = None
        self._last_observation_result = None
        self._last_user_task_fp = None


RUNTIME = BrowserRuntime()
