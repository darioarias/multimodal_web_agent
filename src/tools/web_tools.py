"""ADK-compatible function tools for multimodal web interaction."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import sys
import typing
import urllib.parse
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageDraw
from playwright.async_api import Error as PlaywrightError
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from src.tools.browser_runtime import RUNTIME

if typing.TYPE_CHECKING:
    from playwright.async_api import Page


async def _evaluate_when_stable(
    page: "Page",
    expression: str,
    arg: Any | None = None,
    *,
    retries: int = 5,
) -> Any:
    """Run ``page.evaluate`` after DOM settles; retry if navigation destroys context."""
    last_exc: BaseException | None = None
    for attempt in range(retries):
        try:
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=8000)
            except Exception:
                pass
            if arg is not None:
                return await page.evaluate(expression, arg)
            return await page.evaluate(expression)
        except PlaywrightError as e:
            last_exc = e
            msg = str(e).lower()
            if (
                "execution context was destroyed" in msg
                or "navigation" in msg
                or "most likely because of a navigation" in msg
            ):
                await asyncio.sleep(0.15 * (attempt + 1))
                continue
            raise
        except Exception as e:
            last_exc = e
            msg = str(e).lower()
            if (
                "execution context was destroyed" in msg
                or "navigation" in msg
                or "most likely because of a navigation" in msg
            ):
                await asyncio.sleep(0.15 * (attempt + 1))
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("_evaluate_when_stable: exhausted retries without exception")


def _append_process_log(record: dict[str, Any]) -> str:
    """Append a JSON line to artifacts/process_log.jsonl for transparency."""
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "process_log.jsonl"
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True) + "\n")
    return str(path.resolve())


async def log_step(
    stage: str,
    thought: str = "",
    observation: str = "",
    action: str = "",
    values: dict[str, Any] | None = None,
) -> str:
    """Log a ReAct step (thought/observation/action/values) to a JSONL file."""
    page = await RUNTIME.ensure_page()
    record: dict[str, Any] = {
        "stage": stage,
        "thought": thought,
        "observation": observation,
        "action": action,
        "values": values or {},
        "url": page.url,
    }
    log_path = _append_process_log(record)
    return f"Logged step to {log_path}"


def _sanitize_url(url: str) -> str:
    """Best-effort sanitization for URLs passed by the LLM/tools."""
    u = (url or "").strip()
    # Strip surrounding quotes if the model included them.
    if (u.startswith('"') and u.endswith('"')) or (
        u.startswith("'") and u.endswith("'")
    ):
        u = u[1:-1].strip()
    # Remove accidental trailing delimiters.
    u = u.rstrip(";").strip()
    return u


def _escape_attr_value(value: str) -> str:
    """Escape an attribute value for safe CSS selector usage."""
    return (value or "").replace("\\", "\\\\").replace('"', '\\"')


def _preferred_selector_for_element(el: dict[str, Any]) -> str | None:
    """Build a best-effort stable CSS selector for an extracted element."""
    if el.get("selector"):
        return str(el["selector"])

    tag = str(el.get("tag") or "").strip() or "*"
    data_testid = str(el.get("data_testid") or "").strip()
    aria_label = str(el.get("aria_label") or "").strip()
    name = str(el.get("name") or "").strip()
    placeholder = str(el.get("placeholder") or "").strip()
    role = str(el.get("role") or "").strip()

    if data_testid:
        return f'{tag}[data-testid="{_escape_attr_value(data_testid)}"]'
    if aria_label:
        return f'{tag}[aria-label="{_escape_attr_value(aria_label)}"]'
    if name:
        return f'{tag}[name="{_escape_attr_value(name)}"]'
    if placeholder:
        return f'{tag}[placeholder="{_escape_attr_value(placeholder)}"]'
    if role:
        return f'{tag}[role="{_escape_attr_value(role)}"]'
    return None


async def _is_unique_visible_locator(page: "Page", selector: str) -> bool:
    """Return True when selector resolves to exactly one visible element."""
    try:
        locator = page.locator(selector)
        count = await locator.count()
        if count != 1:
            return False
        return await locator.first.is_visible()
    except Exception:
        return False


async def _safe_page_title(page) -> str:
    """Read page title safely across in-flight navigations."""
    for _ in range(3):
        try:
            return await page.title()
        except PlaywrightError as exc:
            msg = str(exc)
            if "Execution context was destroyed" not in msg:
                return ""
            try:
                await page.wait_for_load_state("domcontentloaded", timeout=1500)
            except Exception:
                continue
        except Exception:
            return ""
    return ""


async def navigate(url: str) -> str:
    """Open a URL in the controlled browser page."""
    url = _sanitize_url(url)
    page = await RUNTIME.ensure_page()
    await page.goto(url, wait_until="domcontentloaded")
    # Capture immediately for transparency (fused multimodal observation).
    post = json.loads(await capture_multimodal_observation())
    return f"Navigated to {page.url}; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


async def search_web(query: str) -> str:
    """Search web with Google-first retries, then fallback to DuckDuckGo."""
    q = (query or "").strip()
    if not q:
        return "search_web failed: empty query"

    page = await RUNTIME.ensure_page()

    async def _google_search(qq: str) -> tuple[bool, str]:
        google_url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(qq)
        await page.goto(google_url, wait_until="domcontentloaded")
        content = await page.content()
        blocked = (
            "400. That’s an error." in content
            or "malformed or illegal request" in content
            or "/sorry/" in page.url
            or "Our systems have detected unusual traffic" in content
        )
        return (not blocked), page.url

    search_route = "google_success"
    attempts = []
    ok, url = await _google_search(q)
    attempts.append({"provider": "google", "query": q, "ok": ok, "url": url})

    # Retry Google with a simplified query before fallback.
    if not ok:
        tokens = [t for t in q.split() if len(t) > 2]
        simplified = " ".join(tokens[:8]) or q
        if simplified != q:
            ok2, url2 = await _google_search(simplified)
            attempts.append(
                {"provider": "google", "query": simplified, "ok": ok2, "url": url2}
            )
            ok = ok2
        if not ok:
            search_route = "fallback_ddg"
            ddg_url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(q)
            await page.goto(ddg_url, wait_until="domcontentloaded")
    elif len(attempts) > 1:
        search_route = "google_retry_success"

    # Best-effort open of first organic result so the agent does not stall on search pages.
    first_result_url = await _evaluate_when_stable(
        page,
        """
        () => {
          const candidates = [
            // Google
            'a h3',
            '#search a',
            // DuckDuckGo
            'a[data-testid="result-title-a"]',
            'article h2 a',
            // Generic SERP fallback
            'main a'
          ];
          const bad = ['google.com/search', 'duckduckgo.com/?q=', '/settings', '/preferences', '/help'];
          for (const sel of candidates) {
            const nodes = Array.from(document.querySelectorAll(sel));
            for (const node of nodes) {
              const a = node.tagName && node.tagName.toLowerCase() === 'a' ? node : node.closest('a');
              if (!a) continue;
              const href = (a.getAttribute('href') || '').trim();
              if (!href) continue;
              if (href.startsWith('/')) continue;
              if (!/^https?:\\/\\//.test(href)) continue;
              const lower = href.toLowerCase();
              if (bad.some((b) => lower.includes(b))) continue;
              return href;
            }
          }
          return null;
        }
        """,
    )
    if first_result_url:
        try:
            await page.goto(first_result_url, wait_until="domcontentloaded")
        except Exception:
            pass

    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "search",
            "url": page.url,
            "query": q,
            "route": search_route,
            "attempts": attempts,
            "opened_result_url": first_result_url or "",
        }
    )
    return (
        f"Searched for '{q}'; route={search_route}; url={page.url}; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def click(selector: str) -> str:
    """Click an element using CSS selector."""
    page = await RUNTIME.ensure_page()
    target = page.locator(selector).first
    try:
        await target.wait_for(state="visible", timeout=15000)
        await target.scroll_into_view_if_needed(timeout=5000)
        await target.click(timeout=10000)
        return f"Clicked {selector}"
    except PlaywrightTimeoutError:
        return f"Click failed: selector not found/visible within timeout: {selector}"


async def type_text(selector: str, text: str, press_enter: bool = False) -> str:
    """Fill text into an input-like element and optionally press Enter."""
    page = await RUNTIME.ensure_page()
    target = page.locator(selector).first
    try:
        await target.wait_for(state="visible", timeout=15000)
        await target.scroll_into_view_if_needed(timeout=5000)
        # Click to focus, then fill for robustness on dynamic sites
        await target.click(timeout=5000)
        await target.fill(text, timeout=15000)
        if press_enter:
            await target.press("Enter")
        return f"Typed into {selector}"
    except PlaywrightTimeoutError:
        return f"Type failed: selector not found/visible within timeout: {selector}"


async def scroll_by(pixels: int = 800) -> str:
    """Scroll page vertically by a given pixel amount."""
    page = await RUNTIME.ensure_page()
    await page.mouse.wheel(0, pixels)
    post = json.loads(await capture_multimodal_observation())
    return f"Scrolled by {pixels}px; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


def _simple_element_id(idx: int) -> str:
    """Simple numeric ID label, as in WebVoyager-style screenshots: 0,1,2,..."""
    return str(idx)


def _element_action_kind(item: dict[str, Any]) -> str:
    """Infer high-level action kind for LLM-facing element text."""
    tag = str(item.get("tag") or "").lower()
    role = str(item.get("role") or "").lower()
    input_type = str(item.get("input_type") or "").lower()
    if tag in {"textarea", "select"}:
        return "Input" if tag == "textarea" else "Select"
    if tag == "input":
        if input_type in {"checkbox", "radio"}:
            return "Select"
        if input_type in {"button", "submit", "reset"}:
            return "Click"
        return "Input"
    if role in {"textbox", "combobox", "searchbox"}:
        return "Input"
    if role in {"option", "menuitem", "tab", "checkbox", "radio", "switch", "gridcell"}:
        return "Select"
    return "Click"


def _element_reasoning_label(item: dict[str, Any]) -> str:
    """Build concise human-readable label for action selection."""
    for key in ("aria_label", "placeholder", "name", "text"):
        value = str(item.get(key) or "").strip()
        if value:
            return " ".join(value.split())[:120]
    tag = str(item.get("tag") or "element")
    return f"{tag} element"


def _annotate_screenshot(screenshot_path: Path, elements: list[dict[str, Any]]) -> str:
    annotated_path = screenshot_path.parent / f"{screenshot_path.stem}_labeled.png"
    image = Image.open(screenshot_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    image_w, image_h = image.size

    for item in elements:
        x = int(item.get("x", 0))
        y = int(item.get("y", 0))
        width = max(1, int(item.get("width", 1)))
        height = max(1, int(item.get("height", 1)))
        label = str(item.get("id", "element"))

        x2 = x + width
        y2 = y + height
        # Clip to viewport image bounds to prevent off-canvas labels.
        if x2 <= 0 or y2 <= 0 or x >= image_w or y >= image_h:
            continue
        x = max(0, x)
        y = max(0, y)
        x2 = min(image_w - 1, x2)
        y2 = min(image_h - 1, y2)
        draw.rectangle([(x, y), (x2, y2)], outline=(0, 0, 0), width=2)
        # Compact label box for numeric IDs
        draw.rectangle([(x, max(0, y - 16)), (x + 18, y)], fill=(0, 0, 0))
        draw.text((x + 4, max(0, y - 15)), label, fill=(255, 255, 255))

    image.save(annotated_path)
    return str(annotated_path.resolve())


async def _collect_dom_elements(page) -> list[dict[str, Any]]:
    dom_data: list[dict[str, Any]] = await _evaluate_when_stable(
        page,
        """
        () => {
          const vw = window.innerWidth || document.documentElement.clientWidth || 0;
          const vh = window.innerHeight || document.documentElement.clientHeight || 0;
          const MIN_AREA = 24;
          const MAX_TOTAL = 200;
          const GRID_COLS = 3;
          const GRID_ROWS = 3;
          const MAX_PER_REGION = 18;
          const MAX_BOX_IOU = 0.75;

          const roleAllow = new Set([
            'button',
            'link',
            'textbox',
            'combobox',
            'option',
            'gridcell',
            'menuitem',
            'tab',
            'checkbox',
            'radio',
            'switch',
            'slider',
            'spinbutton'
          ]);
          const tagAllow = new Set(['a', 'button', 'input', 'textarea', 'select', 'option', 'summary']);

          const selectors = [
            'a',
            'button',
            'input',
            'textarea',
            'select',
            'option',
            'summary',
            '[role]',
            '[onclick]',
            '[contenteditable="true"]',
            '[data-testid]',
            '[aria-label]',
            '[aria-haspopup]',
            '[tabindex]:not([tabindex="-1"])'
          ].join(', ');

          const isVisible = (el) => {
            if (!el) return false;
            const style = window.getComputedStyle(el);
            if (!style || style.display === 'none' || style.visibility === 'hidden') return false;
            if (style.pointerEvents === 'none') return false;
            const rect = el.getBoundingClientRect();
            return rect.width > 0 && rect.height > 0;
          };

          const isLikelyChromeOrAd = (el, rect, text) => {
            const idCls = `${el.id || ''} ${el.className || ''}`.toLowerCase();
            const badTokens = [
              'cookie',
              'consent',
              'banner',
              'advert',
              'ads',
              'sponsored',
              'promo',
              'popup',
              'newsletter',
              'subscribe',
              'footer',
              'header',
              'navbar',
              'breadcrumb'
            ];
            if (badTokens.some((t) => idCls.includes(t))) return true;

            // Very wide top/bottom strips are often site chrome.
            const nearTop = rect.top <= 56;
            const nearBottom = rect.bottom >= vh - 40;
            const veryWide = rect.width >= vw * 0.86;
            const lowInfo = (text || '').trim().length < 12;
            if ((nearTop || nearBottom) && veryWide && lowInfo) return true;
            return false;
          };

          const intersectsViewport = (rect) => {
            const right = rect.x + rect.width;
            const bottom = rect.y + rect.height;
            return right > 0 && bottom > 0 && rect.x < vw && rect.y < vh;
          };

          const centerHitTest = (el, rect) => {
            const cx = Math.min(Math.max(rect.left + rect.width / 2, 0), Math.max(vw - 1, 0));
            const cy = Math.min(Math.max(rect.top + rect.height / 2, 0), Math.max(vh - 1, 0));
            const atCenter = document.elementFromPoint(cx, cy);
            return !!atCenter && (atCenter === el || el.contains(atCenter) || atCenter.contains(el));
          };

          const nodeList = Array.from(document.querySelectorAll(selectors)).slice(0, 1000);
          const candidates = [];
          for (const el of nodeList) {
            if (!isVisible(el)) continue;
            const rect = el.getBoundingClientRect();
            if (!intersectsViewport(rect)) continue;
            const area = rect.width * rect.height;
            if (area < MIN_AREA) continue;
            if (!centerHitTest(el, rect)) continue;

            const tag = (el.tagName || '').toLowerCase();
            const role = (el.getAttribute('role') || '').trim().toLowerCase();
            const name = (el.getAttribute('name') || '').trim();
            const type = (el.getAttribute('type') || '').trim();
            const ariaLabel = (el.getAttribute('aria-label') || '').trim();
            const placeholder = (el.getAttribute('placeholder') || '').trim();
            const dataTestId = (el.getAttribute('data-testid') || '').trim();
            const disabled = !!el.disabled || el.getAttribute('aria-disabled') === 'true';
            const text = (el.innerText || el.value || ariaLabel || '').trim().replace(/\\s+/g, ' ');
            const hasInteractiveHint =
              tagAllow.has(tag) ||
              roleAllow.has(role) ||
              !!el.onclick ||
              (el.getAttribute('contenteditable') === 'true') ||
              ['button', 'submit', 'text', 'search', 'email', 'password', 'tel', 'date', 'time', 'datetime-local'].includes(type) ||
              !!ariaLabel ||
              !!dataTestId;
            if (!hasInteractiveHint || disabled) continue;
            if (isLikelyChromeOrAd(el, rect, text)) continue;

            // For anchors, keep those that are likely actionable navigation.
            if (tag === 'a') {
              const href = (el.getAttribute('href') || '').trim().toLowerCase();
              if (!href || href === '#' || href.startsWith('javascript:')) continue;
            }

            candidates.push({
              element: el,
              tag,
              role,
              name,
              input_type: type,
              aria_label: ariaLabel,
              placeholder,
              data_testid: dataTestId,
              text,
              selector: el.id ? `#${el.id}` : null,
              disabled,
              x: Math.round(rect.x),
              y: Math.round(rect.y),
              width: Math.round(rect.width),
              height: Math.round(rect.height),
              area,
              viewport_width: Math.round(vw),
              viewport_height: Math.round(vh),
              clickable_score:
                (tag === 'button' ? 5 : 0) +
                (tag === 'input' ? 4 : 0) +
                (tag === 'select' ? 4 : 0) +
                (tag === 'textarea' ? 4 : 0) +
                (tag === 'a' ? 3 : 0) +
                (roleAllow.has(role) ? 3 : 0) +
                (dataTestId ? 2 : 0) +
                (ariaLabel ? 1 : 0) +
                (text ? 1 : 0),
            });
          }

          // Prefer leaf actionable elements: remove candidates that only wrap
          // another actionable candidate and have similar bounds.
          const deduped = candidates.filter((item, idx) => {
            return !candidates.some((other, j) => {
              if (idx === j) return false;
              if (!item.element.contains(other.element)) return false;
              const overlapW = Math.min(item.x + item.width, other.x + other.width) - Math.max(item.x, other.x);
              const overlapH = Math.min(item.y + item.height, other.y + other.height) - Math.max(item.y, other.y);
              const overlap = Math.max(0, overlapW) * Math.max(0, overlapH);
              const minArea = Math.max(1, Math.min(item.area, other.area));
              // If mostly overlapping and parent is much larger, keep child only.
              return overlap / minArea > 0.8 && item.area > other.area * 1.15;
            });
          });

          // Region cap: avoid flooding with many elements from one local area.
          deduped.sort((a, b) => {
            if (b.clickable_score !== a.clickable_score) return b.clickable_score - a.clickable_score;
            return b.area - a.area;
          });
          const regionCounts = {};
          const finalItems = [];
          const iou = (a, b) => {
            const left = Math.max(a.x, b.x);
            const top = Math.max(a.y, b.y);
            const right = Math.min(a.x + a.width, b.x + b.width);
            const bottom = Math.min(a.y + a.height, b.y + b.height);
            const iw = Math.max(0, right - left);
            const ih = Math.max(0, bottom - top);
            const inter = iw * ih;
            if (inter <= 0) return 0;
            const union = a.area + b.area - inter;
            return union > 0 ? inter / union : 0;
          };
          for (const item of deduped) {
            const cx = item.x + item.width / 2;
            const cy = item.y + item.height / 2;
            const col = Math.max(0, Math.min(GRID_COLS - 1, Math.floor((cx / Math.max(vw, 1)) * GRID_COLS)));
            const row = Math.max(0, Math.min(GRID_ROWS - 1, Math.floor((cy / Math.max(vh, 1)) * GRID_ROWS)));
            const key = `${row}:${col}`;
            const used = regionCounts[key] || 0;
            if (used >= MAX_PER_REGION) continue;
            // Overlap suppression: skip near-duplicate boxes.
            if (finalItems.some((kept) => iou(kept, item) > MAX_BOX_IOU)) continue;
            regionCounts[key] = used + 1;
            finalItems.push(item);
            if (finalItems.length >= MAX_TOTAL) break;
          }

          return finalItems.map((item) => ({
            tag: item.tag,
            selector: item.selector,
            text: item.text,
            role: item.role,
            name: item.name,
            input_type: item.input_type,
            aria_label: item.aria_label,
            placeholder: item.placeholder,
            data_testid: item.data_testid,
            disabled: item.disabled,
            x: item.x,
            y: item.y,
            width: item.width,
            height: item.height,
            viewport_width: item.viewport_width,
            viewport_height: item.viewport_height,
          }));
        }
        """,
    )
    for idx, item in enumerate(dom_data):
        item["id"] = _simple_element_id(idx)
    return dom_data


async def capture_observation() -> str:
    """Capture multimodal observation: raw screenshot, labeled screenshot, and element list."""
    page = await RUNTIME.ensure_page()

    # Wait for layout to stabilise after any navigation or DOM mutation.
    # Use a short timeout so we don't stall on pages that never reach networkidle.
    try:
        await page.wait_for_load_state("networkidle", timeout=4000)
    except Exception:
        pass  # Proceed even if networkidle is not reached in time.

    # Dedup: if URL and title haven't changed since the last observation, reuse cached result.
    current_url = page.url
    current_title = await _safe_page_title(page)
    last_obs = getattr(RUNTIME, "_last_observation_key", None)
    last_obs_result = getattr(RUNTIME, "_last_observation_result", None)
    obs_key = (current_url, current_title)
    if last_obs == obs_key and last_obs_result is not None:
        # Still log it so action sequence is traceable, but don't re-screenshot.
        cached_result = dict(last_obs_result)
        cached_result["observation_cached"] = True
        _append_process_log(
            {
                "stage": "observation",
                "url": current_url,
                "data": cached_result,
                "cached": True,
            }
        )
        return json.dumps(cached_result, ensure_ascii=True)

    screenshot = RUNTIME.screenshot_path()

    # Take screenshot and collect DOM rects as close together as possible.
    # We call evaluate first (gets current layout rects), then screenshot immediately
    # in the same event-loop tick — this minimises layout-shift between the two.
    dom_data = await _collect_dom_elements(page)
    await page.screenshot(path=str(screenshot), full_page=False)
    # Persist for the action tools that operate via stable element_id.
    RUNTIME.last_elements = dom_data
    if getattr(RUNTIME, "visited_urls", None) is not None:
        RUNTIME.visited_urls.add(page.url)
    labeled_screenshot_path = _annotate_screenshot(screenshot, dom_data)
    text_elements = [
        f"[{item['id']}]: <{_element_action_kind(item)}> {_element_reasoning_label(item)}"
        for item in dom_data
    ]

    result = {
        "url": current_url,
        "title": current_title,
        "screenshot_path": str(screenshot.resolve()),
        "labeled_screenshot_path": labeled_screenshot_path,
        "elements": dom_data,
        "web_elements_text": text_elements,
        "observation_cached": False,
    }
    # Cache result so consecutive same-page calls reuse it without re-screenshotting.
    RUNTIME._last_observation_key = obs_key
    RUNTIME._last_observation_result = result
    # Persist full observation per step for debugging/teaching.
    _append_process_log({"stage": "observation", "url": current_url, "data": result})
    return json.dumps(result, ensure_ascii=True)


async def close_browser() -> str:
    """Close and reset the browser runtime."""
    await RUNTIME.close()
    return "Browser closed"


async def wait(seconds: int = 5) -> str:
    """Pause briefly to wait for async page updates."""
    page = await RUNTIME.ensure_page()
    sleep_for = max(1, min(int(seconds), 15))
    await asyncio.sleep(sleep_for)
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "wait",
            "seconds": sleep_for,
        }
    )
    return f"Waited {sleep_for}s; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


async def go_back() -> str:
    """Navigate back one page in browser history."""
    page = await RUNTIME.ensure_page()
    await page.go_back(wait_until="domcontentloaded")
    post = json.loads(await capture_multimodal_observation())
    _append_process_log({"stage": "action", "url": page.url, "tool": "go_back"})
    return f"Went back; url={page.url}; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


async def go_google() -> str:
    """Navigate directly to Google homepage."""
    page = await RUNTIME.ensure_page()
    await page.goto("https://www.google.com", wait_until="domcontentloaded")
    post = json.loads(await capture_multimodal_observation())
    _append_process_log({"stage": "action", "url": page.url, "tool": "go_google"})
    return (
        f"Opened Google; labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def verify_task_completion(
    must_include_text_1: str = "",
    must_include_text_2: str = "",
) -> str:
    """Verify completion by checking visible page text for required evidence."""
    page = await RUNTIME.ensure_page()
    body_text = await _evaluate_when_stable(
        page,
        "() => (document.body && document.body.innerText) ? document.body.innerText : ''",
    )
    body_lower = (body_text or "").lower()
    t1 = (must_include_text_1 or "").strip().lower()
    t2 = (must_include_text_2 or "").strip().lower()
    ok1 = (not t1) or (t1 in body_lower)
    ok2 = (not t2) or (t2 in body_lower)
    post = json.loads(await capture_multimodal_observation())
    evidence = {
        "must_include_text_1": must_include_text_1,
        "must_include_text_2": must_include_text_2,
        "matched_1": ok1,
        "matched_2": ok2,
        "url": page.url,
        "labeled_screenshot_path": post.get("labeled_screenshot_path"),
    }
    _append_process_log({"stage": "verification", "url": page.url, "data": evidence})
    if ok1 and ok2:
        return json.dumps({"status": "verified", **evidence}, ensure_ascii=True)
    return json.dumps({"status": "not_verified", **evidence}, ensure_ascii=True)


def _parse_action_command(command: str) -> tuple[str | None, dict[str, str]]:
    """Parse a strict single-action command."""
    cmd = (command or "").strip()
    # Non-answer patterns: single-line matching.
    single_line_patterns = {
        "click": r"^Click\s*\[?(\d+)\]?$",
        "type": r"^Type\s*\[?(\d+)\]?\s*;\s*(.+)$",
        "scroll": r"^Scroll\s*\[?(WINDOW|\d+)\]?\s*;\s*\[?(up|down)\]?$",
        "wait": r"^Wait$",
        "goback": r"^GoBack$",
        "google": r"^Google$",
    }
    for key, pat in single_line_patterns.items():
        m = re.match(pat, cmd, flags=re.IGNORECASE)
        if not m:
            continue
        if key == "click":
            return key, {"number": m.group(1)}
        if key == "type":
            return key, {"number": m.group(1), "content": m.group(2).strip()}
        if key == "scroll":
            return key, {"number": m.group(1), "content": m.group(2).lower()}
        return key, {}

    # ANSWER pattern: allow multiline content after the semicolon (re.DOTALL).
    answer_match = re.match(
        r"^ANSWER\s*;\s*(.+)$", cmd, flags=re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        return "answer", {"content": answer_match.group(1).strip()}

    return None, {}


def _is_search_like_url(url: str) -> bool:
    u = (url or "").lower()
    if not u:
        return True
    return (
        "google.com/search" in u
        or "duckduckgo.com/?q=" in u
        or "duckduckgo.com/?ia=" in u
        or u.rstrip("/")
        in {"https://www.google.com", "https://google.com", "about:blank"}
    )


def _is_captcha_like_url(url: str) -> bool:
    u = (url or "").lower()
    return "/sorry/" in u or "captcha" in u or "challenge" in u


def _is_auth_wall_url(url: str) -> bool:
    """Return True if the URL is a login, register, or account-creation page."""
    u = (url or "").lower()
    return (
        "/signin" in u
        or "/login" in u
        or "/ap/signin" in u
        or "/register" in u
        or "/business/register" in u
        or "/account/login" in u
        or "auth.amazon" in u
    )


async def _dismiss_blocking_overlays() -> str:
    """Best-effort close of common popups/modals/overlays, then recapture observation."""
    page = await RUNTIME.ensure_page()
    result = await _evaluate_when_stable(
        page,
        """
        () => {
          const closeHints = [
            'close', 'dismiss', 'not now', 'no thanks', 'no, thanks',
            'done', 'ok', 'got it', 'accept', 'continue'
          ];
          const isVisible = (el) => {
            if (!el) return false;
            const st = window.getComputedStyle(el);
            if (!st || st.display === 'none' || st.visibility === 'hidden' || st.pointerEvents === 'none') return false;
            const r = el.getBoundingClientRect();
            return r.width > 0 && r.height > 0 && r.bottom > 0 && r.right > 0 && r.top < window.innerHeight && r.left < window.innerWidth;
          };
          const labelOf = (el) => {
            const txt = (el.innerText || el.textContent || '').trim();
            const aria = (el.getAttribute('aria-label') || '').trim();
            const title = (el.getAttribute('title') || '').trim();
            const val = (el.getAttribute('value') || '').trim();
            return (txt || aria || title || val || '').replace(/\\s+/g, ' ').toLowerCase();
          };
          const clickEl = (el, why, hits) => {
            try {
              el.click();
              hits.push({ why, label: labelOf(el).slice(0, 80) });
              return true;
            } catch (_) {
              return false;
            }
          };

          const hits = [];
          const roots = Array.from(document.querySelectorAll('[role=\"dialog\"], [aria-modal=\"true\"], .modal, .popup, .overlay, [data-testid*=\"modal\" i], [class*=\"modal\" i]'));
          // Include full doc as fallback.
          roots.push(document.body);

          for (const root of roots) {
            if (!root || !isVisible(root)) continue;
            const candidates = Array.from(root.querySelectorAll('button, [role=\"button\"], input[type=\"button\"], input[type=\"submit\"], [aria-label], [title], [data-testid*=\"close\" i], [class*=\"close\" i]'));
            for (const el of candidates) {
              if (!isVisible(el)) continue;
              const l = labelOf(el);
              if (!l) continue;
              const exactX = l === 'x' || l === '×' || l === '✕' || l === '✖';
              const hinted = closeHints.some((h) => l.includes(h));
              const closeLike = exactX || hinted || l.includes('close');
              if (!closeLike) continue;
              if (clickEl(el, 'candidate_close', hits) && hits.length >= 3) break;
            }
            if (hits.length >= 3) break;
          }
          return { clicked: hits.length, hits };
        }
        """,
    )
    # Also try Esc as a universal modal dismiss.
    try:
        await page.keyboard.press("Escape")
    except Exception:
        pass
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "dismiss_overlays",
            "clicked": int((result or {}).get("clicked", 0)),
            "hits": (result or {}).get("hits", []),
            "next_step": "retry intended action after popup/modal dismissal",
            "labeled_screenshot_path": post.get("labeled_screenshot_path"),
        }
    )
    clicked = int((result or {}).get("clicked", 0))
    return (
        f"[action=recovery progress={'changed' if clicked > 0 else 'no_change'}] "
        f"dismiss_overlays clicked={clicked}; labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


_DEFAULT_MULTIMODAL_PROMPT = (
    "Describe the page goal, key UI regions, and best next actions using visible clues."
)


# First URL path segments that usually mean global discovery / account hubs, not task-local content.
# Domain-agnostic pattern (many large sites use similar top-level sections).
_SITE_HUB_FIRST_SEGMENTS = frozenset(
    {
        "jobs",
        "careers",
        "learning",
        "education",
        "games",
        "feed",
        "notifications",
        "messaging",
        "talent",
        "sales",
        "pub",
        "top-content",
        "explore",
        "discover",
        "trends",
    }
)


def _url_path_segments(url: str) -> list[str]:
    try:
        p = urlparse(url or "").path or "/"
    except Exception:
        return []
    return [s.lower() for s in p.split("/") if s]


def _should_recover_site_hub_detour(before_url: str, after_url: str) -> bool:
    """Recover accidental drift into global hub sections or off-site while focus_domain is set.

    Not site-specific: uses ``focus_domain`` (already set when the agent reached target-site content)
    plus common first-path segments seen on large properties (jobs, feed, learning, …).
    """
    focus = (getattr(RUNTIME, "focus_domain", "") or "").strip().lower()
    if not focus:
        return False
    db = _domain_from_url(before_url)
    da = _domain_from_url(after_url)
    if db != focus:
        return False
    if da != focus:
        return True
    seg_before = _url_path_segments(before_url)
    seg_after = _url_path_segments(after_url)
    if not seg_after:
        return False
    after_hub = seg_after[0] in _SITE_HUB_FIRST_SEGMENTS
    before_hub = bool(seg_before) and seg_before[0] in _SITE_HUB_FIRST_SEGMENTS
    return bool(after_hub and not before_hub)


def _stable_task_fingerprint(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()


def _maybe_reset_session_for_new_task_prompt(vlm_prompt: str) -> None:
    """Clear session_done when ADK sends a new substantive task (follow-up chat message)."""
    t = (vlm_prompt or "").strip().lower()
    if len(t) < 20:
        return
    if t.startswith(_DEFAULT_MULTIMODAL_PROMPT.lower()):
        return
    fp = _stable_task_fingerprint(vlm_prompt)
    prev = getattr(RUNTIME, "_last_user_task_fp", None)
    if prev is not None and fp != prev:
        RUNTIME.session_done = False
        RUNTIME.answer_locked = False
        RUNTIME.observation_streak = 0
        _append_process_log(
            {
                "stage": "session_scope",
                "reason": "new_task_prompt",
                "cleared_session_done": True,
            }
        )
    RUNTIME._last_user_task_fp = fp


def _domain_from_url(url: str) -> str:
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _extract_domains_from_text(text: str) -> set[str]:
    s = (text or "").lower()
    tokens = re.findall(r"\b([a-z0-9-]+\.(?:com|org|net|io|ai|co|edu|gov))\b", s)
    return {t.strip(".") for t in tokens if t}


_CURRENCY_SYMBOL_TO_CODE: dict[str, str] = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "¥": "JPY",
    "₹": "INR",
}


def _normalize_currency_token(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    if t in _CURRENCY_SYMBOL_TO_CODE:
        return _CURRENCY_SYMBOL_TO_CODE[t]
    t_up = t.upper()
    if re.fullmatch(r"[A-Z]{3}", t_up):
        return t_up
    return ""


def _extract_budget_constraint(text: str) -> tuple[float | None, str | None]:
    """Extract generic budget as (amount, ISO currency code) from free text."""
    s = (text or "").strip()
    if not s:
        return None, None
    # Symbol before amount: "$100", "€ 90"
    m = re.search(r"([$€£¥₹])\s*([0-9][0-9,]*(?:\.[0-9]+)?)", s)
    if m:
        cur = _normalize_currency_token(m.group(1))
        return float(m.group(2).replace(",", "")), cur
    # Code before amount: "USD 100", "PKR 50,000"
    m = re.search(r"\b([A-Za-z]{3})\s*([0-9][0-9,]*(?:\.[0-9]+)?)\b", s)
    if m:
        cur = _normalize_currency_token(m.group(1))
        if cur:
            return float(m.group(2).replace(",", "")), cur
    # Amount before code: "100 USD", "50,000 PKR"
    m = re.search(r"\b([0-9][0-9,]*(?:\.[0-9]+)?)\s*([A-Za-z]{3})\b", s)
    if m:
        cur = _normalize_currency_token(m.group(2))
        if cur:
            return float(m.group(1).replace(",", "")), cur
    return None, None


def _currency_aliases(currency_code: str) -> list[str]:
    cur = _normalize_currency_token(currency_code)
    if not cur:
        return []
    aliases = [cur]
    for sym, code in _CURRENCY_SYMBOL_TO_CODE.items():
        if code == cur:
            aliases.append(sym)
    return aliases


def _extract_prices_for_currency(answer_text: str, currency_code: str) -> list[float]:
    ans = answer_text or ""
    aliases = _currency_aliases(currency_code)
    if not aliases:
        return []
    vals: list[float] = []
    seen: set[float] = set()
    for alias in aliases:
        # Currency before amount.
        if len(alias) == 1 and not alias.isalnum():
            pat1 = re.compile(
                rf"{re.escape(alias)}\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)(?!\s*\))"
            )
            pat2 = None
        else:
            pat1 = re.compile(
                rf"\b{re.escape(alias)}\s*([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)(?!\s*\))\b",
                flags=re.IGNORECASE,
            )
            # Amount before currency code.
            pat2 = re.compile(
                rf"\b([0-9][0-9,]*(?:\.[0-9]{{1,2}})?)\s*{re.escape(alias)}\b",
                flags=re.IGNORECASE,
            )
        for m in pat1.finditer(ans):
            v = float(m.group(1).replace(",", ""))
            if v not in seen:
                vals.append(v)
                seen.add(v)
        if pat2 is not None:
            for m in pat2.finditer(ans):
                v = float(m.group(1).replace(",", ""))
                if v not in seen:
                    vals.append(v)
                    seen.add(v)
    return vals


def _extract_task_constraints(text: str) -> dict[str, Any]:
    """Extract general task constraints from user intent text (domain-agnostic)."""
    s = (text or "").strip()
    lower = s.lower()
    out: dict[str, Any] = {}

    # Ignore the default generic observation prompt to avoid false constraints.
    if lower.startswith(
        "describe the page goal, key ui regions, and best next actions"
    ):
        return out

    out["required_domains"] = sorted(_extract_domains_from_text(s))

    # How many items must appear in the *final* answer (not "browse top N then pick M").
    m_final = re.search(
        r"\b(?:recommend|give|pick|choose|select|return|find)\s+(?:the\s+)?(\d{1,2})\s+"
        r"(?:best|options?|choices?|picks?|items?|products?)\b",
        lower,
    )
    if m_final:
        out["required_option_count"] = max(1, int(m_final.group(1)))
    else:
        # Option count: "top 3", "list 5", "3 options", etc.
        m = re.search(r"\b(?:top|best|list|show|return)\s+(\d{1,2})\b", lower)
        if not m:
            m = re.search(r"\b(\d{1,2})\s+(?:options?|items?|results?)\b", lower)
        if m:
            out["required_option_count"] = max(1, int(m.group(1)))

    # Ranking / comparison intents.
    out["requires_ranking"] = bool(
        re.search(
            r"\b(top|best|highest|lowest|cheapest|most expensive|compare)\b", lower
        )
    )
    out["requires_rating"] = bool(re.search(r"\b(rated|rating|stars?)\b", lower))
    out["needs_scroll"] = bool(
        out.get("requires_ranking")
        or out.get("required_option_count", 0) >= 3
        or re.search(r"\b(options?|results?|list)\b", lower)
    )

    # Budget constraints with generic currency parsing.
    budget_max, budget_currency = _extract_budget_constraint(s)
    if budget_max is not None and budget_currency is not None:
        out["budget_max"] = budget_max
        out["budget_currency"] = budget_currency

    return out


def _is_user_intent_like_text(text: str) -> bool:
    """Heuristic gate: distinguish user-intent tasks from operational prompt chatter."""
    t = re.sub(r"\s+", " ", (text or "").strip().lower())
    if not t:
        return False
    # Explicit generic system observation prompt is not user intent.
    if t.startswith("describe the page goal, key ui regions, and best next actions"):
        return False
    # Operational control prompts should not mutate persistent task constraints.
    operational_markers = [
        "after scrolling",
        "identify and click",
        "click the",
        "show the page",
        "if sorting applied",
        "if nothing changed",
        "describe what happened after",
        "modal",
        "dropdown",
        "element id",
    ]
    if any(m in t for m in operational_markers):
        return False
    # Intent-like requests usually include goal verbs and nouns.
    return bool(
        re.search(
            r"\b(find|search|go to|open|book|list|return|compare|cheapest|best|top|price|budget|under)\b",
            t,
        )
    )


def _merge_task_constraints(
    existing: dict[str, Any], incoming: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(existing or {})
    for k, v in (incoming or {}).items():
        if v is None:
            continue
        if k == "required_domains":
            cur = set(merged.get("required_domains") or [])
            cur.update(v or [])
            merged["required_domains"] = sorted(cur)
        elif isinstance(v, bool):
            merged[k] = bool(merged.get(k, False) or v)
        else:
            merged[k] = v
    return merged


def _merge_constraints_with_fallback(
    primary: dict[str, Any], fallback: dict[str, Any]
) -> dict[str, Any]:
    """Merge LLM-extracted constraints first, then fill missing fields from fallback."""
    out = dict(primary or {})
    for k, v in (fallback or {}).items():
        if v is None:
            continue
        if k == "required_domains":
            cur = set(out.get("required_domains") or [])
            cur.update(v or [])
            out["required_domains"] = sorted(cur)
            continue
        if isinstance(v, bool):
            out[k] = bool(out.get(k, False) or v)
            continue
        if k not in out or out.get(k) in (None, "", 0):
            out[k] = v
    return out


def _normalize_constraints_obj(obj: dict[str, Any]) -> dict[str, Any]:
    """Normalize/validate extracted constraint object to expected keys and types."""
    out: dict[str, Any] = {}
    domains = obj.get("required_domains")
    if isinstance(domains, list):
        out["required_domains"] = sorted(
            {str(d).strip().lower() for d in domains if str(d).strip()}
        )
    for k in ("requires_ranking", "requires_rating", "needs_scroll"):
        if k in obj:
            out[k] = bool(obj.get(k))
    if obj.get("required_option_count") not in (None, ""):
        try:
            c = int(obj.get("required_option_count"))
            if c > 0:
                out["required_option_count"] = min(c, 20)
        except Exception:
            pass
    if obj.get("budget_max") not in (None, ""):
        try:
            b = float(obj.get("budget_max"))
            if b > 0:
                out["budget_max"] = b
        except Exception:
            pass
    cur = _normalize_currency_token(str(obj.get("budget_currency") or "").strip())
    if cur:
        out["budget_currency"] = cur
    # LLMs sometimes misread "top 10" as currency TOP + amount 10 (ISO 4217 TOP is Tonga).
    bm = out.get("budget_max")
    if cur == "TOP" and bm is not None:
        try:
            if float(bm) <= 100.0:
                out.pop("budget_max", None)
                out.pop("budget_currency", None)
        except Exception:
            out.pop("budget_max", None)
            out.pop("budget_currency", None)
    return out


async def _extract_task_constraints_llm(text: str) -> dict[str, Any]:
    """LLM-first extraction of task constraints from open-ended user intent."""
    t = (text or "").strip()
    if not t:
        return {}
    if t.lower().startswith(
        "describe the page goal, key ui regions, and best next actions"
    ):
        return {}

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not openai_api_key:
        return {}

    model = os.getenv("ADK_OPENAI_MODEL", "openai/gpt-5-mini").strip()
    if model.startswith("openai/"):
        model = model.split("/", 1)[1]

    schema_hint = (
        "Return only JSON object with keys: "
        "required_domains (string[]), required_option_count (int|null), "
        "requires_ranking (bool), requires_rating (bool), needs_scroll (bool), "
        "budget_max (number|null), budget_currency (string|null), confidence (0..1)."
    )
    payload = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [
            {
                "role": "system",
                "content": (
                    "You extract user task constraints for a web automation agent. "
                    "Be conservative: only include constraints explicitly present or strongly implied. "
                    + schema_hint
                ),
            },
            {"role": "user", "content": t},
        ],
    }
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        if response.status_code >= 400:
            return {}
        data = response.json()
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get(
            "content"
        ) or "{}"
        parsed = json.loads(content)
        return _normalize_constraints_obj(parsed if isinstance(parsed, dict) else {})
    except Exception:
        return {}


async def _update_task_constraints_from_text(text: str, source: str) -> None:
    """Update runtime constraints from text using LLM-first extraction with fallback."""
    t = (text or "").strip()
    if not t:
        return
    if source == "observation_prompt" and not _is_user_intent_like_text(t):
        return
    normalized = re.sub(r"\s+", " ", t.lower()).strip()
    if not normalized:
        return

    meta = getattr(RUNTIME, "task_constraints_meta", None) or {}
    seen = set(meta.get("seen_inputs") or [])
    if normalized in seen:
        return

    llm_constraints = await _extract_task_constraints_llm(t)
    regex_constraints = _extract_task_constraints(t)
    incoming = _merge_constraints_with_fallback(llm_constraints, regex_constraints)
    existing = getattr(RUNTIME, "task_constraints", None) or {}
    RUNTIME.task_constraints = _merge_task_constraints(existing, incoming)

    seen.add(normalized)
    meta["seen_inputs"] = list(seen)[-30:]
    meta["last_source"] = source
    meta["llm_used"] = bool(llm_constraints)
    meta["last_input_preview"] = t[:180]
    RUNTIME.task_constraints_meta = meta


def _looks_like_blocker_or_followup(answer_text: str) -> bool:
    """Detect non-final blocker/follow-up responses that must not be finalized."""
    t = (answer_text or "").strip().lower()
    if not t:
        return True
    blocker_patterns = [
        r"\bi(?:'| )?m blocked\b",
        r"\bcan['’]?t continue\b",
        r"\bunable to continue\b",
        r"\bneed (you|your) (help|action)\b",
        r"\bwhich option do you (want|prefer)\b",
        r"\bpick one option\b",
        r"\bplease pick one\b",
        r"\bpick\s+(?:one|an option)\b",
        r"\breply with\b",
        r"\breply\s+[\"']?[ab][\"']?\b",
        r"\bchoose\s+(?:either\s+)?(?:option\s+)?[ab]\b",
        r"\boption\s+[ab]\)\s",
        r"\b(?:^|\n)\s*[ab]\)\s",
        r"\bi can (either|do option)\b",
        r"\bdo you want me to\b",
        r"\bwould you like\b",
        r"\bwhich do you (want|prefer)\b",
        r"\bgrant permission\b",
        r"\bstart a new session\b",
        r"\b(new|closed|finished) session\b",
        r"\bI need one small action from you\b",
        r"\bwhat remains\b",
        r"\bpaste the top-\d+\b",
        r"\bor paste\b",
    ]
    if any(re.search(p, t) for p in blocker_patterns):
        return True
    # Menu-style A/B prompts asking the user to reply/paste (not product comparisons).
    if len(t) < 1600 and re.search(
        r"\b(reply|paste|grant permission|please pick)\b", t
    ):
        menu = bool(re.search(r"(?:^|[\n\.])\s*[ab]\)\s|\boption\s+[ab]\b", t))
        asks = bool(re.search(r"\b(which do you|would you like to)\b", t))
        if menu and asks:
            return True
    return False


def _push_recent_action(name: str) -> None:
    arr = getattr(RUNTIME, "recent_actions", None)
    if arr is None:
        RUNTIME.recent_actions = []
        arr = RUNTIME.recent_actions
    arr.append(name)
    if len(arr) > 20:
        del arr[:-20]


def _is_churn_loop() -> bool:
    arr = getattr(RUNTIME, "recent_actions", None) or []
    tail = arr[-8:]
    if len(tail) < 6:
        return False
    allowed = {"google", "input", "wait"}
    if any(a not in allowed for a in tail):
        return False
    # Must include at least two kinds, otherwise same-action recovery handles it.
    if len(set(tail)) < 2:
        return False
    return True


async def _fallback_search_duckduckgo(query: str) -> str:
    page = await RUNTIME.ensure_page()
    q = (query or "").strip()
    required: set[str] = getattr(RUNTIME, "required_domains", None) or set()

    # Always scope the query to the required domain if one is known.
    if required:
        primary = next(iter(required))
        site_token = f"site:{primary}"
        if site_token not in q:
            q = f"{q} {site_token}".strip() if q else site_token

    if not q:
        q = "web search"

    ddg_url = "https://duckduckgo.com/?q=" + urllib.parse.quote_plus(q) + "&ia=web"
    await page.goto(ddg_url, wait_until="domcontentloaded")
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "fallback_search",
            "provider": "duckduckgo",
            "query": q,
            "required_domains": sorted(required),
            "reason": "search provider fallback",
            "next_step": "open a concrete result link on required domain and continue task",
        }
    )
    # Immediately try to open a required-domain result if visible on SERP.
    open_result = await _open_top_result_from_current_serp()
    if "progress=changed" in open_result:
        return (
            f"[action=recovery progress=changed] DDG search for '{q}' then opened result; "
            f"{open_result}"
        )
    return (
        f"[action=recovery progress=changed] Fallback search on DuckDuckGo for query='{q}'; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def _search_google_direct(query: str) -> str:
    """Open Google search results directly for a query."""
    page = await RUNTIME.ensure_page()
    q = (query or "").strip()
    if not q:
        return "[action=recovery progress=no_change] Missing query for direct Google search."
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(q)
    await page.goto(url, wait_until="domcontentloaded")
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "search_google_direct",
            "query": q,
            "reason": "recover from google-home loop using last query",
            "next_step": "click a relevant result link",
        }
    )
    return (
        f"[action=recovery progress=changed] Opened Google results for query='{q}'; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def _open_top_result_from_current_serp() -> str:
    """Open the best SERP result, preferring required domains over any other result."""
    page = await RUNTIME.ensure_page()
    required: set[str] = getattr(RUNTIME, "required_domains", None) or set()

    # Collect ALL candidate hrefs from the SERP.
    all_hrefs: list[str] = await _evaluate_when_stable(
        page,
        """
        () => {
          const selectors = [
            'a h3',
            '#search a',
            'a[data-testid="result-title-a"]',
            'article h2 a',
            'main a'
          ];
          const bad = ['google.com/search', 'duckduckgo.com/?q=', '/settings', '/preferences', '/help', '/sorry/'];
          const seen = new Set();
          const results = [];
          for (const sel of selectors) {
            const nodes = Array.from(document.querySelectorAll(sel));
            for (const node of nodes) {
              const a = node.tagName && node.tagName.toLowerCase() === 'a' ? node : node.closest('a');
              if (!a) continue;
              const h = (a.getAttribute('href') || '').trim();
              if (!h || h.startsWith('/')) continue;
              if (!/^https?:\\/\\//.test(h)) continue;
              const lower = h.toLowerCase();
              if (bad.some((b) => lower.includes(b))) continue;
              if (!seen.has(h)) { seen.add(h); results.push(h); }
            }
          }
          return results;
        }
        """,
    )

    if not all_hrefs:
        # No SERP results at all — if we have required domains, go there directly.
        if required:
            target = next(iter(required))
            return await _navigate_direct_to_required_domain(target)
        return "[action=recovery progress=no_change] No SERP results found."

    # Prefer a result whose hostname matches a required domain.
    href = None
    if required:
        for h in all_hrefs:
            d = _domain_from_url(h)
            if any(d == rd or d.endswith("." + rd) for rd in required):
                href = h
                break

    # Fall back to first result if no required-domain match found.
    if not href:
        href = all_hrefs[0]
        if required:
            _append_process_log(
                {
                    "stage": "blocker",
                    "reason": "no_required_domain_in_serp",
                    "required_domains": sorted(required),
                    "available_hrefs": all_hrefs[:5],
                    "next_step": "navigating directly to required domain",
                }
            )
            # No matching result on this SERP → go directly to required domain.
            target = next(iter(required))
            return await _navigate_direct_to_required_domain(target)

    before = page.url
    try:
        await page.goto(href, wait_until="domcontentloaded")
    except Exception as exc:
        return (
            f"[action=recovery progress=no_change] Failed opening result {href}: {exc}"
        )
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "open_top_result",
            "from_url": before,
            "target_url": href,
            "required_domains": sorted(required),
        }
    )
    return (
        f"[action=recovery progress=changed] Opened result url={page.url}; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def _navigate_direct_to_required_domain(domain: str) -> str:
    """Navigate directly to a required domain when SERP can't provide a result link."""
    page = await RUNTIME.ensure_page()
    url = f"https://www.{domain}"
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "navigate_direct",
            "target": url,
            "reason": "required domain not found in SERP results",
        }
    )
    try:
        await page.goto(url, wait_until="domcontentloaded")
    except Exception as exc:
        return f"[action=recovery progress=no_change] Failed navigating to {url}: {exc}"
    RUNTIME.focus_domain = domain
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "navigate_direct_confirmed",
            "target": url,
        }
    )
    return (
        f"[action=recovery progress=changed] Navigated directly to {url}; "
        f"focus_domain={domain}; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


def _can_finalize_answer(proposed_answer: str = "") -> tuple[bool, str]:
    actions = getattr(RUNTIME, "action_history", None) or []
    visited = getattr(RUNTIME, "visited_urls", None) or set()

    meaningful_actions = [a for a in actions if a in {"click", "type", "scroll"}]
    if len(meaningful_actions) < 2:
        return (
            False,
            "ANSWER rejected: not enough meaningful actions (need at least two of click/type/scroll).",
        )

    non_search = [u for u in visited if not _is_search_like_url(u)]
    if not non_search:
        return False, "ANSWER rejected: no non-search target page was reached."

    required = getattr(RUNTIME, "required_domains", None) or set()
    if required:
        visited_domains = {_domain_from_url(u) for u in visited if u}
        missing = [d for d in sorted(required) if d not in visited_domains]
        if missing:
            return (
                False,
                f"ANSWER rejected: required domain(s) not visited yet: {', '.join(missing)}.",
            )
    constraints = getattr(RUNTIME, "task_constraints", None) or {}
    if constraints.get("needs_scroll") and "scroll" not in actions:
        return (
            False,
            "ANSWER rejected: task requires broader result coverage; no scroll action detected yet.",
        )

    required_count = int(constraints.get("required_option_count") or 0)
    if required_count >= 2:
        numbered = list(
            re.finditer(r"(?:^|\s)(\d+)\.\s+", proposed_answer or "", flags=re.DOTALL)
        )
        if len(numbered) < required_count:
            return (
                False,
                f"ANSWER rejected: expected at least {required_count} ranked options, but fewer were provided.",
            )

    # Budget/currency guard (general, not domain-specific).
    budget_max = constraints.get("budget_max")
    budget_currency = str(constraints.get("budget_currency") or "").upper()
    if budget_max is not None and budget_currency:
        ans = proposed_answer or ""
        upper_ans = ans.upper()
        currency_alias = _currency_aliases(budget_currency)
        has_expected_currency = any(
            tok in upper_ans for tok in [a.upper() for a in currency_alias]
        )
        if not has_expected_currency:
            return (
                False,
                f"ANSWER rejected: expected prices in {budget_currency} due to budget constraint, but answer lacks that currency.",
            )

        # Parse prices in expected currency and enforce max budget.
        price_values = _extract_prices_for_currency(ans, budget_currency)
        if not price_values:
            return (
                False,
                f"ANSWER rejected: no parseable {budget_currency} prices found for budget validation.",
            )
        over = [v for v in price_values if v > float(budget_max)]
        if over:
            return (
                False,
                f"ANSWER rejected: some options exceed budget {budget_currency}{budget_max}: {over[:3]}",
            )

    return True, "ok"


def _has_duplicate_ranked_options(answer_text: str) -> tuple[bool, list[str]]:
    """Detect duplicate ranked options like item 1 and 3 being the same."""
    txt = (answer_text or "").strip()
    if not txt:
        return False, []
    # Extract chunks after numeric bullets: "1. ... 2. ... 3. ..."
    parts = re.split(r"\s*\d+\.\s*", txt)
    chunks = [p.strip() for p in parts if p and p.strip()]
    if len(chunks) < 3:
        return False, []
    items: list[str] = []
    for chunk in chunks:
        part = re.sub(r"\s+", " ", chunk.strip().lower())
        if not part:
            continue
        # Keep only product-like leading segment before long price/details tail.
        part = re.split(r"\s+-\s+|\s+\$|\s+pk[r]?\s+", part, maxsplit=1)[0].strip()
        part = re.sub(r"[^a-z0-9 ]+", "", part).strip()
        if part:
            items.append(part)
    seen: set[str] = set()
    dups: list[str] = []
    for it in items:
        if it in seen and it not in dups:
            dups.append(it)
        seen.add(it)
    return len(dups) > 0, dups


def _is_editable_element(el: dict[str, Any] | None) -> bool:
    if not el:
        return False
    tag = str(el.get("tag") or "").lower()
    role = str(el.get("role") or "").lower()
    input_type = str(el.get("input_type") or "").lower()
    if tag in {"textarea", "select"}:
        return True
    if tag == "input":
        return input_type not in {
            "button",
            "submit",
            "checkbox",
            "radio",
            "range",
            "file",
            "image",
            "hidden",
        }
    return role in {"textbox", "combobox", "searchbox"}


def _looks_like_url_or_query_payload(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    return bool(
        "http://" in t
        or "https://" in t
        or "www." in t
        or "://"
        or ("?" in t and "=" in t)
        or "&" in t
        or "s=price-asc-rank" in t
        or "ref=" in t
    )


def _is_site_search_input(el: dict[str, Any] | None) -> bool:
    if not el:
        return False
    role = str(el.get("role") or "").lower()
    name = str(el.get("name") or "").lower()
    aria = str(el.get("aria_label") or "").lower()
    placeholder = str(el.get("placeholder") or "").lower()
    text = str(el.get("text") or "").lower()
    return (
        role in {"searchbox", "combobox"}
        or name in {"k", "q", "field-keywords"}
        or "search" in aria
        or "search" in placeholder
        or "search" in text
    )


async def _scroll_element_center(
    element_id: str, direction: str, pixels: int = 500
) -> str:
    page = await RUNTIME.ensure_page()
    el = _lookup_element(element_id)
    if not el:
        return f"scroll failed: unknown element_id {element_id}"
    x = int(el.get("x", 0)) + int(el.get("width", 1)) // 2
    y = int(el.get("y", 0)) + int(el.get("height", 1)) // 2
    await page.mouse.move(x, y)
    dy = pixels if direction == "down" else -pixels
    await page.mouse.wheel(0, dy)
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "action_scroll",
            "element_id": element_id,
            "direction": direction,
            "pixels": abs(dy),
        }
    )
    return f"Scrolled {direction} on element_id={element_id}; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


async def execute_action(command: str) -> str:
    """Execute one strict action command."""
    action, info = _parse_action_command(command)
    if not action:
        return (
            "Invalid action format. Use one of: "
            "Click [id], Type [id]; [content], Scroll [id|WINDOW]; [up|down], "
            "Wait, GoBack, Google, ANSWER; [content]"
        )
    if getattr(RUNTIME, "session_done", False):
        return (
            "[action=blocked status=session_done] Task is already concluded. "
            "Start a new session/task before executing more actions."
        )

    action_name_map = {
        "click": "click",
        "type": "input",
        "scroll": "scroll",
        "wait": "wait",
        "goback": "back",
        "google": "google",
        "answer": "answer",
    }
    action_name = action_name_map.get(action, action)
    _push_recent_action(action_name)

    # Exit intervention mode after a non-churn action class is attempted.
    if action_name in {"click", "scroll", "back"}:
        RUNTIME.churn_break_count = 0

    # Hard anti-churn guard: break loops of google/input/wait cycling.
    if _is_churn_loop():
        RUNTIME.churn_break_count = getattr(RUNTIME, "churn_break_count", 0) + 1
        if RUNTIME.churn_break_count >= 3 and action_name in {
            "google",
            "input",
            "wait",
        }:
            return (
                "[action=intervention status=blocked] Repeated search churn detected. "
                "Do not use action_google/action_input/action_wait now. "
                "Next required action: action_click on a concrete result id, or action_scroll/action_back."
            )
        q = getattr(RUNTIME, "last_search_query", "")
        # Prefer opening top result if already on a search page.
        if _is_search_like_url((await RUNTIME.ensure_page()).url):
            open_result = await _open_top_result_from_current_serp()
            if "progress=changed" in open_result:
                _append_process_log(
                    {
                        "stage": "action_result",
                        "url": (await RUNTIME.ensure_page()).url,
                        "action_name": action_name,
                        "progress_changed": False,
                        "no_progress_streak": getattr(RUNTIME, "no_progress_streak", 0),
                        "auto_recovery": "churn_break_open_top_result",
                    }
                )
                return f"[action={action_name} progress=loop_detected] {open_result}"
        # Fallback search provider swap as second break strategy.
        ddg = await _fallback_search_duckduckgo(q)
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": False,
                "no_progress_streak": getattr(RUNTIME, "no_progress_streak", 0),
                "auto_recovery": "churn_break_duckduckgo",
            }
        )
        return f"[action={action_name} progress=loop_detected] {ddg}"
    else:
        RUNTIME.churn_break_count = 0

    # Auto-recovery: if repeated same action while no progress persists, force
    # a recovery action before allowing more of the same type.
    if (
        getattr(RUNTIME, "no_progress_streak", 0) >= 2
        and getattr(RUNTIME, "last_action_name", "") == action_name
        and action_name not in {"scroll", "back", "google", "answer"}
    ):
        # First recovery attempt: dismiss blocking overlays/modal dialogs.
        overlay_recovery = await _dismiss_blocking_overlays()
        if "progress=changed" in overlay_recovery:
            RUNTIME.last_action_name = "recovery_dismiss_overlay"
            _append_process_log(
                {
                    "stage": "action_result",
                    "url": (await RUNTIME.ensure_page()).url,
                    "action_name": action_name,
                    "progress_changed": False,
                    "no_progress_streak": RUNTIME.no_progress_streak,
                    "auto_recovery": "dismiss_overlay",
                }
            )
            return (
                "[action=recovery progress=forced] No progress on repeated action; "
                f"executed overlay-dismiss recovery. {overlay_recovery}"
            )
        # First recovery attempt: window scroll down.
        if getattr(RUNTIME, "last_action_name", "") != "recovery_scroll":
            recovery = await scroll_by(700)
            RUNTIME.last_action_name = "recovery_scroll"
            _append_process_log(
                {
                    "stage": "action_result",
                    "url": (await RUNTIME.ensure_page()).url,
                    "action_name": action_name,
                    "progress_changed": False,
                    "no_progress_streak": RUNTIME.no_progress_streak,
                    "auto_recovery": "scroll_window_down",
                }
            )
            return (
                "[action=recovery progress=forced] No progress on repeated action; "
                f"executed recovery scroll. {recovery}"
            )
        # Second recovery attempt: go back.
        recovery = await go_back()
        RUNTIME.last_action_name = "recovery_back"
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": False,
                "no_progress_streak": RUNTIME.no_progress_streak,
                "auto_recovery": "go_back",
            }
        )
        return (
            "[action=recovery progress=forced] No progress on repeated action; "
            f"executed recovery back. {recovery}"
        )

    page = await RUNTIME.ensure_page()
    before_url = page.url
    before_title = await _safe_page_title(page)

    # Invalidate the observation cache before any mutating action.
    RUNTIME._last_observation_key = None

    if action == "click":
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("click")
        result = await click_element(info["number"])
    elif action == "type":
        target_el = _lookup_element(info["number"])
        if not _is_editable_element(target_el):
            elements = getattr(RUNTIME, "last_elements", None) or []
            editable = [e for e in elements if _is_editable_element(e)]
            preview = [
                {
                    "id": e.get("id"),
                    "tag": e.get("tag"),
                    "text": (e.get("text") or "")[:40],
                }
                for e in editable[:8]
            ]
            _append_process_log(
                {
                    "stage": "action",
                    "url": page.url,
                    "tool": "input",
                    "status": "rejected_non_editable_target",
                    "element_id": info["number"],
                    "suggested_editable_ids": preview,
                }
            )
            return (
                f"[action=input status=rejected] element_id={info['number']} is not an editable input. "
                f"Use an editable id from: {json.dumps(preview, ensure_ascii=True)}"
            )
        # Prevent fake URL/query injection into on-site search fields; filters/sorts
        # should be applied via real UI controls (dropdowns/checks), not search text hacks.
        if (
            not _is_search_like_url(page.url)
            and _is_site_search_input(target_el)
            and _looks_like_url_or_query_payload(info["content"])
        ):
            return (
                "[action=input status=rejected] Detected URL/querystring payload into site search input. "
                "Do not force filters/sort via fake URL text in search boxes. "
                "Use visible UI controls: click Sort dropdown and choose options, then apply filters."
            )
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("type")
        if _is_search_like_url(page.url):
            RUNTIME.last_search_query = info["content"]
            required = _extract_domains_from_text(info["content"])
            if required:
                if getattr(RUNTIME, "required_domains", None) is None:
                    RUNTIME.required_domains = set()
                RUNTIME.required_domains.update(required)
            await _update_task_constraints_from_text(
                info["content"], source="search_input"
            )
        result = await type_text_element(
            info["number"], info["content"], press_enter=True
        )
    elif action == "scroll":
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("scroll")
        target = info["number"]
        direction = info["content"]
        if target.upper() == "WINDOW":
            delta = 700 if direction == "down" else -700
            result = await scroll_by(delta)
        else:
            result = await _scroll_element_center(target, direction)
    elif action == "wait":
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("wait")
        result = await wait(5)
    elif action == "goback":
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("goback")
        result = await go_back()
    elif action == "google":
        current_url = page.url
        current_domain = _domain_from_url(current_url)
        focus = (getattr(RUNTIME, "focus_domain", "") or "").lower()
        required = getattr(RUNTIME, "required_domains", None) or set()
        detours = getattr(RUNTIME, "wrong_domain_detour_count", 0)

        # Hard block: if required domains exist and we've already detoured 2+ times
        # without reaching one, go directly instead of looping through Google.
        if required and detours >= 2:
            target = next(iter(required))
            _append_process_log(
                {
                    "stage": "blocker",
                    "reason": "repeated_wrong_domain_detour",
                    "detour_count": detours,
                    "required_domains": sorted(required),
                    "next_step": f"navigating directly to {target}",
                }
            )
            RUNTIME.wrong_domain_detour_count = 0
            return await _navigate_direct_to_required_domain(target)

        focus_enforced = (not required) or (focus in required)
        if (
            focus
            and focus_enforced
            and current_domain == focus
            and not _is_captcha_like_url(current_url)
        ):
            return (
                f"[action=google status=rejected] Focus domain '{focus}' is active. "
                "Do not leave target site unless blocked. Use click/scroll/back within current site."
            )
        # Prevent infinite google-home loops.
        if (
            _is_search_like_url(current_url)
            and "google.com/search" not in (current_url or "").lower()
        ):
            if getattr(RUNTIME, "no_progress_streak", 0) >= 1:
                q = getattr(RUNTIME, "last_search_query", "")
                if q:
                    return await _search_google_direct(q)
                return (
                    "[action=google status=rejected] Already on Google homepage with no query context. "
                    "Next required action: action_input(element_id, text) on the search box."
                )
        RUNTIME.observation_streak = 0
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("google")
        result = await go_google()
    elif action == "answer":
        if getattr(RUNTIME, "answer_locked", False):
            return (
                "[action=answer status=rejected] Answer already recorded for this session. "
                "Do not call action_answer again unless a new task/session is started."
            )
        if _looks_like_blocker_or_followup(info.get("content", "")):
            return (
                "[action=answer status=rejected] Proposed answer is a blocker/follow-up message, not final deliverable output. "
                "Continue browsing and return concrete task results."
            )
        has_dups, dup_items = _has_duplicate_ranked_options(info.get("content", ""))
        if has_dups:
            return (
                "[action=answer status=rejected] Duplicate ranked options detected in proposed answer. "
                f"Duplicate entries: {json.dumps(dup_items, ensure_ascii=True)}. "
                "Collect additional unique options (scroll/filter) before answering."
            )
        ok, reason = _can_finalize_answer(info.get("content", ""))
        if not ok:
            return reason
        # Snapshot right before final answer for evidence traceability.
        post = json.loads(await capture_multimodal_observation())
        if getattr(RUNTIME, "action_history", None) is not None:
            RUNTIME.action_history.append("answer")
        _append_process_log(
            {
                "stage": "action",
                "url": page.url,
                "tool": "answer",
                "content": info.get("content", ""),
                "labeled_screenshot_path": post.get("labeled_screenshot_path"),
                "focus_domain": getattr(RUNTIME, "focus_domain", ""),
                "required_domains": sorted(
                    getattr(RUNTIME, "required_domains", None) or []
                ),
                "task_constraints": getattr(RUNTIME, "task_constraints", None) or {},
            }
        )
        RUNTIME.answer_locked = True
        RUNTIME.session_done = True
        return f"[action=answer status=recorded] {info.get('content', '')}"
    else:
        return "Unknown action"
    after_url = page.url
    after_title = await _safe_page_title(page)
    after_domain = _domain_from_url(after_url)
    if after_domain and not _is_search_like_url(after_url):
        required = getattr(RUNTIME, "required_domains", None) or set()
        if required:
            if after_domain in required:
                # Reached a required domain — set focus and reset detour counter.
                RUNTIME.focus_domain = after_domain
                RUNTIME.wrong_domain_detour_count = 0
            else:
                # Landed on a non-required domain — count it.
                RUNTIME.wrong_domain_detour_count = (
                    getattr(RUNTIME, "wrong_domain_detour_count", 0) + 1
                )
        elif not getattr(RUNTIME, "focus_domain", ""):
            RUNTIME.focus_domain = after_domain

    # Focus-domain drift: accidental hops to global hubs (jobs/feed/learning/…) or off-site links.
    if action == "click" and _should_recover_site_hub_detour(before_url, after_url):
        recovery = await go_back()
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": True,
                "auto_recovery": "site_hub_detour_go_back",
                "before_url": before_url,
                "after_url": after_url,
                "focus_domain": getattr(RUNTIME, "focus_domain", ""),
                "next_step": "stay on task-local URLs; prefer scroll and in-page controls over global nav",
            }
        )
        return (
            f"[action=click progress=recovered] Navigated away from focused task area (hub/off-site detour); "
            f"went back. {recovery}"
        )

    # Auth-wall trap: if a click/navigate lands on a login or register page, go back immediately.
    if _is_auth_wall_url(after_url) and action not in {"goback", "google"}:
        recovery = await go_back()
        _append_process_log(
            {
                "stage": "blocker",
                "reason": "auth_wall_detected",
                "evidence_url": after_url,
                "evidence_hint": "Page is a signin/register/auth wall — cannot complete task here.",
                "next_step": "went back; find another path to the target content without signing in",
            }
        )
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": False,
                "no_progress_streak": RUNTIME.no_progress_streak,
                "auto_recovery": "auth_wall_go_back",
                "focus_domain": getattr(RUNTIME, "focus_domain", ""),
                "required_domains": sorted(
                    getattr(RUNTIME, "required_domains", None) or []
                ),
            }
        )
        return (
            f"[action={action_name} status=blocked reason=auth_wall] "
            f"evidence_url={after_url} | Went back. "
            "Avoid clicking sign-in/register links. Use direct product/search URLs instead."
        )

    # Captcha/block fallback: auto-switch to DuckDuckGo on blocked search pages.
    if _is_captcha_like_url(after_url):
        q = getattr(RUNTIME, "last_search_query", "")
        recovery = await _fallback_search_duckduckgo(q)
        blocker = {
            "stage": "blocker",
            "reason": "google_block_or_captcha",
            "evidence_url": after_url,
            "evidence_hint": "Detected /sorry or captcha/challenge pattern in URL.",
            "next_step": "fallback to DuckDuckGo and open relevant result",
        }
        _append_process_log(blocker)
        RUNTIME.no_progress_streak = 0
        RUNTIME.search_loop_streak = 0
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": False,
                "no_progress_streak": RUNTIME.no_progress_streak,
                "auto_recovery": "captcha_fallback_duckduckgo",
                "focus_domain": getattr(RUNTIME, "focus_domain", ""),
                "required_domains": sorted(
                    getattr(RUNTIME, "required_domains", None) or []
                ),
            }
        )
        return (
            f"[action={action_name} progress=blocked reason=google_block_or_captcha] "
            f"evidence_url={after_url} next_step=fallback_duckduckgo | {result} | {recovery}"
        )

    progress_changed = (after_url != before_url) or (after_title != before_title)
    if not progress_changed and action_name in {"click", "input"}:
        overlay_recovery = await _dismiss_blocking_overlays()
        if "progress=changed" in overlay_recovery:
            RUNTIME.no_progress_streak = 0
            _append_process_log(
                {
                    "stage": "action_result",
                    "url": (await RUNTIME.ensure_page()).url,
                    "action_name": action_name,
                    "progress_changed": False,
                    "no_progress_streak": RUNTIME.no_progress_streak,
                    "auto_recovery": "dismiss_overlay_post_action",
                }
            )
            return (
                f"[action={action_name} progress=no_change] {result} | "
                f"[recovery] {overlay_recovery}"
            )
    if progress_changed:
        RUNTIME.no_progress_streak = 0
    else:
        RUNTIME.no_progress_streak = getattr(RUNTIME, "no_progress_streak", 0) + 1

    # Detect search-query churn (URL changes but still stuck on search pages).
    if _is_search_like_url(before_url) and _is_search_like_url(after_url):
        if action_name in {"input", "click", "wait"}:
            RUNTIME.search_loop_streak = getattr(RUNTIME, "search_loop_streak", 0) + 1
    else:
        RUNTIME.search_loop_streak = 0

    if RUNTIME.search_loop_streak >= 2:
        recovery_result = await _open_top_result_from_current_serp()
        RUNTIME.search_loop_streak = 0
        _append_process_log(
            {
                "stage": "action_result",
                "url": (await RUNTIME.ensure_page()).url,
                "action_name": action_name,
                "progress_changed": progress_changed,
                "no_progress_streak": RUNTIME.no_progress_streak,
                "auto_recovery": "open_top_result_from_serp",
            }
        )
        return f"[action={action_name} progress=loop_detected] {result} | {recovery_result}"

    RUNTIME.last_action_name = action_name
    _append_process_log(
        {
            "stage": "action_result",
            "url": after_url,
            "action_name": action_name,
            "progress_changed": progress_changed,
            "no_progress_streak": RUNTIME.no_progress_streak,
            "search_loop_streak": RUNTIME.search_loop_streak,
            "before_url": before_url,
            "after_url": after_url,
            "before_title": before_title,
            "after_title": after_title,
            "focus_domain": getattr(RUNTIME, "focus_domain", ""),
            "required_domains": sorted(
                getattr(RUNTIME, "required_domains", None) or []
            ),
            "task_constraints": getattr(RUNTIME, "task_constraints", None) or {},
            "wrong_domain_detour_count": getattr(
                RUNTIME, "wrong_domain_detour_count", 0
            ),
            "next_step_hint": (
                "click relevant result or apply filter"
                if progress_changed
                else "choose a different element or use scroll/back"
            ),
        }
    )
    hint = ""
    if RUNTIME.no_progress_streak >= 2:
        hint = " hint=No progress detected. Try different element_id or use Scroll/GoBack/Google."
    return f"[action={action_name} progress={'changed' if progress_changed else 'no_change'}]{hint} {result}"


async def action_click(element_id: str) -> str:
    """Strict action: click an element by id."""
    return await execute_action(f"Click [{element_id}]")


async def action_input(element_id: str, text: str) -> str:
    """Strict action: type text into editable element id."""
    return await execute_action(f"Type [{element_id}]; {text}")


async def action_scroll(target: str = "WINDOW", direction: str = "down") -> str:
    """Strict action: scroll target (WINDOW or element id) up/down."""
    tgt = (target or "WINDOW").strip()
    dirn = (direction or "down").strip().lower()
    if dirn not in {"up", "down"}:
        return "Invalid scroll direction. Use 'up' or 'down'."
    return await execute_action(f"Scroll [{tgt}]; {dirn}")


async def action_wait(seconds: int = 5) -> str:
    """Strict action: wait briefly (seconds currently informational)."""
    _ = seconds
    return await execute_action("Wait")


async def action_back() -> str:
    """Strict action: navigate back."""
    return await execute_action("GoBack")


async def action_google() -> str:
    """Strict action: navigate to Google homepage."""
    return await execute_action("Google")


async def action_answer(content: str) -> str:
    """Strict action: finalize with answer content."""
    # Normalize whitespace so multiline model output doesn't break the parser.
    safe_content = " ".join((content or "").split())
    if not safe_content:
        return "[action=answer status=rejected] Empty answer content — provide the actual results."
    return await execute_action(f"ANSWER; {safe_content}")


def _lookup_element(element_id: str) -> dict[str, Any] | None:
    elements = getattr(RUNTIME, "last_elements", None) or []
    for el in elements:
        if el.get("id") == element_id:
            return el
    return None


async def click_element(element_id: str) -> str:
    """Click using a stable element_id derived from bbox on the latest observation."""
    page = await RUNTIME.ensure_page()
    el = _lookup_element(element_id)
    if not el:
        # Help the model recover by showing a few available IDs.
        elements = getattr(RUNTIME, "last_elements", None) or []
        preview = [
            {"id": e.get("id"), "tag": e.get("tag"), "text": (e.get("text") or "")[:60]}
            for e in elements[:15]
        ]
        return f"click_element failed: unknown element_id {element_id}. Available (first 15): {json.dumps(preview, ensure_ascii=True)}"

    selector = _preferred_selector_for_element(el)
    if selector and await _is_unique_visible_locator(page, selector):
        try:
            target = page.locator(selector).first
            await target.wait_for(state="visible", timeout=4000)
            await target.scroll_into_view_if_needed(timeout=3000)
            await target.click(timeout=5000)
            post = json.loads(await capture_multimodal_observation())
            _append_process_log(
                {
                    "stage": "action",
                    "url": page.url,
                    "tool": "click_element",
                    "element_id": element_id,
                    "mode": "selector",
                    "selector": selector,
                }
            )
            return (
                f"Clicked element_id={element_id} via selector={selector}; "
                f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
            )
        except Exception:
            pass

    x = int(el.get("x", 0)) + int(el.get("width", 1)) // 2
    y = int(el.get("y", 0)) + int(el.get("height", 1)) // 2
    await page.mouse.click(x, y)
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "click_element",
            "element_id": element_id,
            "mode": "coordinates",
            "x": x,
            "y": y,
        }
    )
    return (
        f"Clicked element_id={element_id} (x={x}, y={y}); "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def type_text_element(
    element_id: str, text: str, press_enter: bool = False
) -> str:
    """Type using a stable element_id (bbox click to focus, then keyboard typing)."""
    page = await RUNTIME.ensure_page()
    el = _lookup_element(element_id)
    if not el:
        elements = getattr(RUNTIME, "last_elements", None) or []
        preview = [
            {"id": e.get("id"), "tag": e.get("tag"), "text": (e.get("text") or "")[:60]}
            for e in elements[:15]
        ]
        return f"type_text_element failed: unknown element_id {element_id}. Available (first 15): {json.dumps(preview, ensure_ascii=True)}"

    selector = _preferred_selector_for_element(el)
    tag = str(el.get("tag") or "").lower()
    input_type = str(el.get("input_type") or "").lower()
    editable_kind = tag in {"input", "textarea", "select"} or str(
        el.get("role") or ""
    ).lower() in {
        "textbox",
        "combobox",
    }
    read_only_input = input_type in {"date", "time", "datetime-local"}

    if (
        selector
        and editable_kind
        and not read_only_input
        and await _is_unique_visible_locator(page, selector)
    ):
        try:
            target = page.locator(selector).first
            await target.wait_for(state="visible", timeout=5000)
            await target.scroll_into_view_if_needed(timeout=3000)
            await target.click(timeout=5000)
            await target.fill(text, timeout=10000)
            if press_enter:
                await target.press("Enter")
            post = json.loads(await capture_multimodal_observation())
            _append_process_log(
                {
                    "stage": "action",
                    "url": page.url,
                    "tool": "type_text_element",
                    "element_id": element_id,
                    "mode": "selector",
                    "selector": selector,
                    "press_enter": press_enter,
                }
            )
            return (
                f"Typed into element_id={element_id} via selector={selector}; "
                f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
            )
        except Exception:
            pass

    x = int(el.get("x", 0)) + int(el.get("width", 1)) // 2
    y = int(el.get("y", 0)) + int(el.get("height", 1)) // 2
    await page.mouse.click(x, y)

    # Clear existing text (best-effort).
    if sys.platform == "darwin":
        await page.keyboard.down("Meta")
        await page.keyboard.press("A")
        await page.keyboard.up("Meta")
    else:
        await page.keyboard.down("Control")
        await page.keyboard.press("A")
        await page.keyboard.up("Control")
    await page.keyboard.press("Backspace")

    await page.keyboard.type(text, delay=10)
    if press_enter:
        await page.keyboard.press("Enter")
    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "type_text_element",
            "element_id": element_id,
            "mode": "keyboard",
            "x": x,
            "y": y,
            "press_enter": press_enter,
        }
    )
    return f"Typed into element_id={element_id}; labeled_screenshot_path={post.get('labeled_screenshot_path')}"


async def select_flight_dates(departure_text: str, return_text: str = "") -> str:
    """Set flight dates by opening date picker and clicking day cells by label/text."""
    page = await RUNTIME.ensure_page()

    async def _open_date_picker() -> bool:
        selectors = [
            'input[aria-label*="Departure"]',
            'input[aria-label*="Depart"]',
            'input[aria-label*="Date"]',
            '[role="button"][aria-label*="Departure"]',
            '[role="button"][aria-label*="Depart"]',
            '[data-testid*="departure"]',
        ]
        for selector in selectors:
            locator = page.locator(selector).first
            if await locator.count() > 0:
                try:
                    await locator.scroll_into_view_if_needed(timeout=3000)
                    await locator.click(timeout=5000)
                    return True
                except Exception:
                    continue
        return False

    async def _pick_date(target_text: str) -> bool:
        if not target_text.strip():
            return True
        target = target_text.strip().lower()
        success = await _evaluate_when_stable(
            page,
            """
            ({ target }) => {
              const isVisible = (el) => {
                if (!el) return false;
                const rect = el.getBoundingClientRect();
                const style = window.getComputedStyle(el);
                return rect.width > 0 && rect.height > 0 && style.visibility !== 'hidden' && style.display !== 'none';
              };
              const candidates = Array.from(
                document.querySelectorAll(
                  '[role="gridcell"], [role="button"], button, [aria-label], [data-date], [data-testid*="date"]'
                )
              );
              for (const el of candidates) {
                if (!isVisible(el)) continue;
                const text = ((el.textContent || '') + ' ' + (el.getAttribute('aria-label') || '')).toLowerCase();
                if (!text.includes(target)) continue;
                if (el.getAttribute('aria-disabled') === 'true' || el.hasAttribute('disabled')) continue;
                el.click();
                return true;
              }
              return false;
            }
            """,
            {"target": target},
        )
        return bool(success)

    opened = await _open_date_picker()
    if not opened:
        return "select_flight_dates failed: could not open a date picker from visible departure/date fields."

    depart_ok = await _pick_date(departure_text)
    if not depart_ok:
        return f"select_flight_dates failed: could not find departure date matching '{departure_text}'."

    return_ok = await _pick_date(return_text) if return_text.strip() else True
    if not return_ok:
        return f"select_flight_dates partial: departure selected, but return date matching '{return_text}' was not found."

    post = json.loads(await capture_multimodal_observation())
    _append_process_log(
        {
            "stage": "action",
            "url": page.url,
            "tool": "select_flight_dates",
            "departure_text": departure_text,
            "return_text": return_text,
        }
    )
    return (
        f"Selected dates departure='{departure_text}' return='{return_text}'; "
        f"labeled_screenshot_path={post.get('labeled_screenshot_path')}"
    )


async def analyze_screenshot_with_vlm(
    prompt: str = "Describe the page and important actionable UI elements.",
) -> str:
    """Analyze latest screenshot using Hugging Face vision-language model API."""
    page = await RUNTIME.ensure_page()
    # Reuse the screenshot captured by capture_observation() for the current step
    # (do not take a second screenshot here, to keep VLM input consistent).
    screenshot = getattr(RUNTIME, "last_screenshot_path", None)
    if screenshot is None or not Path(str(screenshot)).exists():
        screenshot = RUNTIME.screenshot_path()
        await page.screenshot(path=str(screenshot), full_page=False)

    image_b64 = base64.b64encode(screenshot.read_bytes()).decode("ascii")
    provider = os.getenv("VLM_PROVIDER", "hf").strip().lower()

    if provider == "openai":
        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not openai_api_key:
            return "Missing OPENAI_API_KEY. Set it to enable OpenAI VLM analysis."

        openai_model = os.getenv("OPENAI_VLM_MODEL", "gpt-5-mini")
        endpoint = "https://api.openai.com/v1/chat/completions"
        payload = {
            "model": openai_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
        }
        headers = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=90.0) as client:
            response = await client.post(endpoint, headers=headers, json=payload)

        if response.status_code >= 400:
            return f"OpenAI VLM request failed ({response.status_code}): {response.text[:400]}"

        try:
            data = response.json()
        except ValueError:
            return f"OpenAI VLM returned non-JSON response: {response.text[:400]}"

        return json.dumps(
            {
                "provider": "openai",
                "model": openai_model,
                "prompt": prompt,
                "screenshot_path": str(Path(screenshot).resolve()),
                "vlm_output": data,
            },
            ensure_ascii=True,
        )

    # Default: Hugging Face VLM (free-tier-first).
    hf_api_key = os.getenv("HUGGINGFACE_API_KEY", "").strip()
    if not hf_api_key:
        return "Missing HUGGINGFACE_API_KEY. Set it in your environment to enable VLM analysis."

    hf_model = os.getenv("HF_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    endpoint = f"https://api-inference.huggingface.co/models/{hf_model}"
    payload = {
        "inputs": {
            "image": image_b64,
            "text": prompt,
        }
    }
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=90.0) as client:
        response = await client.post(endpoint, headers=headers, json=payload)

    if response.status_code >= 400:
        return f"VLM request failed ({response.status_code}): {response.text[:400]}"

    try:
        data = response.json()
    except ValueError:
        return f"VLM returned non-JSON response: {response.text[:400]}"

    return json.dumps(
        {
            "provider": "hf",
            "model": hf_model,
            "prompt": prompt,
            "screenshot_path": str(Path(screenshot).resolve()),
            "vlm_output": data,
        },
        ensure_ascii=True,
    )


async def reset_browser_task_state() -> str:
    """Clear one-shot answer/session locks so browsing can continue (same browser tab).

    Call after the model finalized with action_answer, when the user sends a new request
    and capture_multimodal_observation did not yet receive distinct task text.
    """
    RUNTIME.session_done = False
    RUNTIME.answer_locked = False
    RUNTIME.observation_streak = 0
    RUNTIME._last_user_task_fp = None
    _append_process_log(
        {"stage": "session_scope", "reason": "manual_reset_browser_task_state"}
    )
    return (
        "Browsing task locks cleared (session_done/answer_locked). "
        "Continue with capture_multimodal_observation and actions."
    )


async def capture_multimodal_observation(
    vlm_prompt: str = _DEFAULT_MULTIMODAL_PROMPT,
) -> str:
    """Create a fused observation that combines DOM grounding and VLM understanding."""
    _maybe_reset_session_for_new_task_prompt(vlm_prompt)

    if getattr(RUNTIME, "session_done", False):
        page = await RUNTIME.ensure_page()
        done_payload = {
            "observation_type": "multimodal_fused",
            "url": page.url,
            "title": await _safe_page_title(page),
            "session_done": True,
            "action_hint": "Task already concluded. Do not continue browsing in this session.",
        }
        _append_process_log(
            {"stage": "multimodal_observation", "url": page.url, "data": done_payload}
        )
        return json.dumps(done_payload, ensure_ascii=True)

    RUNTIME.observation_streak = getattr(RUNTIME, "observation_streak", 0) + 1
    required_from_prompt = _extract_domains_from_text(vlm_prompt)
    if required_from_prompt:
        if getattr(RUNTIME, "required_domains", None) is None:
            RUNTIME.required_domains = set()
        RUNTIME.required_domains.update(required_from_prompt)
    # Build general task constraints from task text (LLM-first, regex fallback).
    await _update_task_constraints_from_text(vlm_prompt, source="observation_prompt")
    observation = json.loads(await capture_observation())
    vlm_raw = await analyze_screenshot_with_vlm(vlm_prompt)

    try:
        vlm_data = json.loads(vlm_raw)
    except ValueError:
        vlm_data = {"error": vlm_raw}

    fused = {
        "observation_type": "multimodal_fused",
        "url": observation.get("url"),
        "title": observation.get("title"),
        "screenshot_path": observation.get("screenshot_path"),
        "labeled_screenshot_path": observation.get("labeled_screenshot_path"),
        "elements": observation.get("elements", []),
        "web_elements_text": observation.get("web_elements_text", []),
        "vlm_analysis": vlm_data,
        "observation_streak": RUNTIME.observation_streak,
        "observation_cached": bool(observation.get("observation_cached", False)),
        "required_domains": sorted(getattr(RUNTIME, "required_domains", None) or []),
        "task_constraints": getattr(RUNTIME, "task_constraints", None) or {},
        "task_constraints_meta": getattr(RUNTIME, "task_constraints_meta", None) or {},
        "action_hint": (
            "Observation-only loop detected. Execute a concrete action now (scroll/click/input)."
            if RUNTIME.observation_streak >= 3
            else "Proceed with one concrete action."
        ),
    }
    _append_process_log(
        {
            "stage": "multimodal_observation",
            "url": observation.get("url"),
            "data": fused,
        }
    )
    return json.dumps(fused, ensure_ascii=True)
