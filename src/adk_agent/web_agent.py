"""Google ADK multi-agent builder for a multimodal web agent."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google.adk.agents import Agent, LoopAgent

from src.tools.web_tools import (
    action_answer,
    action_back,
    action_click,
    action_google,
    action_input,
    action_scroll,
    action_wait,
    analyze_screenshot_with_vlm,
    capture_observation,
    capture_multimodal_observation,
    click_element,
    close_browser,
    log_step,
    navigate,
    go_back,
    go_google,
    reset_browser_task_state,
    select_flight_dates,
    search_web,
    scroll_by,
    type_text_element,
    verify_task_completion,
    wait,
)


def _create_agent(
    agent_cls, *, name: str, model: str, instruction: str, tools=None, sub_agents=None
):
    """Create ADK agent while tolerating minor API differences between ADK versions."""
    kwargs = {
        "name": name,
        "model": model,
        "instruction": instruction,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if sub_agents is not None:
        kwargs["sub_agents"] = sub_agents

    try:
        return agent_cls(**kwargs)
    except TypeError:
        # Fallback for ADK variants that do not support sub_agents in ctor.
        kwargs.pop("sub_agents", None)
        return agent_cls(**kwargs)


def _create_workflow_agent(agent_cls, *, name: str, sub_agents, **extra):
    """Create ADK workflow agent while tolerating ctor differences."""
    kwargs = {
        "name": name,
        "sub_agents": sub_agents,
        **extra,
    }
    try:
        return agent_cls(**kwargs)
    except TypeError:
        # Some variants may use children/agents instead of sub_agents.
        kwargs.pop("sub_agents", None)
        for field_name in ("agents", "children", "steps"):
            try:
                return agent_cls(name=name, **{field_name: sub_agents})
            except TypeError:
                continue
        raise


def _resolve_llm_model() -> str:
    """Resolve LLM model for ADK agents.

    Uses OpenAI only by default (reliable tool/function calling). Override with ``ADK_MODEL``
    if you need another LiteLLM-supported id (advanced).
    """
    explicit = os.getenv("ADK_MODEL", "").strip()
    if explicit:
        return explicit

    model = os.getenv("ADK_OPENAI_MODEL", "openai/gpt-5-mini").strip()
    return model or "openai/gpt-5-mini"


def build_root_agent():
    """Build a Google ADK multi-agent web automation system."""
    load_dotenv()

    model_name = _resolve_llm_model()
    strict_mode = os.getenv("ADK_STRICT_MODE", "true").strip().lower() == "true"

    if strict_mode:
        strict_instruction = (
            "You are a general multimodal web agent operating in strict mode. "
            "Imagine you are a robot browsing the web, just like humans. Now you need to complete a task. "
            "In each iteration, you will receive an Observation that includes a screenshot of a webpage and some texts. "
            "This screenshot will feature Numerical Labels placed in the TOP LEFT corner of each Web Element. "
            "Carefully analyze the visual information to identify the Numerical Label corresponding to the Web Element that requires interaction, "
            "then follow the guidelines and choose one of the following actions:\n"
            "1. Click a Web Element.\n"
            "2. Delete existing content in a textbox and then type content.\n"
            "3. Scroll up or down. Multiple scrolls are allowed to browse the webpage. "
            "Pay attention: the default scroll is the whole window. If the scroll widget is located in a certain area of the webpage, "
            "then you have to specify a Web Element in that area.\n"
            "4. Wait. Typically used to wait for unfinished webpage processes, with a duration of 5 seconds.\n"
            "5. Go back, returning to the previous webpage.\n"
            "6. Google, directly jump to the Google search page. When you can't find information on some websites, try starting over with Google.\n"
            "7. Answer. This action should only be chosen when all questions in the task have been solved.\n"
            "Correspondingly, Action should STRICTLY follow the format:\n"
            "- Click [Numerical_Label]\n"
            "- Type [Numerical_Label]; [Content]\n"
            "- Scroll [Numerical_Label or WINDOW]; [up or down]\n"
            "- Wait\n"
            "- GoBack\n"
            "- Google\n"
            "- ANSWER; [content]\n"
            "Key Guidelines You MUST follow:\n"
            "1) To input text, no need to click textbox first; directly type content. After typing, ENTER is usually applied.\n"
            "2) Distinguish between textbox and search button; don't type into buttons.\n"
            "3) Execute only one action per iteration.\n"
            "4) Strictly avoid repeating the same action if webpage remains unchanged.\n"
            "5) For complex tasks with multiple requirements, select ANSWER only at the very end after solving all requirements.\n"
            "6) Don't interact with useless elements like login/sign-in/donation when irrelevant.\n"
            "7) Focus on numerical labels in the TOP LEFT corner of each rectangle and do not mix them with page numbers.\n"
            "8) Pay attention to date constraints in tasks; match year, month, and day when needed.\n"
            "9) Use filters/sort + scroll for constraints like cheapest, earliest, highest, lowest.\n"
            "10) If a popup/modal/overlay blocks controls, close it first (X/Close/Done/Not now/Esc) and then continue the task.\n"
            "11) Never type raw URLs or query parameters into on-site search boxes to force sorting/filtering; use actual UI controls.\n"
            "12) After you reach the correct site section for the task, prefer scrolling and in-page controls to extract evidence. "
            "Avoid global chrome (Jobs, Learning, Feed, discovery hubs, sign-in CTAs, random outbound links) unless the task explicitly requires them—those routes often waste steps.\n"
            "Concrete decision policy for any general web task:\n"
            "A) Goal decomposition: split task into subgoals (discover target source -> navigate -> apply constraints -> extract evidence -> finalize).\n"
            "B) Constraint ledger: keep explicit required constraints (entity, time/date, location, numeric bounds, format) and verify each before ANSWER.\n"
            "C) Candidate ranking on search/result pages: prioritize elements whose text/aria-label/href best match required constraints and trusted domains; "
            "deprioritize ads, logins, popups, newsletter prompts, and unrelated navigational chrome.\n"
            "D) Filter drilling strategy: after reaching relevant page, first identify filter/sort controls, then apply highest-impact constraints first "
            "(date/time/range/type), re-observe after each action, and avoid applying multiple uncertain filters without intermediate checks.\n"
            "E) Evidence-first extraction: before ANSWER, capture at least one observation that contains direct evidence for each required field; "
            "if any required field is missing, continue browsing instead of summarizing.\n"
            "F) Recovery rules: if 2 consecutive actions do not change page state or progress, do one of {Scroll, GoBack, Google} and choose a different candidate; "
            "if blocked by captcha/login/permissions/paywall or missing user preference, ask a concise follow-up question.\n"
            "G) Stop condition: choose ANSWER only when all required constraints are satisfied and supported by visible page evidence from the latest observations.\n"
            "H) Never claim you cannot access websites unless there is explicit blocker evidence on the page (captcha/login/paywall/permission) and you explain that evidence.\n"
            "Few-shot action patterns (imitate structure, adapt ids/content to current page):\n"
            "Example 1 (Ecommerce, domain constrained):\n"
            "- Task intent: open amazon.com, find noise-canceling headphones, price under $100, return top-rated 3.\n"
            "- Step pattern: capture_multimodal_observation -> action_google -> capture_multimodal_observation -> "
            'action_input(search_box_id, "amazon.com noise-canceling headphones under 100") -> capture_multimodal_observation -> '
            "action_click(result_for_amazon_id) -> capture_multimodal_observation -> "
            'action_input(amazon_search_box_id, "noise-canceling headphones") -> capture_multimodal_observation -> '
            "action_click(price_filter_under_100_id) -> capture_multimodal_observation -> "
            "action_click(sort_by_rating_or_best_id) -> capture_multimodal_observation -> "
            "action_scroll(WINDOW/down) as needed -> capture_multimodal_observation -> "
            'action_answer("Top 3 with names, ratings, prices from visible evidence").\n'
            "- Rule: once amazon.com is reached, stay on amazon.com unless explicit blocker.\n"
            "Example 2 (Travel, date constraints):\n"
            "- Task intent: find one-way flight with date window and budget constraints.\n"
            "- Step pattern: reach target travel site -> set origin/destination -> open date picker and set valid dates -> "
            "apply stops/budget/time filters -> sort by cheapest -> extract top options from result cards -> one final action_answer.\n"
            "- Rule: if date selection fails, re-observe and retry with fresh ids; do not answer before constraints are visibly applied.\n"
            "Example 3 (General information extraction):\n"
            "- Task intent: gather specific facts from a requested website.\n"
            "- Step pattern: reach requested domain -> navigate to relevant section/page -> scroll for required fields -> "
            "cross-check each required field against visible evidence -> single action_answer.\n"
            "- Rule: if required domain is stated in user query, do not finalize on a different domain.\n"
            "Global completion rule:\n"
            "- Never call action_answer more than once for the same user task. After a successful answer, browsing locks until the user sends a new request; "
            "then capture_multimodal_observation(vlm_prompt=<their new full message>) clears the lock automatically. "
            "If you must issue tools before that observation, call reset_browser_task_state() once.\n"
            "Operational protocol for this ADK environment:\n"
            "- First call capture_multimodal_observation(vlm_prompt=<full original user task text>) to obtain latest labeled elements and preserve domain constraints.\n"
            "- Then call exactly one action tool: action_click/action_input/action_scroll/action_wait/action_back/action_google/action_answer.\n"
            "- Repeat until task is completed.\n"
            "Important: During execution, do not output plain-text Thought/Action blocks as your primary response. "
            "Use tool calls for each step, and only provide a final natural-language answer after sufficient evidence."
        )
        strict_agent = _create_agent(
            Agent,
            name="strict_multimodal_agent",
            model=model_name,
            instruction=strict_instruction,
            tools=[
                capture_multimodal_observation,
                action_click,
                action_input,
                action_scroll,
                action_wait,
                action_back,
                action_google,
                action_answer,
                reset_browser_task_state,
                log_step,
            ],
        )
        try:
            loop_max_iters = int(os.getenv("ADK_LOOP_MAX_ITERS", "15"))
            return _create_workflow_agent(
                LoopAgent,
                name="strict_multimodal_root",
                sub_agents=[strict_agent],
                max_iterations=loop_max_iters,
            )
        except Exception:
            return strict_agent

    # If enabled, run as a single agent with tools (no agent transfer).
    # This avoids provider/tool-schema validation issues around transfer_to_agent
    # and is the most reliable mode for demos.
    if os.getenv("ADK_SINGLE_AGENT_MODE", "true").strip().lower() == "true":
        single_instruction = (
            "You are a WebVoyager-style multimodal web agent. "
            "Follow a strict one-action-per-iteration ReAct loop: Thought -> one Action tool call -> Observation. "
            "When user intent is ambiguous or under-specified, ask a concise follow-up question before taking risky actions. "
            "If website choice matters and user did not specify one, ask for preference (for example Google Flights, Skyscanner, Trip.com) or proceed with a safe default and state it. "
            "Google Search first rule: unless the user provides a website, call search_web(query) first to find a relevant site. "
            "Do not remain on search result pages unless the user asked for SERP analysis; open a relevant result and continue task execution. "
            "After every navigation or action, call capture_multimodal_observation for fused DOM+vision grounding. "
            "Act using element IDs: prefer click_element/type_text_element. "
            "IMPORTANT: element_id values are numeric strings like '0','1','2' taken from the latest observation. "
            "Never invent element IDs; if unsure, capture_observation again and choose from the returned web_elements_text list. "
            "For date pickers and dropdown widgets: click to open the widget first, capture_observation again, then choose day/option cells by fresh element IDs. "
            "Do not try typing into read-only date fields unless the site clearly supports direct text entry. "
            "For flight/travel sites, prefer select_flight_dates(departure_text, return_text) when date widgets are complex. "
            "If a click/type action does not change page state, immediately recapture and try a different element_id rather than repeating the same action. "
            "Use wait(seconds=5) when waiting for dynamic content. Use go_back() to undo navigation mistakes. Use go_google() to restart from Google. "
            "Only finalize success after calling verify_task_completion(...) with task-critical evidence text and getting status='verified'. "
            "For transparency, at each step call log_step(stage, thought, observation, action, values) to log your ReAct loop. "
            "Never place final orders or do irreversible actions unless user explicitly asks. "
            "Before final answer, you MUST reference at least one concrete insight from vlm_analysis and map it to element_id actions. "
            "In your final answer, include a short Process Log with labeled_screenshot_path values and element_id actions."
        )

        single_agent = _create_agent(
            Agent,
            name="multimodal_web_agent",
            model=model_name,
            instruction=single_instruction,
            tools=[
                navigate,
                search_web,
                capture_observation,
                capture_multimodal_observation,
                analyze_screenshot_with_vlm,
                click_element,
                type_text_element,
                select_flight_dates,
                scroll_by,
                wait,
                go_back,
                go_google,
                verify_task_completion,
                log_step,
                close_browser,
            ],
        )
        # Wrap the single agent in a LoopAgent so the system keeps iterating
        # even if the model returns STOP after a partial step.
        try:
            loop_max_iters = int(os.getenv("ADK_LOOP_MAX_ITERS", "15"))
            return _create_workflow_agent(
                LoopAgent,
                name="multimodal_web_loop_root",
                sub_agents=[single_agent],
                max_iterations=loop_max_iters,
            )
        except Exception:
            return single_agent

    planner_instruction = (
        "You are PlannerAgent. The user gives open-ended natural-language web goals. "
        "Break each goal into short, safe steps for web execution, including target website selection when unspecified. "
        "If the task is vague (missing constraints like site, dates, budget, filters, or output format), ask targeted follow-up questions first. "
        "When tradeoffs exist across websites, ask user preference explicitly; if no preference is provided, select a sensible default and mention it. "
        "Google Search first rule: Unless the user explicitly provides a target website, begin by navigating to Google Search "
        "(`https://www.google.com`), entering the appropriate query, and opening the most relevant safe result. "
        "After opening a result, immediately capture a multimodal observation before taking further actions. "
        "Ask for an updated observation whenever confidence is low. Keep plans explicit and actionable."
    )

    browser_instruction = (
        "You are BrowserAgent and you execute actions using tools. "
        "Use this strict loop for transparency: navigate -> capture_multimodal_observation -> act using element_id -> capture_multimodal_observation -> repeat. "
        "Prefer `click_element(element_id)` and `type_text_element(element_id, text)` over raw CSS selectors. "
        "Only use `click(selector)`/`type_text(selector, ...)` when an element_id is not available. "
        "For search boxes, prefer `type_text_element(element_id, query, press_enter=true)` unless the page clearly shows a clickable search/submit button. "
        "For calendar/date widgets, first click the date field, then capture a fresh observation and click explicit day cells by element_id. "
        "If date widget interactions are unstable, call `select_flight_dates(departure_text, return_text)` directly. "
        "Use wait(seconds=5), go_back(), and go_google() as explicit recovery actions. "
        "Before completion, call verify_task_completion(...) and require status='verified'. "
        "Never perform destructive or irreversible actions unless user explicitly asks. "
    )

    vision_instruction = (
        "You are VisionAgent. Use capture_multimodal_observation() as the primary observation API. "
        "Ground every recommendation in element IDs and bounding boxes from the element list. "
        "Output the chosen `element_id` values explicitly (for the BrowserAgent to use). "
        "Use analyze_screenshot_with_vlm() for extra visual semantics when needed."
    )

    coordinator_instruction = (
        "You are RootCoordinatorAgent for a multimodal web workflow. "
        "Delegate planning to PlannerAgent, page understanding to VisionAgent, and web execution to BrowserAgent. "
        "Treat user input as a general task request, not a fixed tool script. "
        "Operate in a ReAct style loop until the task is complete. "
        "For transparency, include a short Process Log in the final report listing each page-step you took: "
        "the tool called (especially capture_observation/capture_multimodal_observation), the resulting labeled_screenshot_path, "
        "and the element_id actions executed for that step."
    )

    planner_agent = _create_agent(
        Agent,
        name="planner_agent",
        model=model_name,
        instruction=planner_instruction,
        tools=[],
    )

    browser_agent = _create_agent(
        Agent,
        name="browser_agent",
        model=model_name,
        instruction=browser_instruction,
        tools=[
            navigate,
            capture_observation,
            capture_multimodal_observation,
            click_element,
            type_text_element,
            select_flight_dates,
            scroll_by,
            wait,
            go_back,
            go_google,
            verify_task_completion,
            close_browser,
        ],
    )

    vision_agent = _create_agent(
        Agent,
        name="vision_agent",
        model=model_name,
        instruction=vision_instruction,
        tools=[
            capture_observation,
            capture_multimodal_observation,
            analyze_screenshot_with_vlm,
        ],
    )

    coordinator_agent = _create_agent(
        Agent,
        name="coordinator_agent",
        model=model_name,
        instruction=coordinator_instruction,
        sub_agents=[planner_agent, vision_agent, browser_agent],
        tools=[],
    )

    # For ADK versions where sub-agents are attached after creation.
    if getattr(coordinator_agent, "sub_agents", None) in (None, []):
        try:
            coordinator_agent.sub_agents = [planner_agent, vision_agent, browser_agent]
        except Exception:
            pass

    use_workflow = os.getenv("ADK_ENABLE_WORKFLOW_AGENT", "true").lower() == "true"
    if not use_workflow:
        return coordinator_agent

    try:
        # IMPORTANT: Keep planner/vision/browser as direct sub-agents of the workflow root.
        # ADK's `transfer_to_agent` tool validates agent_name against *top-level* transferable
        # agents. If we nest these under a SequentialAgent, `browser_agent` may not be eligible
        # for transfer and tool calls will fail validation.
        loop_max_iters = int(os.getenv("ADK_LOOP_MAX_ITERS", "15"))
        root_agent = _create_workflow_agent(
            LoopAgent,
            name="multimodal_web_root_agent",
            sub_agents=[planner_agent, vision_agent, browser_agent],
            max_iterations=loop_max_iters,
        )
        return root_agent
    except Exception:
        # Fall back to coordinator when workflow construction fails (e.g. API mismatch).
        return coordinator_agent
