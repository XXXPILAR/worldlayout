"""The workflow's entry point: the plan, the sizes, then ``simple.run_simple``."""

from __future__ import annotations

import time
import traceback

from .credentials import FatalAPIError
from .contract import check_request, fill_room_defaults, validate_condition
from .planning import make_plan


def run_draft_workflow(condition: dict, sizes: dict[str, list[float]] | None, *, catalog_path: str | None = None,
                       llm: bool = True, model: str | None = None, base_url: str | None = None,
                       api_key: str | None = None, repair: bool = True, beam_width: int = 24,
                       candidate_limit: int = 48, sizer=None, yaws: dict[str, float] | None = None,
                       yaw_mode: str = "llm", templates_path: str | None = None, semantic: bool = True,
                       semantic_ask=None, robot=None, drafts: int = 1, robot_margin: float = 0.0,
                       hard_zones: bool = False, relocate: bool = False, repair_rounds: int = 1,
                       repair_seconds: float | None = None, draft_sizes: bool = False,
                       required: set[str] | None = None, articulation: set[str] | None = None,
                       placement: str = "simple", critic: bool = True,
                       task_context: dict | None = None) -> dict:
    """One room: ① the LLM's semantic plan, ② one size per object (``sizes``, or when it is None
    ``sizer.predict(condition)``, a ``fastfill_sizes.FastFillSizer`` run on the request as given; an id without one
    gets the train catalog's, and a room the sizer cannot take is recorded in ``size_error``), then steps ③-⑦ of
    ``simple.run_simple``. A layout is always returned. No step resizes an object.

    ``yaw_mode``: "llm" (default) gives no suggestion: FastFill supplies the sizes only and the LLM decides every
    facing. FastFill's yaw is trained modulo pi (and pi/2 for swappable boxes), so it gives a box axis, not which way
    the front faces: on test300 only 73% of its wall-side furniture faces into the room (GPT-5.6 92%, ground truth
    91%). "prompt" gives the yaws (``yaws``, or the sizer's) to step ③ as ``suggested_yaw_deg``, so the LLM picks
    each position for that facing and may depart from it where it cannot work.

    ``draft_sizes`` (the workflow without FastFill; no ``sizes`` or ``sizer``): step ② is left out and step ③'s
    placement call gives every size with the poses (``simple.run_simple``).

    ``required`` / ``articulation`` (the task mode): the task's required objects are never removed or shrunk and must
    be legal and reachable, expansion objects give way (``simple.run_simple``). The task mode always delivers a room:
    when the plan call fails the rules' plan is used, and when the workflow fails the required objects alone are laid
    out by the program (``simple.required_only``).

    ``task_context`` carries the original task to semantic review without changing the size model's input schema.

    ``placement`` / ``critic``: kept for callers of the earlier interface (infinite-task-world's bridge passes
    ``placement="simple", critic=True``); the simple placement with the critic is the only workflow left."""
    if placement != "simple" or critic is not True:
        raise ValueError("only placement='simple' with the critic remains (the staged, group and critic-free "
                         "workflows are gone)")
    if draft_sizes and (sizes is not None or sizer is not None):
        raise ValueError("draft_sizes needs no sizes or sizer")
    if yaw_mode not in ("prompt", "llm"):
        raise ValueError(f"unknown yaw_mode {yaw_mode!r}")
    from .simple import required_only, run_simple
    start = time.perf_counter()
    validate_condition(condition)
    request = condition
    condition, _notes = fill_room_defaults(condition)
    condition = check_request(condition)
    try:
        plan = make_plan(condition, llm=llm, model=model, base_url=base_url, api_key=api_key)
    except FatalAPIError:
        raise
    except Exception:
        if required is None:
            raise
        plan = make_plan(condition, llm=False)  # the task mode goes on with the rules' plan
    planned = time.perf_counter()
    size_error = None
    if sizes is None and sizer is not None:
        try:
            sizes, yaws = sizer.predict(request)
        except ValueError as exc:  # over the model's object or context budget
            size_error = f"{type(exc).__name__}: {exc}"[:300]
    sized = time.perf_counter()
    try:
        result = run_simple(request, sizes, plan=plan, yaws=yaws if yaw_mode == "prompt" else None,
                            catalog_path=catalog_path, llm=llm, model=model, base_url=base_url, api_key=api_key,
                            repair=repair, semantic=semantic, semantic_ask=semantic_ask,
                            templates_path=templates_path, beam_width=beam_width, candidate_limit=candidate_limit,
                            robot=robot, drafts=drafts, robot_margin=robot_margin,
                            hard_zones=hard_zones, relocate=relocate, repair_rounds=repair_rounds,
                            repair_seconds=repair_seconds, draft_sizes=draft_sizes, required=required,
                            articulation=articulation, task_context=task_context)
    except FatalAPIError:
        raise  # a rejected key or an exhausted quota stops the run (the caller decides)
    except Exception as exc:
        if required is None:
            raise
        # the task mode always delivers a room: the required objects alone, laid out by the program
        result = required_only(request, sizes, required, articulation or set(), plan=plan, robot=robot,
                               robot_margin=robot_margin, hard_zones=hard_zones, catalog_path=catalog_path,
                               templates_path=templates_path, beam_width=beam_width,
                               candidate_limit=candidate_limit, repair_seconds=repair_seconds,
                               repair_rounds=repair_rounds,
                               reason=f"the workflow failed: {type(exc).__name__}: {exc}"[:300])
        result["diagnostics"]["workflow_error"] = traceback.format_exc()[-2000:]
    result["diagnostics"].update(size_error=size_error,
                                 yaw={"mode": yaw_mode if yaws else "llm", "suggested": len(yaws or {})})
    result["timing_s"].update(plan=round(planned - start, 3), size=round(sized - planned, 3),
                              total=round(time.perf_counter() - start, 3))
    return result
