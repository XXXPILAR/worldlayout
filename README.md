# WorldLayout

The layout stage of the pipeline: given a room and its furniture list (with FastFill's sizes), it places every object
legally and so that a robot can do the room's tasks. **One placement, one program check, one LLM check**
(`workflow/simple.py`). The code keeps its working name, FastFill workflow 2 (packages `workflow` and `fastfill`).
This repository holds the workflow code only, without the experiments.

**Same code as infinite-task-world's copy** (aligned 2026-11-11 with GitHub `Slian22/infinite-task-world` main
05ef1d65, `thirdparty/worldedge_delivery/fastfill`, which is our fastfill-workflow cf4c161 with the collaborators'
patches). The collaborators' changes taken over: the step ⑤ review with bbox images, task context and the rejected
edits' reasons (`llm.py`, `loop.py`, `pipeline.py`, `semantic.py`, `simple.py`, new `visual.py`, their tests,
Pillow in `requirements.txt`), and `fastfill/v2/ops/layout_quality.py` (the 0.8 m door depth read without Torch).
Every code file is byte-identical to theirs except:
- `workflow/credentials.py` and its test in `tests/test_simple.py`: where this folder's key file is (below);
- this README.

Their copy also has `run.py` (their launcher), `SOURCE_MANIFEST.json`, `docs/pipeline.md` and
`examples/task_rooms.jsonl`; this repository has `fastfill/SOURCE.json`. The Panel A run in infinite-task-world used
this code through their bridge.

1. **Plan**: one LLM call for supports, groups and relations (`planning.make_plan`).
2. **Sizes**: one per object, from FastFill when given (or `fastfill_sizes.FastFillSizer`), otherwise from the size
   catalog. No step ever resizes an object.
3. **Place**: one LLM call places every object in the room (`drafting.make_draft`) and decides every facing. FastFill
   gives the sizes only: its yaw is trained modulo π, so it gives a box axis, not which way the front faces (73% of
   its wall-side furniture faces into the room, against 92% for GPT-5.6). `yaw_mode="prompt"` still passes it to the
   LLM as a suggestion.
4. **Check**: the program checks once (`critic.Critic`). If there are hard violations, the repair loop
   (`loop.Loop`) takes each of them up:
   - hard violations: collision, out of bounds, missing support, blocked door, a wall object over a window, a broken
     hard input constraint, and with a robot the robot checks;
   - the program search keeps sizes fixed;
   - soft costs are not chased and there is no polish, so any part of the layout that is already legal stays as the
     LLM placed it.
5. **Review**: the LLM reviews the whole room (`semantic.SemanticPass`), and only severe issues are fixed. Its review,
   fix and pairwise requests carry annotated bbox top-down images (`visual.layout_image`; ids and front arrows, not
   mesh renders) and run at reasoning effort xhigh. Without a `task_context` it is one round. With one (the original
   task and why each required object is needed, passed by infinite-task-world's bridge), the review sees the task and
   runs up to two rounds; the second round also gets the reasons the first round's edits were rejected. A fix is kept
   only if all of the following hold:
   - the program repairs it;
   - it passes the program gate: no new hard violation, and reach, access and doors are no worse;
   - a blind pairwise comparison prefers it in both orders.
6. **Result**: the layout, plus the layout after each step (`diagnostics.stages`), so a bad result can be traced back
   to the step that caused it.

## Task mode: required and expansion furniture

The upstream task agent sends, per room: the room shell and type, the full furniture list (every object with a
detailed description and its support), which objects the task **requires**, which ones the task **opens**
(articulation), and hard constraints. Every other object is **expansion** furniture. The task mode places all of them,
with this order of priority: legality, then the robot's reach, then the task.

- **Required objects** are never removed and never shrunk. Their violations are repaired first, and they must end
  legal (no collision, inside the room, no blocked door, constraints met) and reachable (the robot reaches them, items
  on furniture are within arm's reach, articulation objects can open).
- **Expansion objects** are checked and repaired the same way. An expansion object is removed only to give way to a
  required object (after moving it anywhere in the room failed), or at the end when its own physical violation cannot
  be fixed (④d).

Steps added to `workflow/simple.py` (on top of steps 1-6 above):

| Step | What it does |
|---|---|
| 3 place | the placement request marks required objects (`"required": true`, `drafting.REQUIRED_RULE`) |
| 4 check | the robot checks cover every object plus the task's (`simple.task_robot`); the repair takes the required objects' violations first (`Loop(first=...)`) |
| 4b give way | an expansion object in a required object's violation is moved anywhere in the room, else removed with what stands on it; when several expansion objects block together, the nearest ones are tried (at most `NEAR_BLOCKER_TRIES`), a removal kept only if it helps; one more repair pass after removals |
| 4c required first | if a required object still has a violation: the required objects alone are repaired (or searched for from scratch), then the expansion objects come back one at a time, each kept only where the required objects stay as good (`carry` moves objects standing on required furniture with it) |
| 4d left out | after the review: an expansion object still in a physical violation (collision, out of bounds, no support, blocking a door or a window) goes to a free spot on its parent's top (`free_slot`, `repack`), else it is removed |
| 4e wall fronts | a wall object whose front floor strip, as SceneEval's accessibility scores it, is under 55% free (it hangs over furniture) slides along the walls to the nearest place at least 65% free, with what stands on it, if that adds no violation and breaks no relation (`clear_wall_fronts`); nothing on the floor moves and nothing is removed |
| 7 fallback | if a required object is still not legal and reachable, or the workflow fails on an error: only the required objects, laid out by the program without the LLM (`simple.required_only`), legality and reach before the constraints; returned when it leaves the required objects better off, with `status` `fallback_required_only` |

**Input.** The request is the usual `fastfill.v2` condition; every object needs a description beyond its category
(`contract.check_request`). The task's ids are passed beside it: `required=ids, articulation=ids`.

**Output.** The layout holds the objects kept. `diagnostics.task` gives `required`, `articulation`, `expansion`,
`expansion_kept`, `moved` (each with the step that moved it), `removed` (each with what stood on it and the
violations it was removed for), `required_first` (whether step 4c ran and helped), `wall_clear` (step 4e: the wall
objects, those under the 55%, what moved where and how far, what found no place), `expansion_issues`, `required_ok`
and `required_issues`. A fallback result also keeps the full layout and its diagnostics in
`diagnostics.fallback_from` and lists the constraints it could not meet (`unmet_constraints`).

**When the fallback runs.** Only when the steps above leave a required object with a violation (1 room of the 150 of
the final test150doors run), or on an unexpected error. A busy or failing relay does not trigger it: LLM calls are
retried (`WORKFLOW_LLM_ATTEMPTS=0`: until they pass), and a step whose call fails falls back on its own (rules plan,
default draft positions, no review). A rejected key or an exhausted quota (`FatalAPIError`) still stops the run.

**Known limits.**
- Constraint types: `on`, `between`, `against_wall`, `faces`, `faces_direction`, `near`, `clearance`, `keepout`;
  any other type is refused at the input.
- Constraints that involve a removed expansion object are dropped with it; they are not reported yet.
- Contradictory constraints are the upstream agent's to avoid; the fallback still delivers the required objects
  legal and reachable and lists the constraints it could not meet.
- Large rooms take minutes (with a 180 s limit on the program search per pass); median 200 s per room on
  test150doors.

## Run

The whole path from the command line (`workflow/cli.py`), with the paper's settings as defaults:

```bash
# both steps on one machine (a GPU, or Apple's MPS, and the LLM API)
python -m workflow run --rows rows.jsonl --index 61 \
    --checkpoint <models>/main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500 --out runs/room61
# or in two: FastFill's sizes where the GPU is, then the placement where the API is
python -m workflow sizes --rows rows.jsonl --checkpoint <run>/model-step-4500 --out my_sizes.jsonl
python -m workflow run --rows rows.jsonl --sizes-file my_sizes.jsonl --workers 4 --out my_out
```

- `--checkpoint`: the FastFill model-step folder (`backbone/` with the weights, `geometry_model.pt`,
  `model_config.json`, `checkpoint_manifest.json`; `tokenizer/` beside it in the run folder). The paper's model is
  the 1.7B full fine-tuning on v4detail with every room, `main4-cell05-v4detail-e5-1.7b-all-full/model-step-4500`
  (about 6.9 GB), public on HuggingFace: `hf download XXXpilar/fastfill-v4detail-1.7b-full --local-dir <models>`.
- `--device auto` uses CUDA, else Apple's MPS, else the CPU. `--draft-sizes` runs without FastFill (the placement
  call gives every size); `--no-llm` makes no LLM call.
- `--workers N` places N rooms at a time; `--rpm 30` keeps all of them together under 30 requests a minute. A room
  whose `room_XXX.json` exists is skipped, so a stopped run is resumed by running it again.
- Output: `<out>/room_XXX.json` (the result, where its sizes came from, the LLM calls and their cost),
  `<out>/llm_calls/room_XXX.jsonl` (every call, never the key), `<out>/summary.json`.

From Python:

```python
from workflow.credentials import api_settings
from workflow.pipeline import run_draft_workflow
from workflow.robot import Robot

result = run_draft_workflow(
    condition, sizes,                       # a fastfill.v2 condition; {id: [depth, width, height]} from FastFill
    robot=Robot(width_m=.5, arm_m=.4), robot_margin=.1, hard_zones=True, relocate=True,
    drafts=2, repair_rounds=3, repair_seconds=180,
    required=set(task["required_ids"]), articulation=set(task["articulation_ids"]),
    **api_settings("relay", base_url="https://api.kbsfdxc.net/codex/v1", model="gpt-6.1-sol"))
```

These are the settings of the test150doors runs (reasoning effort `WORKFLOW_REASONING_EFFORT=medium`; the step ⑤
calls with images always use xhigh). The command line passes no `task_context`. infinite-task-world's bridge passes a
row's `task_context` as `run_draft_workflow(..., task_context=...)`, which is how the Panel A run used it. `api_settings`
gives `model`, `base_url` and `api_key`: `"relay"` reads the key in place from `~/.fastfill_api.env`; leave it out to
use the official API settings file (below). `result` is `{status, layout, diagnostics, timing_s}`; `status` is
`clean`, `best_effort` (violations left; listed in `diagnostics.hard_issues`) or `fallback_required_only`.
`llm=False` runs without any LLM call (rules plan, program placement), as the tests do.

- **API settings** (`workflow/credentials.py`): taken from `<this folder>/.env`; if that file does not exist, a
  `.env` in a sibling `Fastfill workflow/` folder is read in place, otherwise `~/.fastfill_openai.env`. The key is never printed or copied.
  Never commit a `.env`.
- **Request limit**: `WORKFLOW_RPM` (with `WORKFLOW_RPM_FILE`, shared by processes) keeps every request under that
  many per minute.
- **LLM call logs**: with `WORKFLOW_LLM_LOG_DIR` set, every call (instructions, prompt, raw answer, usage, cost,
  latency; never the key) is written there (`workflow/llm.py`).
- **The FastFill package**: the code imports the copy in `fastfill/` (else a sibling `fastfill-main/`, or the
  repository this folder sits in; `workflow/contract.py`). Python packages: `requirements.txt` (Python 3.11); the model
  was saved with transformers 5.14.1, and transformers 4.x cannot read its tokenizer. Pillow draws the review's
  images (`requirements.txt` asks for 13.x, as infinite-task-world does; the tests also pass with 9.5).
- **Tests**: `python -m pytest -q tests` (120 tests, no LLM call; `test_run_cli.py` runs the whole path with a tiny FastFill model).

## Files

| File | What it does |
|---|---|
| `workflow/cli.py` | the command line, `python -m workflow run` / `sizes`: FastFill's sizes, then the workflow |
| `workflow/pipeline.py` | entry point `run_draft_workflow`: the plan, the sizes, then `simple.run_simple` (the task-mode fallback on an error) |
| `workflow/simple.py` | the workflow: steps ③-⑦, the task mode's 4b-4e and the fallback |
| `workflow/planning.py` | step ①: supports, groups, relations, priority (LLM, checked; rules when it fails) |
| `workflow/catalog.py` | the train-only size catalog (`size_catalog.json`), for objects without a FastFill size |
| `workflow/fastfill_sizes.py` | step ② in process: FastFill's sizes from a checkpoint |
| `workflow/drafting.py` | step ③: the placement request and its answer |
| `workflow/semantic.py` | step ⑤: the LLM review, its fixes and the pairwise votes |
| `workflow/visual.py` | step ⑤'s images: the annotated bbox top-down view of a layout (the collaborators' module) |
| `workflow/geometry.py` | the geometric checks: footprints, collisions, support, walls, doors, windows, keepouts |
| `workflow/critic.py` | the critic: hard violations and soft costs, and the rule every change is kept or rolled back by |
| `workflow/robot.py` | the robot checks: doors linked, furniture reached, arm reach, doors and drawers free to open |
| `workflow/solver.py` | the candidate search (beam search) the repair uses |
| `workflow/loop.py` | the repair loop: each violation taken up, scope grown, blockers relocated |
| `workflow/contract.py` | the input and output format (`fastfill.v2`) |
| `workflow/orders.py` | kept for callers of the earlier interface: `orders.critic_geometry` (= `critic.critic_geometry`) |
| `workflow/llm.py`, `workflow/credentials.py` | LLM requests (retries, request limit, usage and cost) and where their settings come from |
| `critic_templates.json`, `critic_wall_rules.json` | furniture pair templates and wall rules mined from the training rooms (soft costs) |
| `size_catalog.json` | the size catalog |
| `docs/program_repair_rules.md` | every rule of the program repair, with its threshold and where it is in the code (in Chinese) |
| `examples/bedroom.json` | an example request |
| `fastfill/` | the FastFill files the workflow and FastFill inference need (28 files, unchanged apart from `v2/ops/layout_quality.py`'s import; `fastfill/SOURCE.json`: where from, their hashes, the change) |

## Robot checks (`workflow/robot.py`)

The program checker also asks whether a mobile robot of a given width can use the room (2D footprints only, about
2 ms per room):

- **`robot_door`:** every door is linked to the others through floor the robot can drive on.
- **`robot_reach`:** the robot can reach the furniture it has to work at: furniture carrying another requested
  object, openable furniture, and the task's objects (`Robot.reach_ids`).
- **`robot_openable`:** the sweep in front of a cabinet, wardrobe, fridge and so on is free, so its doors and drawers
  can open.
- **`robot_arm`:** an object standing on furniture is within the arm's 0.85 m (in plan) of a spot the robot can drive
  to. This follows EmbodiedGen's `check_reachable`.

The navigation check computes the same free space as EmbodiedGen's `RoamTrajectoryGenerator` (floor minus furniture,
keeping a clearance), but exactly instead of on a 5 cm grid.

With `robot=...`, these checks are hard violations, the result reports them for that robot (`diagnostics.robot`),
and the robot is also part of placement and repair:

- **Placement request** (`drafting.ROBOT_RULE`): the LLM is told the robot's width. Openable furniture carries
  `front_clearance_m`, and furniture that holds objects carries `robot_works_here`.
- **Repair search** (`robot.RobotZones`): the search ranks its candidates with the checks' cheap local form, added to
  `Geometry.soft_score`: the zone in front of openable furniture; a standing strip on one approach side of furniture
  that holds objects; objects on that furniture within arm's reach of the strip. These zones play the role of the
  usage areas in Make it Home and Merrell et al. (2011), and of Infinigen's accessibility cost. Door linking is not
  modelled in the search; only the critic checks it.
- **Objects on furniture**: with a robot, the search also tries spots along each edge of the support's top
  (EmbodiedGen's surface sampling). The arm-reach zone then ranks these spots.
- **Commit rule** (`Critic.better`): decisions run in tiers. Legality comes first, then the robot checks, then the
  soft cost. A robot check therefore never blocks a legality repair. Robot violations are identified by their object,
  not by the list of blockers.

Two more options:

- **`drafts=N`:** makes N placement calls in parallel and keeps the draft with the fewest hard violations of any kind
  (`draft_rank`). Ranking by legality first was wrong: the repair fixes small collisions easily but rarely fixes robot
  violations. This follows the generate-then-verify idea of WeakVerifiers. Every call is paid.
- **`draft_sizes=True`** (no FastFill): the placement call gives every object's size with its pose, and the later
  steps keep the chosen draft's sizes.

Footprints, wall contacts and robot-zone coverage are cached by pose. This made the repair of the largest room about
5 times faster.
