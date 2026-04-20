# Agentic Scenario Generation Technical Design

## Overview

This document describes the agentic scenario generation pipeline used in this repository for pedestrian social simulation, scenario execution, and data collection.

The system is designed around one goal:

- automatically generate executable, diverse, topology-consistent scenarios
- execute them in CARLA
- collect structured outputs under a unified dataset layout

The design is not based on free-form text-to-config generation. Instead, it follows a topology-first workflow in which scenario semantics are built on top of valid map cells and routes that have already been extracted from the target map.

## Design Principles

### 1. Topology First

The system does not start from unconstrained natural-language scene imagination. It first selects a valid scene cell and route from the topology catalog, and only then expands the scene into actor composition, behavior semantics, and textual description.

This keeps generated scenarios aligned with:

- existing walkable topology
- supported route families
- executable route geometry
- current simulation capabilities

### 2. Separation Between Global Planning and Single-Case Construction

The architecture separates:

- batch-level task selection and coverage control
- single-case construction and execution

This separation reduces coupling between:

- global coverage objectives
- local semantic generation
- execution-time robustness

### 3. LLM as a Constrained Semantic Component

The LLM is used only in a constrained role:

- expanding scenario semantics on top of a selected topology skeleton
- composing realistic actor patterns
- generating natural-language scene descriptions

The LLM does not directly:

- invent arbitrary map geometry
- write final configuration files freely
- control low-level route validity

Configuration compilation, route binding, validation, and execution are handled by deterministic code.

### 4. Dataset-Oriented Output Organization

The system is designed for scalable data generation, not only one-off simulation demos.

All accepted cases are written into a unified directory structure:

- `data/<Town>/<NNN>/...`

This keeps scenario configuration, simulation outputs, sampling outputs, and visualization artifacts co-located and easy to manage.

## System Components

The current implementation consists of two main components.

### ScenarioDirector

`ScenarioDirector` is the batch-level controller.

Its responsibilities are:

- selecting the next map and scene cell to cover
- tracking coarse coverage state across maps, cells, main routes, and ego action buckets
- passing targeted generation requirements to `CaseBuilder`
- running multiple cases until the requested target count is reached

Current implementation path:

- [tools/scenario_director/director.py](/home/zmh/carla-social-force-model/tools/scenario_director/director.py)
- [tools/scenario_director/models.py](/home/zmh/carla-social-force-model/tools/scenario_director/models.py)
- [tools/run_scenario_batch.py](/home/zmh/carla-social-force-model/tools/run_scenario_batch.py)

### CaseBuilder

`CaseBuilder` is the single-case worker.

Its responsibilities are:

- loading one topology catalog
- selecting a valid scene skeleton
- expanding semantics through a heuristic or LLM planner
- instantiating actors
- compiling the final scenario configuration
- executing the scenario
- optionally collecting sampling outputs

Current implementation path:

- [tools/case_builder/builder.py](/home/zmh/carla-social-force-model/tools/case_builder/builder.py)
- [tools/case_builder/catalog.py](/home/zmh/carla-social-force-model/tools/case_builder/catalog.py)
- [tools/case_builder/planner.py](/home/zmh/carla-social-force-model/tools/case_builder/planner.py)
- [tools/case_builder/compiler.py](/home/zmh/carla-social-force-model/tools/case_builder/compiler.py)
- [tools/build_case.py](/home/zmh/carla-social-force-model/tools/build_case.py)

## End-to-End Pipeline

The pipeline can be understood as a four-stage process.

### Stage 1. Offline Topology Preparation

Before agentic generation starts, each target map must be converted into a topology catalog containing:

- usable scene cells
- route candidates
- route metadata
- scene structure needed for later scenario generation

This step produces `walkable_cells.json`, which becomes the base inventory for the rest of the system.

Relevant implementation:

- [utils/generate_walkable_cells.py](/home/zmh/carla-social-force-model/utils/generate_walkable_cells.py)

### Stage 2. Scenario Selection

At generation time, the system does not start from free-form scene text. It first chooses:

- a map
- a scene cell
- a main route
- an ego action target

For batch generation, this step is handled by `ScenarioDirector`.

For single-case generation, this step is handled by `CaseBuilder` directly.

### Stage 3. Semantic Expansion and Compilation

Once a valid topology skeleton is selected, the system expands it into:

- ego actor settings
- support actors
- group patterns
- vehicle background settings
- weather settings
- sampling settings
- scene description text

This intermediate scene plan is then deterministically compiled into the final TOML configuration consumed by `run_simulation.py`.

### Stage 4. Execution and Collection

The compiled configuration is executed in CARLA. If sampling is enabled, the system resolves the declared ego pedestrian, activates the sampling collector, and writes outputs into the case directory.

Execution robustness includes:

- bounded wall-clock case execution
- controlled ego spawn retry
- early termination when ego cannot be resolved for sampling

## Coverage-Driven Batch Generation

The current batch-level logic is intentionally lightweight.

`ScenarioDirector` currently tracks coverage over:

- maps
- cells
- main routes
- ego action buckets

Ego actions are not treated as raw route types only. They are grouped into higher-level semantic buckets such as:

- `sidewalk_straight`
- `junction_straight`
- `crosswalk_cross`
- `unmarked_cross`
- `left_turn`
- `right_turn`
- `u_turn`

These buckets are derived from route metadata and route geometry, and are used to decide which type of ego behavior is still under-covered.

Current state persistence:

- `ScenarioDirector` persists coverage state into a JSON state file
- resuming with the same state file allows continued coverage-aware generation

Current limitation:

- the director currently resumes from its own state file
- it does not yet rebuild state by scanning existing case directories automatically

## Unified Output Layout

Each accepted case is written to:

```text
data/<Town>/<NNN>/
```

For example:

```text
data/Town07/001/
```

The current per-case layout is sequence-like and contains:

```text
data/<Town>/<NNN>/
  manifest.json
  config/
    scenario.toml
    case.json
    scene_description.txt
    sampling_policy.toml
    sampling_resolved.json
    sensor_rig.toml
  simulation/
  sampling/
  visualization/
  logs/
```

This design makes each case self-contained and suitable for:

- dataset curation
- debugging
- replay
- downstream offline processing

## Runtime Robustness Design

### Grounded Route Spawn Initialization

Scene-based pedestrian spawn locations are derived from topology routes. To improve spawn validity, route spawn transforms are generated from map-aware grounded positions during topology export, rather than relying on planar route points alone.

This reduces the gap between:

- route-level semantic anchors
- physically valid CARLA spawn transforms

### Ego-First Sampling Safety

Sampling depends on successful ego resolution. If the declared ego pedestrian cannot be spawned and resolved, the scenario should not continue as an apparently valid collection run.

The runtime therefore applies:

- ego spawn retry for a small number of ticks
- early abort if ego remains unresolved after the retry budget is exhausted

This prevents long-running invalid cases that never activate the sampling collector.

### Wall-Clock Guard

Each case can optionally be limited by a wall-clock timeout. This is especially useful for batch generation where rare cases may otherwise stall execution for too long.

## Command-Line Interfaces

The system exposes two primary entry points.

### 1. Build One Case

Use [tools/build_case.py](/home/zmh/carla-social-force-model/tools/build_case.py) to generate one case.

Example:

```bash
python3 tools/build_case.py \
  --planner llm \
  --model qwen-plus \
  --catalog topology/Town07/walkable_cells.json \
  --output-root ./data \
  --seed 2026 \
  --execute \
  --wall-time-limit 300
```

This command:

- loads the `Town07` topology catalog
- builds one case
- writes it under `./data/Town07/<NNN>/`
- executes the scenario immediately
- limits total wall-clock execution time to 300 seconds

#### Main Parameters

- `--catalog`
  Path to `walkable_cells.json`

- `--output-root`
  Dataset root directory. Cases are written under `data/<map>/<NNN>/`

- `--planner {heuristic,llm}`
  Semantic planner mode

- `--model`
  LLM model name when `--planner llm` is used

- `--seed`
  Deterministic generation seed

- `--scene-kind {crossing,sidewalk}`
  Optional scene kind filter

- `--no-sampling`
  Disable sampling and ego declaration

- `--execute`
  Execute the generated case immediately

- `--duration`
  Optional maximum simulated time passed to `run_simulation.py`

- `--wall-time-limit`
  Wall-clock timeout in seconds for one case execution

- `--conda-env`
  Conda environment used for execution

- `--carla-host`
  CARLA host

- `--carla-port`
  CARLA port

- `--scenario-prefix`
  Prefix for generated `scenario_name`

- `--base-url`
  OpenAI-compatible endpoint for the LLM planner

- `--api-key`
  Optional LLM API key override

### 2. Run Batch Generation

Use [tools/run_scenario_batch.py](/home/zmh/carla-social-force-model/tools/run_scenario_batch.py) to run multiple cases under `ScenarioDirector`.

Example:

```bash
python3 tools/run_scenario_batch.py \
  --output-root ./data \
  --target-count 20 \
  --catalog topology/Town05_Opt/walkable_cells.json \
  --catalog topology/Town07/walkable_cells.json \
  --planner llm \
  --model qwen-plus \
  --execute \
  --wall-time-limit 300 \
  --seed 2026
```

This command:

- loads multiple topology catalogs
- maintains batch-level coverage state
- repeatedly issues single-case tasks
- stops after reaching 20 accepted cases or when no more eligible tasks remain

#### Main Parameters

- `--output-root`
  Dataset root directory

- `--target-count`
  Number of accepted cases to generate

- `--catalog`
  Can be passed multiple times to include several maps

- `--planner {heuristic,llm}`
  Planner mode for delegated case construction

- `--model`
  LLM model when `--planner llm` is used

- `--execute`
  Execute generated cases immediately

- `--no-sampling`
  Disable sampling for generated cases

- `--seed`
  Batch-level deterministic seed

- `--resume`
  Resume from the saved director state file

- `--state-path`
  Optional path to the director state file

- `--wall-time-limit`
  Per-case wall-clock timeout

- `--dry-run`
  Select and print one task without building or executing it

## Recommended Usage Patterns

### Generate Topology First

Before generating cases, prepare topology catalogs for each target map.

Example:

```bash
python3 utils/generate_walkable_cells.py \
  --map-name Town07 \
  --output-dir topology/Town07
```

### Single-Case Debugging First

Before launching a large batch, first run a few single cases to verify:

- CARLA is reachable
- topology is usable
- sampling activates correctly
- outputs are written to the expected location

### Then Use Batch Generation

Once a single-case workflow is validated, switch to `run_scenario_batch.py` for coverage-aware generation across maps.

## Runtime Logging

The current command-line tools print default progress logs to the terminal.

Typical log streams include:

- `[build_case] ...`
- `[case] ...`
- `[scenario_batch] ...`
- `[director] ...`
- `[coverage] ...`

These logs are intended to expose:

- selected map and scene
- target ego action
- case build start and completion
- execution start, timeout, failure, or completion
- coverage progression across the batch

## Important Configuration Notes

### `draw_bounding_boxes`

`draw_bounding_boxes` is a runtime visualization/debug flag under `[walker]`.

When enabled, the simulator draws pedestrian bounding boxes in CARLA. It does not affect:

- route selection
- actor semantics
- sampling annotations
- scenario compilation logic

It is mainly useful for interactive debugging.

### `ego_spawn_retry_ticks`

`ego_spawn_retry_ticks` is a runtime robustness parameter under `[walker]`.

It controls how many ticks the system retries ego spawn before aborting a sampling-enabled case.

Example:

```toml
[walker]
ego_spawn_retry_ticks = 5
ego_spawn_retry_interval = 0.05
```

This allows a small retry window without letting a broken sampling case run for a long time.

## Current Scope

The current implementation already supports:

- topology-first scenario generation
- single-case construction
- coverage-aware batch selection
- unified dataset output layout
- LLM-based constrained semantic expansion
- execution-time safeguards such as wall-clock timeout and unresolved-ego early abort

The current implementation does not yet fully support:

- automatic reconstruction of coverage state from already generated case directories
- multi-worker parallel case generation
- explicit multi-level complexity presets at the batch-director interface

These are natural extensions of the current architecture rather than changes to the overall design.

## Summary

The core contribution of this system is not a free-form prompt-to-config generator. It is a topology-grounded, coverage-aware, agentic pipeline for scenario construction and data collection.

Its central properties are:

- topology-constrained generation
- separation of batch-level direction and single-case construction
- constrained LLM usage
- deterministic compilation into executable simulation configs
- unified dataset-oriented outputs
- robustness-oriented runtime control

Together, these properties make the system suitable not only for scenario authoring, but also for repeatable dataset production at scale.
