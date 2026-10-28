from __future__ import annotations

import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from case_builder.catalog import CatalogInventory
from case_builder.compiler import (
    compile_case_config,
    instantiate_case,
    validate_case_config,
    write_case_outputs,
)
from case_builder.models import BuildRequest, BuiltCase, ExecutionResult
from case_builder.planner import build_planner


def _log_case(message: str) -> None:
    print(f"[case] {message}", flush=True)


class CaseBuilder:
    def __init__(self, request: BuildRequest):
        self.request = request
        self.rng = random.Random(request.seed)
        self.inventory = CatalogInventory(request.catalog_path)
        if request.map_name is not None and request.map_name != self.inventory.map_name:
            raise ValueError(
                f"Request map_name={request.map_name!r} does not match catalog map_name={self.inventory.map_name!r}."
            )
        self.planner = build_planner(request)

    def build(self) -> BuiltCase:
        target_scene = self.request.target_scene_id or "auto"
        scene_kind = self.request.scene_kind or "any"
        _log_case(
            "build start "
            f"planner={self.request.planner} seed={self.request.seed} "
            f"map={self.inventory.map_name} scene_kind={scene_kind} target_scene={target_scene} "
            f"sampling={self.request.sampling_enabled}"
        )
        skeleton = self.inventory.select_skeleton(
            rng=self.rng,
            scene_kind=self.request.scene_kind,
            require_sampling=self.request.sampling_enabled,
            target_scene_id=self.request.target_scene_id,
            preferred_main_route_ids=self.request.preferred_main_route_ids,
            excluded_main_route_ids=self.request.excluded_main_route_ids,
        )
        plan = self.planner.plan(
            request=self.request,
            skeleton=skeleton,
            inventory=self.inventory,
            rng=self.rng,
        )
        actors = instantiate_case(
            request=self.request,
            skeleton=skeleton,
            plan=plan,
            inventory=self.inventory,
        )
        map_root = (self.request.output_root / skeleton.map_name).resolve()
        case_index = self._allocate_case_index(map_root)
        case_id = f"{case_index:03d}"
        case_dir = map_root / case_id
        scenario_config = compile_case_config(
            request=self.request,
            skeleton=skeleton,
            plan=plan,
            actors=actors,
            sampling_output_root=case_dir.parent,
            sequence_id=case_id,
        )
        validate_case_config(scenario_config, self.inventory)

        built_case = BuiltCase(
            case_id=case_id,
            case_index=case_index,
            case_dir=case_dir,
            scenario_path=case_dir / "config" / "scenario.toml",
            description_path=case_dir / "config" / "scene_description.txt",
            metadata_path=case_dir / "config" / "case.json",
            manifest_path=case_dir / "manifest.json",
            sampling_policy_path=case_dir / "config" / "sampling_policy.toml",
            skeleton=skeleton,
            plan=plan,
            actors=actors,
            scenario_config=scenario_config,
        )
        write_case_outputs(built_case=built_case)
        _log_case(
            "build done "
            f"case_dir={built_case.case_dir} scene={built_case.skeleton.scene_id} "
            f"main_route={built_case.skeleton.main_route.route_id} "
            f"support_routes={len(built_case.skeleton.support_routes)} actors={len(built_case.actors)} "
            f"scenario={built_case.plan.scenario_name}"
        )
        return built_case

    def execute(self, built_case: BuiltCase) -> ExecutionResult:
        env = self._build_execution_env()

        with tempfile.TemporaryDirectory(prefix="case_builder_sim_") as tmp_output_root:
            temp_output_root = Path(tmp_output_root)
            command = [
                "conda",
                "run",
                "-n",
                self.request.conda_env,
                "python",
                "run_simulation.py",
                "--scenario-config",
                str(built_case.scenario_path),
                "--carla-host",
                self.request.carla_host,
                "--carla-port",
                str(self.request.carla_port),
            ]
            if self.request.duration is not None:
                command.extend(["--duration", str(self.request.duration)])
            command.extend(["--output", str(temp_output_root), "--csv"])

            timeout_text = (
                f"{self.request.wall_time_limit_s:.1f}s" if self.request.wall_time_limit_s is not None else "none"
            )
            _log_case(
                "execute start "
                f"case_dir={built_case.case_dir} scenario={built_case.plan.scenario_name} "
                f"timeout={timeout_text} host={self.request.carla_host}:{self.request.carla_port}"
            )

            started = time.time()
            timed_out = False
            timeout_note = ""
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(Path(__file__).resolve().parents[2]),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                    env=env,
                    timeout=self.request.wall_time_limit_s,
                )
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                timeout_note = (
                    f"\n[case_builder] wall-clock timeout after {self.request.wall_time_limit_s:.1f}s; "
                    "terminated run_simulation.py\n"
                )
                completed = subprocess.CompletedProcess(
                    args=command,
                    returncode=124,
                    stdout=(stdout or "") + (stderr or "") + timeout_note,
                )
            elapsed_s = time.time() - started

            if completed.returncode == 0 and not built_case.plan.sampling_enabled:
                self._materialize_simulation_output(temp_output_root, built_case.case_dir / "simulation")

        logs_dir = built_case.case_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "run.log"
        log_path.write_text(completed.stdout, encoding="utf-8")

        result = ExecutionResult(
            command=command,
            returncode=int(completed.returncode),
            log_path=log_path,
            elapsed_s=elapsed_s,
            timed_out=timed_out,
        )
        (logs_dir / "run_result.json").write_text(
            json.dumps(result.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if result.returncode == 0:
            _log_case(
                "execute done "
                f"case_dir={built_case.case_dir} status=executed elapsed={result.elapsed_s:.1f}s"
            )
        elif result.timed_out:
            _log_case(
                "execute timeout "
                f"case_dir={built_case.case_dir} limit={self.request.wall_time_limit_s:.1f}s "
                f"elapsed={result.elapsed_s:.1f}s"
            )
        else:
            _log_case(
                "execute failed "
                f"case_dir={built_case.case_dir} returncode={result.returncode} "
                f"elapsed={result.elapsed_s:.1f}s log={result.log_path}"
            )
        return result

    @staticmethod
    def _materialize_simulation_output(temp_output_root: Path, destination_dir: Path) -> None:
        output_dirs = sorted(path for path in temp_output_root.iterdir() if path.is_dir())
        if not output_dirs:
            return

        source_dir = output_dirs[-1]
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in source_dir.iterdir():
            if not source.is_file():
                continue
            shutil.copy2(source, destination_dir / source.name)

    @staticmethod
    def _allocate_case_index(map_root: Path) -> int:
        if not map_root.exists():
            return 1

        numeric_dirs = []
        for child in map_root.iterdir():
            if not child.is_dir():
                continue
            if re.fullmatch(r"\d{3}", child.name):
                numeric_dirs.append(int(child.name))
        if not numeric_dirs:
            return 1
        return max(numeric_dirs) + 1

    @staticmethod
    def _build_execution_env():
        env = os.environ.copy()
        carla_root = env.get("CARLA_ROOT")
        if not carla_root:
            return env

        pythonapi_root = Path(carla_root) / "PythonAPI"
        pythonapi_carla = pythonapi_root / "carla"
        existing = [entry for entry in env.get("PYTHONPATH", "").split(os.pathsep) if entry]

        prepend = []
        for candidate in (pythonapi_root, pythonapi_carla):
            candidate_str = str(candidate)
            if candidate.exists() and candidate_str not in existing and candidate_str not in prepend:
                prepend.append(candidate_str)

        if prepend:
            env["PYTHONPATH"] = os.pathsep.join(prepend + existing)

        return env
