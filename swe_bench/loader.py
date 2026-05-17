"""
swe_bench/loader.py
───────────────────
Load and iterate over SWE-bench Lite instances.

SWE-bench Lite: 300 real GitHub issues from popular Python repositories,
each with a verified patch that makes all tests pass.

Schema per instance:
  instance_id   : str   — unique identifier e.g. "django__django-12345"
  repo          : str   — "owner/repo"
  base_commit   : str   — SHA of the commit where the bug exists
  problem_statement : str — the GitHub issue text
  patch         : str   — gold unified diff (the correct fix)
  test_patch    : str   — tests that were added / modified to verify the fix
  PASS_TO_PASS  : list  — tests that must still pass
  FAIL_TO_PASS  : list  — tests that must now pass (previously failing)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)


@dataclass
class SWEInstance:
    """A single SWE-bench problem instance."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str          # gold patch — used only for evaluation
    test_patch: str     # tests that verify the fix
    fail_to_pass: list[str]  # tests that must now pass
    pass_to_pass: list[str]  # regression tests that must still pass
    created_at: str = ""
    version: str = ""
    environment_setup_commit: str = ""

    @property
    def repo_name(self) -> str:
        """e.g. 'django__django' from 'django/django'."""
        return self.repo.replace("/", "__")

    @property
    def org(self) -> str:
        return self.repo.split("/")[0]

    @property
    def project(self) -> str:
        return self.repo.split("/")[1]


def load_swebench_lite(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    max_instances: int | None = None,
    instance_ids: list[str] | None = None,
    cache_dir: Path | None = None,
) -> list[SWEInstance]:
    """
    Load SWE-bench Lite from HuggingFace or a local JSON cache.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split — 'test' (300 issues) or 'dev' (23 issues).
        max_instances: Limit for quick debugging (None = all).
        instance_ids: Filter to specific instance IDs.
        cache_dir: Local cache directory; saves downloaded data as JSON.

    Returns:
        List of SWEInstance objects.
    """
    cache_path: Path | None = None
    if cache_dir is not None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"swebench_lite_{split}.json"

    # ── Try local cache first ─────────────────────────────────────────────
    if cache_path and cache_path.exists():
        logger.info("Loading SWE-bench Lite from local cache: %s", cache_path)
        raw = json.loads(cache_path.read_text())
        instances = [_dict_to_instance(r) for r in raw]
    else:
        logger.info("Downloading SWE-bench Lite from HuggingFace: %s", dataset_name)
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Install 'datasets': pip install datasets"
            ) from exc

        ds = load_dataset(dataset_name, split=split)
        instances = [_dict_to_instance(dict(row)) for row in ds]

        if cache_path:
            logger.info("Saving to cache: %s", cache_path)
            cache_path.write_text(
                json.dumps([_instance_to_dict(i) for i in instances], indent=2)
            )

    # ── Apply filters ─────────────────────────────────────────────────────
    if instance_ids:
        id_set = set(instance_ids)
        instances = [i for i in instances if i.instance_id in id_set]
        logger.info("Filtered to %d instances by ID", len(instances))

    if max_instances is not None:
        instances = instances[:max_instances]

    logger.info("Loaded %d SWE-bench Lite instances (split=%s)", len(instances), split)
    return instances


def iter_instances(
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
    split: str = "test",
    cache_dir: Path | None = None,
) -> Iterator[SWEInstance]:
    """Streaming iterator — useful for large splits."""
    yield from load_swebench_lite(dataset_name, split=split, cache_dir=cache_dir)


# ── Private helpers ───────────────────────────────────────────────────────────

def _dict_to_instance(row: dict) -> SWEInstance:
    return SWEInstance(
        instance_id=row.get("instance_id", ""),
        repo=row.get("repo", ""),
        base_commit=row.get("base_commit", ""),
        problem_statement=row.get("problem_statement", ""),
        patch=row.get("patch", ""),
        test_patch=row.get("test_patch", ""),
        fail_to_pass=_parse_list(row.get("FAIL_TO_PASS", "[]")),
        pass_to_pass=_parse_list(row.get("PASS_TO_PASS", "[]")),
        created_at=row.get("created_at", ""),
        version=row.get("version", ""),
        environment_setup_commit=row.get("environment_setup_commit", ""),
    )


def _instance_to_dict(instance: SWEInstance) -> dict:
    return {
        "instance_id": instance.instance_id,
        "repo": instance.repo,
        "base_commit": instance.base_commit,
        "problem_statement": instance.problem_statement,
        "patch": instance.patch,
        "test_patch": instance.test_patch,
        "FAIL_TO_PASS": json.dumps(instance.fail_to_pass),
        "PASS_TO_PASS": json.dumps(instance.pass_to_pass),
        "created_at": instance.created_at,
        "version": instance.version,
        "environment_setup_commit": instance.environment_setup_commit,
    }


def _parse_list(value: str | list) -> list[str]:
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []
