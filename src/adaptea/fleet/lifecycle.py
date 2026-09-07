from __future__ import annotations

import os
import platform
import re
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from adaptea.config import Config
from adaptea.fleet.models import FleetInventory, ModelInstance
from adaptea.lmstudio.client import LMStudioClient
from adaptea.lmstudio.lms_cli import CommandResult, run_command


@dataclass(frozen=True, slots=True)
class ResourceEstimate:
    model: str
    estimated_total_bytes: int | None
    estimated_gpu_bytes: int | None
    physical_memory_bytes: int | None
    safely_fits: bool | None
    detail: str


def parse_resource_estimate(
    model: str, text: str, physical: int | None, headroom: float
) -> ResourceEstimate:
    total = _memory_value(text, "Estimated Total Memory")
    gpu = _memory_value(text, "Estimated GPU Memory")
    if total is None or physical is None:
        fits = None
        detail = (
            "Memory estimate or physical-memory capacity is unknown; automatic loading refused."
        )
    else:
        budget = int(physical * (1 - headroom))
        fits = total <= budget
        detail = (
            f"Estimated total memory {total} bytes is within the headroom-adjusted budget."
            if fits
            else f"Estimated total memory {total} bytes exceeds the headroom-adjusted budget."
        )
    return ResourceEstimate(model, total, gpu, physical, fits, detail)


def physical_memory_bytes() -> int | None:
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullTotalPhys)
        except Exception:
            return None
    if not hasattr(os, "sysconf"):
        return None
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        return None
    return int(pages * size) if isinstance(pages, int) and isinstance(size, int) else None


def available_memory_bytes() -> int | None:
    """Return memory currently available to new work, or None when the OS cannot report it."""
    if platform.system() == "Windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):  # type: ignore[attr-defined]
                return int(stat.ullAvailPhys)
        except Exception:
            return None
    if hasattr(os, "sysconf"):
        available_pages: int | None
        page_size: int | None
        try:
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (OSError, ValueError):
            available_pages = page_size = None
        if (
            isinstance(available_pages, int)
            and isinstance(page_size, int)
            and available_pages >= 0
            and page_size > 0
        ):
            return available_pages * page_size
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["vm_stat"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode == 0:
            return parse_vm_stat_available(result.stdout)
    return None


def parse_vm_stat_available(text: str) -> int | None:
    header = re.search(r"page size of\s+(\d+) bytes", text, re.I)
    if not header:
        return None
    page_size = int(header.group(1))
    pages = 0
    found = False
    for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
        match = re.search(rf"^{re.escape(label)}:\s*([0-9.]+)\.", text, re.M)
        if match:
            pages += int(match.group(1))
            found = True
    return pages * page_size if found else None


class InstanceLifecycleManager:
    def __init__(self, config: Config, inventory: FleetInventory) -> None:
        self.config = config
        self.inventory = inventory

    async def estimate(self, model: str, context_length: int | None = None) -> ResourceEstimate:
        command = [self.config.lmstudio.lms_executable, "load", model, "--estimate-only"]
        if context_length:
            command.extend(["--context-length", str(context_length)])
        result = await run_command(*command, timeout=120)
        if result.returncode != 0:
            return ResourceEstimate(
                model,
                None,
                None,
                physical_memory_bytes(),
                None,
                f"LM Studio estimator failed: {_reason(result)}",
            )
        estimate = parse_resource_estimate(
            model,
            result.stdout + "\n" + result.stderr,
            physical_memory_bytes(),
            self.config.fleet.memory_headroom_fraction,
        )
        if estimate.safely_fits is None or not self.inventory.instances:
            return estimate
        sizes = {item.model_key: item.size_bytes for item in self.inventory.downloaded}
        loaded_sizes = [sizes.get(item.model_key) for item in self.inventory.instances]
        if any(size is None for size in loaded_sizes):
            return ResourceEstimate(
                model,
                estimate.estimated_total_bytes,
                estimate.estimated_gpu_bytes,
                estimate.physical_memory_bytes,
                None,
                "Existing loaded-instance memory is not fully known; automatic loading refused.",
            )
        assert estimate.physical_memory_bytes is not None
        assert estimate.estimated_total_bytes is not None
        known_loaded = sum(int(size) for size in loaded_sizes if size is not None)
        budget = int(
            estimate.physical_memory_bytes * (1 - self.config.fleet.memory_headroom_fraction)
        )
        fits = known_loaded + estimate.estimated_total_bytes <= budget
        return ResourceEstimate(
            model,
            estimate.estimated_total_bytes,
            estimate.estimated_gpu_bytes,
            estimate.physical_memory_bytes,
            fits,
            (
                "Estimated additional instance fits after conservative loaded-model accounting."
                if fits
                else "Estimated additional instance exceeds the remaining headroom-adjusted budget."
            ),
        )

    async def load(
        self,
        model: str,
        identifier: str,
        *,
        context_length: int | None = None,
        explicit_override: bool = False,
        requested_by_user: bool = False,
    ) -> ModelInstance:
        """Load one instance, refusing anything the memory estimate cannot justify.

        ``max_loaded_instances`` bounds what Adaptea loads *on its own*: it is written as
        the number of instances the saved fleet asks for, so it says nothing about a model
        the user is turning on right now. Enforcing it against a deliberate click was the
        reason Load reported a full fleet while assigning the same model as Strong or Fast
        quietly succeeded — the assignment raised the ceiling on its way through. Memory is
        the real guard, and ``requested_by_user`` never relaxes it.
        """
        if self.inventory.instance(identifier):
            raise ValueError(f"LM Studio instance identifier already exists: {identifier}")
        if (
            not requested_by_user
            and len(self.inventory.instances) >= self.config.fleet.max_loaded_instances
        ):
            raise RuntimeError("Configured maximum loaded fleet instances has been reached.")
        estimate = await self.estimate(model, context_length)
        if estimate.safely_fits is not True and not explicit_override:
            raise RuntimeError(
                estimate.detail + " Use an explicit override only after reviewing it."
            )
        command = [
            self.config.lmstudio.lms_executable,
            "load",
            model,
            "--identifier",
            identifier,
            "--ttl",
            str(self.config.fleet.instance_ttl_seconds),
        ]
        if context_length:
            command.extend(["--context-length", str(context_length)])
        result = await run_command(*command, timeout=600)
        if result.returncode != 0:
            raise RuntimeError(f"LM Studio model load failed: {_reason(result)}")
        assignment = next((item for item in self.config.fleet.models if item.model == model), None)
        instance = ModelInstance(
            instance_id=identifier,
            model_key=model,
            capability_tier=assignment.tier if assignment else "strong",
            roles=assignment.roles if assignment else ["worker"],
            context_length=context_length,
            parallel_limit=assignment.parallel_limit if assignment else None,
        )
        self.inventory.instances.append(instance)
        return instance

    async def unload(self, instance: ModelInstance) -> None:
        if instance.running_workers:
            raise RuntimeError("Refusing to unload an instance with an active coding worker.")
        async with LMStudioClient(
            self.config.lmstudio.base_url, self.config.lmstudio.api_token, timeout=120
        ) as client:
            await client.unload(instance.instance_id)
        self.inventory.instances = [
            item for item in self.inventory.instances if item.instance_id != instance.instance_id
        ]


def _memory_value(text: str, label: str) -> int | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.,]+)\s*([KMGT]?i?B)", text, re.I)
    if not match:
        return None
    value = float(match.group(1).replace(",", ""))
    units = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    return int(value * units[match.group(2).upper()])


def _reason(result: CommandResult) -> str:
    text = (result.stderr or result.stdout).strip().splitlines()
    return text[-1] if text else f"exit code {result.returncode}"


async def ensure_configured_instances(
    root: Path,
    config: Config,
    inventory: FleetInventory,
    demand: Mapping[str, int] | None = None,
) -> bool:
    """Safely load missing configured instances; never unload or override an uncertain estimate.

    ``demand`` is what the work in front of the fleet can actually keep busy, derived from
    the plan and the measured profile. It only ever lowers the count: the configuration
    stays the ceiling, so a user who asked for two instances never silently gets three.
    """
    manager = InstanceLifecycleManager(config, inventory)
    changed = False
    for configured in config.fleet.models:
        current = [item for item in inventory.instances if item.model_key == configured.model]
        allowed = (
            configured.instances if isinstance(configured.instances, int) else max(1, len(current))
        )
        desired = (
            min(allowed, demand[configured.model])
            if demand and configured.model in demand
            else allowed
        )
        for index in range(len(current) + 1, desired + 1):
            # LM Studio displays the instance identifier as its model name. Base it on the
            # actual model key, not an internal fleet role such as "fast-1".
            model_name = configured.model.rsplit("/", 1)[-1]
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", model_name).strip("-")
            await manager.load(
                configured.model,
                f"adaptea-{safe_name or 'model'}-{index}",
                context_length=configured.context_length,
            )
            changed = True
    return changed
