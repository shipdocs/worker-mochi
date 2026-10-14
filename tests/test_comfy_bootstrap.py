"""Regression test for "'utils' is not a package" during ComfyUI startup.

Builds a minimal fake ComfyUI tree with the same shape as the real one:
``nodes.py`` puts ``comfy/`` first on sys.path (so ``comfy/utils.py`` shadows the
``utils/`` package) and ``server.py`` imports ``nodes`` before ``utils.json_util``.
Each case runs in a fresh interpreter so ``sys.modules`` / ``sys.path`` are clean.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "utils/__init__.py": "",
    "utils/json_util.py": "def merge_json_recursive(a, b):\n    return {**a, **b}\n",
    "comfy/__init__.py": "",
    "comfy/utils.py": "",
    "nodes.py": """
        import os, sys
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), "comfy"))
    """,
    "server.py": """
        import nodes
        from utils.json_util import merge_json_recursive
    """,
}


@pytest.fixture
def fake_comfyui(tmp_path: Path) -> Path:
    for name, content in FILES.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(content))
    return tmp_path


def run(code: str, comfyui_dir: Path) -> subprocess.CompletedProcess:
    # cwd + PYTHONPATH mirror the old entrypoint (WORKDIR / with PYTHONPATH=/ComfyUI).
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(comfyui_dir), "COMFYUI_DIR": str(comfyui_dir)},
        capture_output=True,
        text=True,
        check=False,
    )


def test_fake_tree_reproduces_the_bug(fake_comfyui: Path) -> None:
    result = run("import server", fake_comfyui)
    assert result.returncode != 0
    assert "'utils' is not a package" in result.stderr


def test_prepare_comfyui_imports_fixes_server_import(fake_comfyui: Path) -> None:
    result = run(
        """
        from src.comfy_bootstrap import prepare_comfyui_imports
        prepare_comfyui_imports()
        import server, utils
        print(utils.__file__)
        """,
        fake_comfyui,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(fake_comfyui / "utils" / "__init__.py")


def test_prepare_comfyui_imports_recovers_from_shadowed_utils(
    fake_comfyui: Path,
) -> None:
    result = run(
        """
        import nodes, utils  # utils is now comfy/utils.py
        from src.comfy_bootstrap import prepare_comfyui_imports
        prepare_comfyui_imports()
        import server
        """,
        fake_comfyui,
    )
    assert result.returncode == 0, result.stderr


def test_prepare_comfyui_imports_rejects_missing_comfyui(tmp_path: Path) -> None:
    from src.comfy_bootstrap import prepare_comfyui_imports

    with pytest.raises(RuntimeError, match="not found"):
        prepare_comfyui_imports(tmp_path)


# Arch lists as reported by torch.cuda.get_arch_list() for each build.
TORCH_25_CU124 = ["sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]
TORCH_28_CU128 = ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120"]


@pytest.mark.parametrize(
    ("capability", "arch_list", "expected"),
    [
        ((9, 0), TORCH_25_CU124, True),  # H100
        ((8, 9), TORCH_25_CU124, True),  # L40S / RTX 4090 run sm_86 code
        ((12, 0), TORCH_25_CU124, False),  # RTX 5090 / RTX PRO 6000: the reported bug
        ((10, 0), TORCH_25_CU124, False),  # B200
        ((12, 0), TORCH_28_CU128, True),
        ((10, 0), TORCH_28_CU128, True),
        ((8, 9), TORCH_28_CU128, True),
        ((12, 0), ["sm_90", "compute_90"], True),  # PTX JIT fallback
        ((9, 0), ["sm_90a"], True),
    ],
)
def test_arch_supported(
    capability: tuple[int, int], arch_list: list[str], expected: bool
) -> None:
    from src.comfy_bootstrap import arch_supported

    assert arch_supported(capability, arch_list) is expected
