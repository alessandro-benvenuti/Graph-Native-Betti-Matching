"""Storage migration checks, including recoverability and deletion guards."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from cluster.jean_zay import cleanup_work_environments as cleanup
from cluster.jean_zay.snapshot_environment_requirements import requirements_for


REPO = Path(__file__).resolve().parents[1]


class ScratchPathTests(unittest.TestCase):
    def test_old_exports_are_mapped_and_custom_paths_preserved(self):
        environment = dict(os.environ)
        for key in (
            "UV_CACHE_DIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "PIP_CACHE_DIR",
            "TORCH_HOME", "HF_HOME", "WANDB_CACHE_DIR", "WANDB_DATA_DIR",
        ):
            environment.pop(key, None)
        environment.update({
            "WORK": "/work/project/user", "SCRATCH": "/scratch/project/user",
            "GNBM_A100_VENV": "/work/project/user/venvs/model-fgw",
            "UV_CACHE_DIR": "/work/project/user/.cache/uv",
            "GNBM_A100_EXTENSIONS_DIR": "/work/project/user/.cache/torch-extensions/fgw",
            "HF_HOME": "/custom/shared/hf",
        })
        result = subprocess.run(
            ["bash", "-eu", "-c", 'source "$1"; printf "%s\\n" '
             '"$GNBM_A100_VENV" "$UV_CACHE_DIR" "$GNBM_A100_EXTENSIONS_DIR" '
             '"$XDG_CACHE_HOME" "$WANDB_CACHE_DIR" "$HF_HOME"',
             "bash", str(REPO / "cluster/jean_zay/scratch_paths.sh")],
            env=environment, check=True, capture_output=True, text=True,
        )
        self.assertEqual(result.stdout.splitlines(), [
            "/scratch/project/user/venvs/model-fgw",
            "/scratch/project/user/.cache/uv",
            "/scratch/project/user/.cache/torch-extensions/fgw",
            "/scratch/project/user/.cache",
            "/scratch/project/user/.cache/wandb",
            "/custom/shared/hf",
        ])

    def test_both_environment_entry_points_select_scratch(self):
        for architecture, suffix in (("a100", "a100-torch230"), ("h100", "h100-torch231")):
            with self.subTest(architecture=architecture), tempfile.TemporaryDirectory() as directory:
                root = Path(directory).resolve()
                work, scratch = root / "work", root / "scratch"
                work.mkdir()
                venv = scratch / "venvs" / f"vascular-graph-extraction-{suffix}"
                (venv / "bin").mkdir(parents=True)
                (venv / "bin/python").symlink_to("/usr/bin/true")
                (venv / "bin/activate").write_text("# mocked activation\n")
                environment = dict(os.environ)
                for key in tuple(environment):
                    if key.startswith(("GNBM_", "WANDB_")) or key in (
                        "UV_CACHE_DIR", "XDG_CACHE_HOME", "MPLCONFIGDIR", "TORCH_EXTENSIONS_DIR",
                    ):
                        environment.pop(key)
                environment.update({"WORK": str(work), "SCRATCH": str(scratch),
                                    "TMPDIR": str(scratch / "tmp"), "GNBM_REPO_DIR": str(REPO)})
                result = subprocess.run(
                    ["bash", "-e", "-c", 'module() { :; }; source "$1"; '
                     'printf "%s\\n" "$GNBM_VENV" "$UV_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"',
                     "bash", str(REPO / f"cluster/jean_zay/env_{architecture}.sh")],
                    env=environment, check=True, capture_output=True, text=True,
                )
                lines = result.stdout.splitlines()
                self.assertEqual(lines[0], str(venv))
                self.assertTrue(all(line.startswith(str(scratch)) for line in lines))
                self.assertEqual(list(work.iterdir()), [])


class CleanupTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.work = self.root / "work"
        self.work.mkdir()
        self.target = self.work / "venvs/vascular-graph-extraction-a100-torch230"
        self.target.mkdir(parents=True)
        (self.target / "package.py").write_text("retained source\n")
        (self.target / "package-link.py").symlink_to("package.py")
        os.link(self.target / "package.py", self.target / "package-hardlink.py")
        self.archives = self.root / "archives"
        self.archives.mkdir()

    def archive(self):
        with redirect_stdout(io.StringIO()):
            return cleanup.archive_targets(self.work, [self.target], self.archives)

    def test_archive_verified_before_removal_and_recoverable(self):
        records = self.archive()
        self.assertTrue(self.target.is_dir())
        with redirect_stdout(io.StringIO()):
            cleanup.remove_archived_targets(self.work, records, idle_check=lambda: None)
        self.assertFalse(self.target.exists())
        self.assertTrue(Path(records[0]["archive"]).is_file())
        self.assertEqual((self.archives / "removed.txt").read_text(), str(self.target) + "\n")
        # Extract our own verified fixture using the system tar, as documented.
        subprocess.run(["tar", "-xf", records[0]["archive"], "-C", str(self.work)], check=True)
        self.assertEqual((self.target / "package.py").read_text(), "retained source\n")
        self.assertEqual(os.readlink(self.target / "package-link.py"), "package.py")

    def test_changed_source_is_not_removed(self):
        records = self.archive()
        (self.target / "package.py").write_text("changed since backup\n")
        with self.assertRaisesRegex(RuntimeError, "contents changed"):
            cleanup.remove_archived_targets(self.work, records, idle_check=lambda: None)
        self.assertTrue(self.target.is_dir())

    def test_changed_archive_is_not_removed(self):
        records = self.archive()
        with Path(records[0]["archive"]).open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaisesRegex(RuntimeError, "checksum changed"):
            cleanup.remove_archived_targets(self.work, records, idle_check=lambda: None)
        self.assertTrue(self.target.is_dir())

    def test_new_job_blocks_removal(self):
        records = self.archive()
        def busy():
            raise RuntimeError("Queued/running jobs exist")
        with self.assertRaisesRegex(RuntimeError, "jobs exist"):
            cleanup.remove_archived_targets(self.work, records, idle_check=busy)
        self.assertTrue(self.target.is_dir())

    def test_no_cleanup_outside_allowlist_and_no_directory_symlink_following(self):
        unrelated = self.work / "checkpoints"
        unrelated.mkdir()
        with self.assertRaisesRegex(RuntimeError, "allowlist"):
            cleanup.assert_target(self.work, unrelated)
        external = self.root / "external"
        external.mkdir()
        (self.work / "environments").symlink_to(external, target_is_directory=True)
        (external / "greed-3d-vascular").mkdir()
        with redirect_stdout(io.StringIO()):
            targets = cleanup.discover_targets(self.work, include_other_projects=True)
        self.assertEqual(targets, [self.target])

    def test_other_project_environments_require_explicit_flag(self):
        other = self.work / "environments/greed-3d-vascular"
        other.mkdir(parents=True)
        self.assertNotIn(other, cleanup.discover_targets(self.work, False))
        self.assertIn(other, cleanup.discover_targets(self.work, True))


class PackageSnapshotTests(unittest.TestCase):
    def test_records_extras_without_including_module_cuda_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            venv = Path(directory)
            packages = venv / "lib/python3.11/site-packages"
            for name, version in (("optuna", "4.0"), ("torch", "2.3.0"),
                                  ("MultiScaleDeformableAttention3D", "1.0")):
                metadata = packages / f"{name}-{version}.dist-info"
                metadata.mkdir(parents=True)
                (metadata / "METADATA").write_text(f"Name: {name}\nVersion: {version}\n")
            self.assertEqual(requirements_for(venv), ["optuna==4.0"])


if __name__ == "__main__":
    unittest.main()
