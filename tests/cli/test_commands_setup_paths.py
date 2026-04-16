"""Tests for setup path resolution behavior in CLI commands."""

from src.bacterai.cli import commands
from src.bacterai.cli import prompts


def test_missing_path_uses_current_location_and_folder_name(monkeypatch, temp_experiment_dir):
    monkeypatch.setattr(commands.Path, "cwd", lambda: temp_experiment_dir)
    monkeypatch.setattr(prompts, "prompt_missing_experiment_path_source", lambda: "current")
    monkeypatch.setattr(prompts, "prompt_experiment_folder_name", lambda: "experiment_alpha")

    experiments = commands.prepare_experiment_directories([{"experiment_path": None}])

    assert len(experiments) == 1
    exp_dir = experiments[0][1]
    assert exp_dir == (temp_experiment_dir / "experiment_alpha").resolve()
    assert exp_dir.exists()
    assert exp_dir.is_dir()


def test_missing_path_uses_full_path_and_creates_directory(monkeypatch, temp_experiment_dir):
    target_path = temp_experiment_dir / "nested" / "experiment_beta"

    monkeypatch.setattr(prompts, "prompt_missing_experiment_path_source", lambda: "full_path")
    monkeypatch.setattr(prompts, "prompt_full_experiment_path", lambda: target_path)

    experiments = commands.prepare_experiment_directories([{"experiment_path": None}])

    assert len(experiments) == 1
    exp_dir = experiments[0][1]
    assert exp_dir == target_path.resolve()
    assert exp_dir.exists()
    assert exp_dir.is_dir()


def test_existing_empty_directory_does_not_prompt_overwrite(monkeypatch, temp_experiment_dir):
    existing = temp_experiment_dir / "experiment_empty"
    existing.mkdir(parents=True)

    called = {"overwrite": 0}

    def _should_not_be_called(_):
        called["overwrite"] += 1
        return False

    monkeypatch.setattr(prompts, "prompt_overwrite_nonempty_directory", _should_not_be_called)

    experiments = commands.prepare_experiment_directories([{"experiment_path": str(existing)}])

    assert len(experiments) == 1
    assert experiments[0][1] == existing.resolve()
    assert called["overwrite"] == 0


def test_existing_nonempty_directory_overwrite_yes_recreates(monkeypatch, temp_experiment_dir):
    existing = temp_experiment_dir / "experiment_nonempty"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old")

    monkeypatch.setattr(prompts, "prompt_overwrite_nonempty_directory", lambda _: True)

    experiments = commands.prepare_experiment_directories([{"experiment_path": str(existing)}])

    assert len(experiments) == 1
    exp_dir = experiments[0][1]
    assert exp_dir == existing.resolve()
    assert exp_dir.exists()
    assert list(exp_dir.iterdir()) == []


def test_existing_nonempty_directory_overwrite_no_prompts_new_full_path(monkeypatch, temp_experiment_dir):
    existing = temp_experiment_dir / "experiment_nonempty"
    existing.mkdir(parents=True)
    (existing / "old.txt").write_text("old")

    replacement = temp_experiment_dir / "replacement" / "experiment_gamma"

    monkeypatch.setattr(prompts, "prompt_overwrite_nonempty_directory", lambda _: False)
    monkeypatch.setattr(prompts, "prompt_full_experiment_path", lambda: replacement)

    experiments = commands.prepare_experiment_directories([{"experiment_path": str(existing)}])

    assert len(experiments) == 1
    exp_dir = experiments[0][1]
    assert exp_dir == replacement.resolve()
    assert exp_dir.exists()
    assert exp_dir.is_dir()
