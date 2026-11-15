"""Repair agent against a throw-away git repo (under .tmp/pytest) and the fake claude CLI."""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from clipper import paths
from clipper.config import load_config
from clipper.supervisor import repair as rp
from clipper.supervisor.repair import RepairAgent, is_protected


def git(repo, *args):
    r = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    return r.stdout.strip()


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "clipper" / "supervisor").mkdir(parents=True)
    (r / "tests").mkdir()
    (r / "config").mkdir()
    (r / "secrets").mkdir()
    (r / "data").mkdir()
    (r / "clipper" / "__init__.py").write_text("")
    (r / "clipper" / "calc.py").write_text("def add(a, b):\n    return a - b\n")  # the planted bug
    (r / "clipper" / "supervisor" / "supervisor.py").write_text("# supervisor\n")
    (r / "tests" / "test_calc.py").write_text(
        "from clipper.calc import add\n\ndef test_add():\n    assert add(2, 3) == 5\n")
    (r / "config" / "locked.yaml").write_text("test_mode: true\n")
    (r / "requirements.txt").write_text("psutil\n")
    (r / "README.txt").write_text("hello\n")
    (r / ".gitignore").write_text("secrets/\ndata/\n__pycache__/\n.pytest_cache/\n*.pyc\n")
    (r / "secrets" / "token.json").write_text("TOKEN")
    git(r, "init", "-q")
    git(r, "config", "user.email", "t@example.com")
    git(r, "config", "user.name", "t")
    git(r, "config", "core.autocrlf", "false")
    git(r, "add", "-A")
    git(r, "commit", "-q", "-m", "init")
    return r


@pytest.fixture
def cfg():
    return load_config(overrides={"supervisor": {"repair_timeout_s": 60, "repair_test_timeout_s": 120}})


def make_agent(cfg, state, repo, **kw):
    kw.setdefault("test_cmd", [str(paths.VENV_PY), "-m", "pytest", "-x", "-q", "-p", "no:cacheprovider",
                               "-m", "not network and not claude"])
    return RepairAgent(cfg, state, repo_root=repo, **kw)


def set_script(tmp_path, monkeypatch, body: str):
    p = tmp_path / "agent_script.py"
    p.write_text(textwrap.dedent(body), encoding="utf-8")
    monkeypatch.setenv("FAKE_CLAUDE_REPAIR_SCRIPT", str(p))


REPORT = {"signature": "AssertionError@clipper/calc.py:add", "type": "AssertionError", "message": "bad add",
          "traceback": "Traceback ...", "ts": 1.0}

FIX = """
    import pathlib
    p = pathlib.Path("clipper/calc.py")
    p.write_text(p.read_text().replace("a - b", "a + b"))
"""


def events(kind):
    p = paths.DATA.logs / "clips.jsonl"
    ev = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()] if p.exists() else []
    return [e for e in ev if e["kind"] == kind]


def test_is_protected():
    for p in ("config/locked.yaml", "secrets/token.json", "client_secret.json", "clipper/supervisor/repair.py",
              "requirements.txt", ".gitignore", "tools/ffmpeg/bin/x.exe", ".venv/Lib/x.py", "./config/locked.yaml",
              "clipper\\supervisor\\x.py", "CONFIG/locked.yaml"):
        assert is_protected(p), p
    for p in ("clipper/worker.py", "tests/unit/test_x.py", "config/settings.yaml", "clipper/supervisorx.py"):
        assert not is_protected(p), p


def test_claude_args_match_spec(cfg):
    a = rp.claude_args(cfg)
    assert a[:7] == ["-p", "--model", cfg.claude.model, "--effort", "high", "--no-session-persistence", "--safe-mode"]
    assert a[a.index("--permission-mode") + 1] == "dontAsk"
    assert a[a.index("--tools") + 1] == "Read,Edit,Write,Grep,Glob,Bash"
    allowed = a[a.index("--allowedTools") + 1]
    assert "Bash(.venv/Scripts/python.exe -m pytest*)" in allowed and "Bash(git diff*)" in allowed
    dis = a[a.index("--disallowedTools") + 1]
    for s in ("Edit(./config/locked.yaml)", "Write(./clipper/supervisor/**)", "Bash(pip*)", "WebFetch",
              "Read(./secrets/**)"):
        assert s in dis


def test_successful_repair_commits(cfg, state, repo, tmp_path, monkeypatch, fake_claude_calls):
    set_script(tmp_path, monkeypatch, FIX)
    head0 = git(repo, "rev-parse", "HEAD")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "log tail text") is True
    assert (repo / "clipper" / "calc.py").read_text() == "def add(a, b):\n    return a + b\n"
    head1 = git(repo, "rev-parse", "HEAD")
    assert head1 != head0
    msg = git(repo, "log", "-1", "--format=%B")
    assert msg.startswith("repair: AssertionError@clipper/calc.py:add")
    assert "Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>" in msg
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["clipper/calc.py"]
    assert git(repo, "status", "--porcelain") == ""
    # holds are set for the duration; the supervisor clears them on success
    assert state.publish_hold() and state.kv_get("hold.publish_reason") == "repair"
    assert state.keep_chunks()
    assert state.kv_get("repair.attempts.test-stream") == 1
    ev = events("repair")[-1]
    assert ev["ok"] is True and ev["commit"] == head1
    # the agent really was called with the repair args and the filled prompt
    call = fake_claude_calls()[0]
    assert "--safe-mode" in call["argv"]
    assert "AssertionError@clipper/calc.py:add" in call["prompt"] and "log tail text" in call["prompt"]
    assert "config/locked.yaml" in call["prompt"] and "{{" not in call["prompt"]
    assert Path(call["cwd"]).resolve() == repo.resolve()


def test_protected_file_edit_is_reverted(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, FIX + """
    pathlib.Path("config/locked.yaml").write_text("test_mode: false\\n")
    """)
    head0 = git(repo, "rev-parse", "HEAD")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert git(repo, "rev-parse", "HEAD") == head0
    assert (repo / "config" / "locked.yaml").read_text() == "test_mode: true\n"
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()  # the good part of the fix is gone too
    assert git(repo, "status", "--porcelain") == ""
    ev = events("repair")[-1]
    assert ev["ok"] is False and "protected" in ev["reason"] and "config/locked.yaml" in ev["reason"]
    assert state.publish_hold()  # stays paused after a failed repair


def test_supervisor_dir_edit_and_new_file_reverted(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, FIX + """
    pathlib.Path("clipper/supervisor/supervisor.py").write_text("# hacked\\n")
    pathlib.Path("clipper/supervisor/extra.py").write_text("# new\\n")
    """)
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert (repo / "clipper" / "supervisor" / "supervisor.py").read_text() == "# supervisor\n"
    assert not (repo / "clipper" / "supervisor" / "extra.py").exists()


def test_gitignored_secret_changes_are_reverted(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, FIX + """
    pathlib.Path("secrets/token.json").write_text("STOLEN")
    pathlib.Path("secrets/new.json").write_text("x")
    pathlib.Path("PAUSE").write_text("")
    """)
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert (repo / "secrets" / "token.json").read_text() == "TOKEN"
    assert not (repo / "secrets" / "new.json").exists()
    assert not (repo / "PAUSE").exists()
    assert "secrets/token.json" in events("repair")[-1]["reason"]


def test_fix_that_fails_tests_is_reverted(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, """
    import pathlib
    p = pathlib.Path("clipper/calc.py")
    p.write_text(p.read_text().replace("a - b", "a * b"))
    pathlib.Path("clipper/junk.py").write_text("x = 1\\n")
    pathlib.Path("clipper/pkg").mkdir()
    pathlib.Path("clipper/pkg/mod.py").write_text("y = 2\\n")
    pathlib.Path("data/agent_note.txt").write_text("keep me")
    """)
    (repo / "data" / "chunk.ts").write_text("precious")
    head0 = git(repo, "rev-parse", "HEAD")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert git(repo, "rev-parse", "HEAD") == head0
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()
    assert not (repo / "clipper" / "junk.py").exists()
    assert not (repo / "clipper" / "pkg").exists()
    assert (repo / "data" / "chunk.ts").read_text() == "precious"   # never cleaned
    assert (repo / "secrets" / "token.json").read_text() == "TOKEN"
    assert "tests failed" in events("repair")[-1]["reason"]


def test_failed_repair_preserves_users_uncommitted_edits(cfg, state, repo, tmp_path, monkeypatch):
    (repo / "README.txt").write_text("user edit\n")
    (repo / "clipper" / "mine.py").write_text("# user's untracked file\n")
    set_script(tmp_path, monkeypatch, """
    import pathlib
    p = pathlib.Path("clipper/calc.py")
    p.write_text(p.read_text().replace("a - b", "a * b"))
    """)
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert (repo / "README.txt").read_text() == "user edit\n"
    assert (repo / "clipper" / "mine.py").exists()
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()


def test_successful_repair_does_not_commit_users_other_edits(cfg, state, repo, tmp_path, monkeypatch):
    (repo / "README.txt").write_text("user edit\n")
    set_script(tmp_path, monkeypatch, FIX)
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is True
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == ["clipper/calc.py"]
    assert (repo / "README.txt").read_text() == "user edit\n"
    assert "README.txt" in git(repo, "status", "--porcelain")


def test_agent_commit_is_rejected(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, FIX + """
    import subprocess
    subprocess.run(["git", "add", "-A"], check=True)
    subprocess.run(["git", "commit", "-q", "-m", "sneaky"], check=True)
    """)
    head0 = git(repo, "rev-parse", "HEAD")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert git(repo, "rev-parse", "HEAD") == head0
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()


def test_no_changes_is_a_failure(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, "pass\n")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert "no changes" in events("repair")[-1]["reason"]


def test_integration_test_runs_when_sample_chunk_exists(cfg, state, repo, tmp_path, monkeypatch):
    (repo / "tests" / "integration").mkdir()
    (repo / "tests" / "integration" / "test_it.py").write_text("def test_it():\n    assert False\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "integration")
    set_script(tmp_path, monkeypatch, FIX)
    # no sample chunk: integration test is skipped, unit run (-x) hits the failing integration test too,
    # so exclude it from the unit command
    agent = make_agent(cfg, state, repo, test_cmd=[str(paths.VENV_PY), "-m", "pytest", "-x", "-q",
                                                   "-p", "no:cacheprovider", "tests/test_calc.py"])
    paths.DATA.samples.mkdir(parents=True, exist_ok=True)
    (paths.DATA.samples / "sample_chunk.ts").write_bytes(b"x")
    assert agent.attempt(REPORT, "") is False
    assert "integration test failed" in events("repair")[-1]["reason"]
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()
    # with the sample missing the integration step is skipped and the fix is kept
    (paths.DATA.samples / "sample_chunk.ts").unlink()
    assert agent.attempt(REPORT, "") is True


def test_attempt_cap(cfg, state, repo, tmp_path, monkeypatch, fake_claude_calls):
    set_script(tmp_path, monkeypatch, FIX)
    state.kv_set("repair.attempts.test-stream", 3)
    agent = make_agent(cfg, state, repo)
    assert agent.exhausted()
    assert agent.attempt(REPORT, "") is False
    assert fake_claude_calls() == []                       # claude never launched
    assert state.publish_hold() and state.kv_get("hold.publish_reason") == "repair"
    assert events("attention")
    assert "Repair agent used all 3 attempts" in (paths.DATA.logs / "ATTENTION.txt").read_text()
    n = len(events("attention"))
    assert agent.attempt(REPORT, "") is False
    assert len(events("attention")) == n                   # attention written once per stream
    assert "a - b" in (repo / "clipper" / "calc.py").read_text()


def test_attempts_counted_per_stream(cfg, state, repo, tmp_path, monkeypatch):
    set_script(tmp_path, monkeypatch, "pass\n")
    agent = make_agent(cfg, state, repo)
    for _ in range(3):
        assert agent.attempt(REPORT, "") is False
    assert state.kv_get("repair.attempts.test-stream") == 3 and agent.exhausted()
    state.set_current_stream("another-stream")
    assert not agent.exhausted()


def test_not_a_repo_root_fails_safely(cfg, state, tmp_path, monkeypatch):
    d = tmp_path / "plain"
    d.mkdir()
    set_script(tmp_path, monkeypatch, FIX)
    agent = make_agent(cfg, state, d)
    assert agent.attempt(REPORT, "") is False
    assert "git" in events("repair")[-1]["reason"] or "work tree" in events("repair")[-1]["reason"]


def test_injectable_runner_used_for_git_and_tests(cfg, state, tmp_path, monkeypatch):
    calls = []

    def runner(cmd, cwd, timeout):
        calls.append(list(cmd))
        if "rev-parse" in cmd and "--show-toplevel" in cmd:
            return 0, str(tmp_path) + "\n"
        if "rev-parse" in cmd:
            return 0, "abc123\n"
        return 0, ""

    set_script(tmp_path, monkeypatch, "pass\n")
    agent = RepairAgent(cfg, state, repo_root=tmp_path, runner=runner)
    assert agent.attempt(REPORT, "") is False       # fake runner reports no changes
    assert calls and all(c[0] == "git" for c in calls)
    assert any("stash" in c for c in calls)


def test_claude_unavailable_reverts(cfg, state, repo, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_CLAUDE_MODE", "crash")
    agent = make_agent(cfg, state, repo)
    assert agent.attempt(REPORT, "") is False
    assert git(repo, "status", "--porcelain") == ""
