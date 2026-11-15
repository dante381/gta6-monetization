"""Autostart (schtasks command construction, injectable runner) and nosleep. Nothing is registered
for real and power settings are never touched."""
from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from clipper import paths
from clipper.supervisor import autostart as au
from clipper.supervisor import nosleep

URL = "https://www.youtube.com/@x/live"


class Runner:
    def __init__(self, *results):
        self.calls = []
        self.results = list(results) or [(0, "SUCCESS", "")]

    def __call__(self, cmd):
        self.calls.append(list(cmd))
        return self.results.pop(0) if len(self.results) > 1 else self.results[0]


@pytest.fixture(autouse=True)
def fake_pythonw(monkeypatch, tmp_path):
    monkeypatch.setattr(au, "pythonw", lambda: paths.VENV / "Scripts" / "pythonw.exe")
    monkeypatch.setattr(au.paths, "TMP", tmp_path / "tmp", raising=False)
    fake = tmp_path / "pythonw.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(au, "pythonw", lambda: fake)


def test_simple_command_shape():
    cmd = au.create_simple_cmd(URL)
    assert cmd[:3] == ["schtasks", "/Create", "/F"]
    assert cmd[cmd.index("/SC") + 1] == "ONLOGON"
    assert cmd[cmd.index("/TN") + 1] == "GTA6Clipper"
    tr = cmd[cmd.index("/TR") + 1]
    assert "pythonw.exe" in tr and URL in tr
    assert "supervisor" in tr


def test_xml_task_definition():
    root = ET.fromstring(au.task_xml(URL, user="PC\\me").split("?>", 1)[1])
    ns = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}
    ex = root.find("t:Actions/t:Exec", ns)
    assert ex.find("t:Command", ns).text.endswith("pythonw.exe")
    assert ex.find("t:Arguments", ns).text == f"-m clipper supervise --url {URL}"
    assert ex.find("t:WorkingDirectory", ns).text == str(paths.ROOT)
    assert root.find("t:Triggers/t:LogonTrigger", ns) is not None
    assert root.find("t:Settings/t:ExecutionTimeLimit", ns).text == "PT0S"
    assert root.find("t:Principals/t:Principal/t:RunLevel", ns).text == "LeastPrivilege"


def test_register_uses_xml_first():
    r = Runner((0, "SUCCESS", ""))
    res = au.register(URL, runner=r)
    assert res["method"] == "xml"
    assert len(r.calls) == 1 and r.calls[0][:3] == ["schtasks", "/Create", "/F"] and "/XML" in r.calls[0]


def test_register_falls_back_to_onlogon_then_reports_error():
    r = Runner((1, "", "ERROR: bad xml"), (0, "SUCCESS", ""))
    res = au.register(URL, runner=r)
    assert res["method"] == "onlogon" and "/SC" in r.calls[1] and "ONLOGON" in r.calls[1]
    r = Runner((1, "", "ERROR: Access is denied."), (1, "", "ERROR: Access is denied."))
    with pytest.raises(au.AutostartError) as e:
        au.register(URL, runner=r)
    assert "denied" in str(e.value).lower() and "Administrator" in str(e.value)


def test_unregister_and_status():
    r = Runner((0, "SUCCESS", ""))
    assert au.unregister(runner=r) is True
    assert r.calls[0] == ["schtasks", "/Delete", "/F", "/TN", "GTA6Clipper"]
    assert au.unregister(runner=Runner((1, "", "ERROR: The system cannot find the file specified."))) is False
    with pytest.raises(au.AutostartError):
        au.unregister(runner=Runner((1, "", "ERROR: Access is denied.")))
    out = "TaskName:      \\GTA6Clipper\nStatus:        Ready\nNext Run Time: N/A\n"
    st = au.status(runner=Runner((0, out, "")))
    assert st["registered"] and st["Status"] == "Ready"
    assert au.status(runner=Runner((1, "", "ERROR: not found")))["registered"] is False


def test_main_dispatch(monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(au, "register", lambda url=None, runner=None: seen.append(("reg", url)) or {"method": "xml"})
    monkeypatch.setattr(au, "unregister", lambda runner=None: seen.append("unreg") or True)
    monkeypatch.setattr(au, "status", lambda runner=None: {"registered": False})
    assert au.main(["register", "--url", URL]) == 0
    assert au.main(["unregister"]) == 0
    assert au.main(["status"]) == 0
    assert seen == [("reg", URL), "unreg"]
    monkeypatch.setattr(au, "register", lambda url=None, runner=None: (_ for _ in ()).throw(au.AutostartError("nope")))
    assert au.main(["register", "--url", URL]) == 1
    assert "nope" in capsys.readouterr().out


def test_nosleep_flags(monkeypatch):
    calls = []
    monkeypatch.setattr(nosleep, "set_state", lambda f: calls.append(f) or 1)
    assert nosleep.keep_awake() and nosleep.allow_sleep()
    assert calls == [nosleep.ES_CONTINUOUS | nosleep.ES_SYSTEM_REQUIRED, nosleep.ES_CONTINUOUS]
    assert nosleep.ES_CONTINUOUS == 0x80000000 and nosleep.ES_SYSTEM_REQUIRED == 1
