import json
import subprocess
from types import SimpleNamespace

import pytest

from gui import constants as C
from gui.core import preflight as PF


def _cuda(**kw):
    data = {"cuda": True, "torch": "2.14.1+cu130", "name": "NVIDIA GeForce RTX 4070", "vram": 12 * 1024**3, "cc": [8, 9]}
    data.update(kw)
    return lambda timeout: subprocess.CompletedProcess([], 0, json.dumps(data) + "\n", "")


def _ids(checks):
    return [(c.id, c.level) for c in checks]


def test_ffmpeg_found_and_missing():
    assert PF.check_ffmpeg(lambda name: f"C:/x/{name}.exe").level == PF.OK

    def boom(name):
        raise RuntimeError("nope")
    c = PF.check_ffmpeg(boom)
    assert c.level == PF.ERROR and "winget" in c.text


def test_weights_complete_missing_and_wrong_size(tmp_path):
    required = (("a.pth", 3), ("b.pkl", 4))
    (tmp_path / "a.pth").write_bytes(b"123")
    assert _ids(PF.check_weights(tmp_path, required)) == [("weights", PF.ERROR)]
    assert "b.pkl" in PF.check_weights(tmp_path, required)[0].text and "README" in PF.check_weights(tmp_path, required)[0].text
    (tmp_path / "b.pkl").write_bytes(b"12")
    wrong = PF.check_weights(tmp_path, required)
    assert wrong[0].level == PF.ERROR and "b.pkl" in wrong[0].text
    (tmp_path / "b.pkl").write_bytes(b"1234")
    assert _ids(PF.check_weights(tmp_path, required)) == [("weights", PF.OK)]


def test_repo_weights_table_matches_readme():
    readme = (C.ROOT / "README.md").read_text(encoding="utf-8")
    for name, size in C.REQUIRED_WEIGHTS:
        assert name in readme and f"{size:,}" in readme


def test_cuda_ok_rtx4070():
    checks, gpu = PF.check_cuda(_cuda())
    assert _ids(checks) == [("cuda", PF.OK)] and gpu.supports_av1
    assert "RTX 4070" in checks[0].text and "12,0" in checks[0].text


def test_cuda_no_gpu():
    checks, gpu = PF.check_cuda(_cuda(cuda=False))
    assert gpu is None and _ids(checks) == [("cuda", PF.ERROR)] and "NVIDIA" in checks[0].text


def test_cuda_gpu_too_old_names_the_card():
    checks, _ = PF.check_cuda(_cuda(name="GeForce GTX 1060", cc=[6, 1]))
    assert _ids(checks) == [("cuda", PF.ERROR)] and "GTX 1060" in checks[0].text


def test_low_vram_warns_and_8gb_card_does_not():
    assert ("vram", PF.WARN) in _ids(PF.check_cuda(_cuda(vram=6 * 1024**3))[0])
    assert ("vram", PF.WARN) not in _ids(PF.check_cuda(_cuda(vram=int(7.99 * 1024**3)))[0])


def test_no_av1_before_rtx40_is_info():
    assert ("av1", PF.INFO) in _ids(PF.check_cuda(_cuda(cc=[8, 6], name="RTX 3080"))[0])
    assert ("av1", PF.INFO) not in _ids(PF.check_cuda(_cuda())[0])


def test_cuda_subprocess_failures():
    def fail(timeout):
        return subprocess.CompletedProcess([], 1, "", "Traceback...\nModuleNotFoundError: No module named 'torch'")
    c, _ = PF.check_cuda(fail)
    assert c[0].level == PF.ERROR and "torch" in c[0].text

    def hang(timeout):
        raise subprocess.TimeoutExpired("x", timeout)
    assert PF.check_cuda(hang)[0][0].level == PF.ERROR

    def garbage(timeout):
        return subprocess.CompletedProcess([], 0, "hello", "")
    assert PF.check_cuda(garbage)[0][0].level == PF.ERROR

    def nopython(timeout):
        raise FileNotFoundError("python")
    assert PF.check_cuda(nopython)[0][0].level == PF.ERROR


def test_driver_checks():
    assert PF.check_driver(lambda: "617.14") is None
    old = PF.check_driver(lambda: "552.44")
    assert old.level == PF.ERROR and "552.44" in old.text and "580" in old.text
    assert PF.check_driver(lambda: None).level == PF.WARN


def test_disk_estimate_and_low_space(tmp_path):
    est = PF.estimate_output_bytes(60)
    assert est == int(60 * 80e6 / 8 * 1.10)
    free = lambda path: SimpleNamespace(free=est // 2)
    out = PF.check_disk(60, tmp_path / "w", tmp_path / "o", usage=free)
    assert out[0].level == PF.ERROR and "GB" in out[0].text
    # both folders on one drive: need is twice the output
    one = lambda path: SimpleNamespace(free=int(est * 1.5))
    assert PF.check_disk(60, tmp_path / "w", tmp_path / "o", usage=one)[0].level == PF.ERROR
    plenty = lambda path: SimpleNamespace(free=est * 3)
    assert PF.check_disk(60, tmp_path / "w", tmp_path / "o", usage=plenty)[0].level == PF.OK


def test_disk_without_video_is_info():
    assert PF.check_disk(None)[0].level == PF.INFO


def test_run_preflight_collects_everything(tmp_path):
    for name, size in C.REQUIRED_WEIGHTS:
        (tmp_path / name).write_bytes(b"\0" * size)
    checks = PF.run_preflight(None, find=lambda n: n, models_dir=tmp_path, cuda_run=_cuda(), driver_query=lambda: "617.14")
    assert not PF.has_error(checks)
    assert {c.id for c in checks} >= {"ffmpeg", "weights", "cuda", "disk"}


@pytest.mark.gpu
def test_real_cuda_probe_in_subprocess():
    checks, gpu = PF.check_cuda()
    assert not PF.has_error(checks), checks
    assert gpu.vram_bytes > 0
