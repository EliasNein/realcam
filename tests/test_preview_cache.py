"""Preview cache: keys, size limit with least-recently-used removal, cleanup at start."""
import os
import time

from gui.core import previewcache as PC
from gui.core.jobs import AppPaths


def _img(folder, size, name="x.jpg"):
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_bytes(b"j" * size)
    return p


def test_key_is_stable_and_depends_on_every_part():
    base = ("src", 100, 5, 1200, "lite", "showroom", "cfg", 4)
    assert PC.make_key(*base) == PC.make_key(*base)
    assert PC.KEY_RE.match(PC.make_key(*base))
    for i, other in enumerate((9, 9, 9, 9, "ai", "subtle", "cfg2", 5)):
        changed = list(base)
        changed[i] = other if i else "src2"
        assert PC.make_key(*changed) != PC.make_key(*base), i


def test_put_get_and_miss(tmp_path):
    cache = PC.PreviewCache(tmp_path / "previews")
    key = PC.make_key("a")
    assert cache.get(key) is None
    path = cache.put(key, _img(tmp_path / "tmp", 100))
    assert path == cache.image_path(key) and cache.get(key) == path and cache.total_bytes() == 100


def test_limit_removes_least_recently_used(tmp_path):
    cache = PC.PreviewCache(tmp_path / "previews", limit=250)
    keys = [PC.make_key(i) for i in range(3)]
    for n, key in enumerate(keys):
        cache.put(key, _img(tmp_path / f"t{n}", 100))
        os.utime(cache.image_path(key), (1000 + n, 1000 + n))   # keys[0] oldest
    # 3 x 100 > 250: putting the third already removed the oldest
    assert cache.get(keys[0]) is None and cache.get(keys[1]) and cache.get(keys[2])
    cache.get(keys[1])                                          # use refreshes: keys[2] is now the oldest
    os.utime(cache.image_path(keys[2]), (500, 500))
    cache.put(PC.make_key(9), _img(tmp_path / "t9", 100))
    assert cache.get(keys[2]) is None and cache.get(keys[1]) and cache.total_bytes() <= 250


def test_new_entry_is_never_removed_even_if_alone_over_the_limit(tmp_path):
    cache = PC.PreviewCache(tmp_path / "previews", limit=50)
    key = PC.make_key("big")
    cache.put(key, _img(tmp_path / "t", 100))
    assert cache.get(key) is not None


def test_default_limit_is_2_gb():
    assert PC.CACHE_LIMIT_BYTES == 2 * 1024**3


def test_clean_start_removes_incomplete_entries_and_old_work_folders_only(tmp_path):
    cache = PC.PreviewCache(tmp_path / "previews", limit=10**9)
    good = PC.make_key("good")
    cache.put(good, _img(tmp_path / "t", 10))
    broken = cache.root / PC.make_key("broken")
    broken.mkdir()
    foreign = cache.root / "notes"
    foreign.mkdir()
    tmp_root = tmp_path / "tmp"
    old, fresh, other = tmp_root / "pv_old", tmp_root / "pv_fresh", tmp_root / "keep_me"
    for d in (old, fresh, other):
        d.mkdir(parents=True)
    past = time.time() - 2 * PC.TMP_MAX_AGE_S
    os.utime(old, (past, past))
    os.utime(other, (past, past))
    cache.clean_start(tmp_root)
    assert cache.get(good) and not broken.exists() and foreign.exists()
    assert not old.exists() and fresh.exists() and other.exists()


def test_clean_start_enforces_limit_and_handles_missing_folders(tmp_path):
    cache = PC.PreviewCache(tmp_path / "previews", limit=150)
    assert cache.clean_start(tmp_path / "nothing") == 0
    for n in range(3):
        _img(cache.root / PC.make_key(n), 100, PC.IMAGE)
        os.utime(cache.image_path(PC.make_key(n)), (1000 + n, 1000 + n))
    assert cache.clean_start() == 2 and cache.total_bytes() == 100 and cache.get(PC.make_key(2))


def test_app_paths_for_previews(tmp_path):
    p = AppPaths(data=tmp_path)
    assert p.previews == tmp_path / "previews" and p.preview_tmp == tmp_path / "preview_tmp" and p.short_previews == tmp_path / "short_previews"
