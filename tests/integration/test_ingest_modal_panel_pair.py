import hashlib
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"scripts"))
import ingest_modal_panel_pair as p


def test_copy_once_is_exact_and_idempotent(tmp_path):
    source=tmp_path/"source"; source.write_bytes(b"known replay bytes")
    destination=tmp_path/"output/game.slp"
    row={"bytes":source.stat().st_size,"sha256":hashlib.sha256(source.read_bytes()).hexdigest()}
    p.copy_once(source,destination,row)
    stamp=destination.stat().st_mtime_ns
    p.copy_once(source,destination,row)
    assert destination.read_bytes()==source.read_bytes()
    assert destination.stat().st_mtime_ns==stamp


@pytest.mark.parametrize("change",["source_hash","destination_bytes","destination_symlink","source_symlink","parent_symlink"])
def test_copy_refuses_changed_or_aliased_paths(tmp_path,change):
    source=tmp_path/"source"; source.write_bytes(b"abc")
    destination=tmp_path/"out/game.slp"; destination.parent.mkdir()
    row={"bytes":3,"sha256":hashlib.sha256(b"abc").hexdigest()}
    if change=="source_hash": row["sha256"]="0"*64
    elif change=="destination_bytes": destination.write_bytes(b"xyz")
    elif change=="destination_symlink": destination.symlink_to(source)
    elif change=="source_symlink":
        alias=tmp_path/"alias"; alias.symlink_to(source); source=alias
    else:
        directory=tmp_path/"aliased"; directory.symlink_to(destination.parent,target_is_directory=True)
        destination=directory/"game.slp"
    with pytest.raises(ValueError): p.copy_once(source,destination,row)
