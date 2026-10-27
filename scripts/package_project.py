"""Архив воспроизводимого проекта и результатов без данных на уровне клиентов."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]


def main():
    destination=ROOT.parent/"promise-delivery-lab.zip"
    paths=[]
    for path in sorted(ROOT.rglob("*")):
        relative=path.relative_to(ROOT)
        if not path.is_file() or set(relative.parts)&{".git",".venv","__pycache__",".pytest_cache","_site"}:
            continue
        if relative.parts[:2] in [("data","raw"),("data","processed")] or path.name.startswith(".env"):
            continue
        paths.append(path)
    with ZipFile(destination,"w",ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(path,Path(ROOT.name)/path.relative_to(ROOT))
    with ZipFile(destination) as archive:
        assert archive.testzip() is None
        assert not any("/data/raw/" in n or "/data/processed/" in n for n in archive.namelist())
    print(json.dumps({"file":destination.name,"files":len(paths),"bytes":destination.stat().st_size,
        "sha256":hashlib.sha256(destination.read_bytes()).hexdigest()},ensure_ascii=False))


if __name__=="__main__":
    main()
