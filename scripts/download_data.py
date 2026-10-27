"""Загрузка восьми таблиц Olist версии 2 с проверкой SHA-256."""
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import hashlib
import io
import json

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT/"data/manifest.json").read_text(encoding="utf-8-sig"))
    folder = ROOT/"data/raw"
    folder.mkdir(parents=True, exist_ok=True)
    missing = [f for f in manifest["files"] if not (folder/f["file"]).exists()]
    if missing:
        print("Загрузка официального архива Olist версии 2...", flush=True)
        request = Request(manifest["url"], headers={"User-Agent": "promise-delivery-lab/1.0"})
        with urlopen(request, timeout=180) as response:
            payload = response.read()
        with ZipFile(io.BytesIO(payload)) as archive:
            for entry in missing:
                name = entry["file"]
                assert Path(name).name == name
                content = archive.read(name)
                if hashlib.sha256(content).hexdigest() != entry["sha256"]:
                    raise ValueError(f"SHA-256 не совпал: {name}")
                (folder/name).write_bytes(content)
    for entry in manifest["files"]:
        if hashlib.sha256((folder/entry["file"]).read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Изменился исходный файл: {entry['file']}")
    print(f"Проверены {len(manifest['files'])} таблиц Olist.", flush=True)


if __name__ == "__main__":
    main()
