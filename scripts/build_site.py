"""Публикация только готового HTML, без клиентских данных."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REPO="https://github.com/vladislavsavin543/promise-delivery-lab"


def main():
    source=(ROOT/"reports/dashboard.html").read_text(encoding="utf-8")
    if "__RESULTS__" in source:
        raise ValueError("Сначала выполните python run.py")
    source=source.replace('href="REPORT_RU.md"',f'href="{REPO}/blob/main/reports/REPORT_RU.md"')
    folder=ROOT/"_site"
    folder.mkdir(exist_ok=True)
    (folder/"index.html").write_text(source,encoding="utf-8")
    (folder/".nojekyll").touch()
    print("Демо собрано: _site/index.html")


if __name__=="__main__":
    main()
