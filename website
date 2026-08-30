from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "ae_results_github_pages"
VIEWER_DIR = OUT_DIR / "viewers"

# Keep the original Plotly viewer and crack calculations, but limit the number
# of stored animation frames so every HTML remains suitable for GitHub Pages.
MAX_GITHUB_FRAMES = 56

EXPERIMENTS = [
    {
        "id": "t0188",
        "folder": ROOT / "t0188.exp",
        "labels": {
            "test_1": "test1_intact",
            "test_2": "test2_90",
            "test_3": "test3_90",
        },
    },
    {
        "id": "t0192",
        "folder": ROOT / "t0192.exp",
        "labels": {
            "test_1_80": "test1_80",
            "test_2_90": "test2_90",
            "test_3_70": "test3_70",
        },
    },
    {
        "id": "t0199",
        "folder": ROOT / "t0199.exp",
        "labels": {
            "test_2_80": "test2_80",
        },
    },
    {
        "id": "t0208",
        "folder": ROOT / "t0208.exp",
        "labels": {
            "test_1_90": "test1_90",
            "test_2_90": "test2_90",
        },
    },
]


def safe_name(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in text)


def load_figure_module(script_path: Path) -> dict:
    """Load the original viewer functions without running its export block."""
    source = script_path.read_text(encoding="utf-8")
    marker = "\nfor results_path in RESULTS_PATHS:"
    if marker not in source:
        raise RuntimeError(f"Cannot find export block in {script_path}")

    library_source = source.split(marker, 1)[0]
    namespace = {
        "__file__": str(script_path),
        "__name__": f"github_viewer_{script_path.parent.name}",
    }
    exec(compile(library_source, str(script_path), "exec"), namespace)
    return namespace


def build_index(cards: list[dict[str, str]]) -> str:
    sections = []
    experiments = sorted({card["experiment"] for card in cards})
    for experiment in experiments:
        experiment_cards = [card for card in cards if card["experiment"] == experiment]
        items = "\n".join(
            f'''        <article class="result" data-search="{card["search"]}">
          <div>
            <h3>{card["name"]}</h3>
            <p>{card["source"]} · HypoDD</p>
          </div>
          <div class="actions">
            <a class="time" href="{card["time_href"]}">Time</a>
            <a class="depth" href="{card["depth_href"]}">Depth</a>
          </div>
        </article>'''
            for card in experiment_cards
        )
        sections.append(
            f'''    <section data-experiment="{experiment}">
      <div class="section-heading"><h2>{experiment}</h2><span>{len(experiment_cards)} 組結果</span></div>
      <div class="results">{items}</div>
    </section>'''
        )

    options = "\n".join(f'<option value="{experiment}">{experiment}</option>' for experiment in experiments)
    return f'''<!doctype html>
<html lang="zh-Hant">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>AE 3D Viewer Results</title>
  <style>
    :root {{ --bg:#f4f6fb; --panel:#fff; --text:#17233c; --muted:#65728a; --line:#dfe5ef; --blue:#1769e0; --green:#16845f; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; min-height:100vh; background:var(--bg); color:var(--text); font-family:"Segoe UI","Microsoft JhengHei","Noto Sans TC",Arial,sans-serif; }}
    main {{ width:min(1160px,calc(100% - 32px)); margin:0 auto; padding:48px 0 72px; }}
    header {{ display:flex; align-items:end; justify-content:space-between; gap:24px; margin-bottom:30px; }}
    h1 {{ margin:0; font-size:36px; letter-spacing:0; }}
    header p {{ margin:8px 0 0; color:var(--muted); }}
    .filters {{ display:flex; gap:10px; }}
    input,select {{ height:40px; min-width:180px; padding:0 12px; border:1px solid var(--line); border-radius:7px; background:#fff; color:var(--text); font:inherit; }}
    section {{ margin-top:28px; }}
    .section-heading {{ display:flex; align-items:baseline; justify-content:space-between; margin-bottom:12px; }}
    h2 {{ margin:0; font-size:20px; }}
    .section-heading span,.result p {{ color:var(--muted); font-size:13px; }}
    .results {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:14px; }}
    .result {{ min-height:132px; padding:18px; display:flex; flex-direction:column; justify-content:space-between; gap:16px; background:var(--panel); border:1px solid var(--line); border-radius:8px; box-shadow:0 7px 18px rgba(24,36,64,.08); }}
    h3 {{ margin:0; font-size:18px; letter-spacing:0; word-break:break-word; }}
    .result p {{ margin:6px 0 0; }}
    .actions {{ display:flex; gap:9px; }}
    .actions a {{ min-width:88px; height:37px; display:inline-flex; align-items:center; justify-content:center; border-radius:6px; color:#fff; font-weight:700; text-decoration:none; }}
    .time {{ background:var(--blue); }} .depth {{ background:var(--green); }}
    .hidden {{ display:none; }}
    .note {{ margin-top:28px; color:var(--muted); font-size:13px; }}
    @media (max-width:850px) {{ header {{ align-items:stretch; flex-direction:column; }} .filters {{ flex-wrap:wrap; }} input,select {{ flex:1; }} .results {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
    @media (max-width:560px) {{ main {{ padding-top:30px; }} h1 {{ font-size:28px; }} .results {{ grid-template-columns:1fr; }} .filters {{ flex-direction:column; }} }}
  </style>
</head>
<body>
  <main>
    <header>
      <div><h1>AE 3D Viewer Results</h1><p>選擇實驗，可以查看 HypoDD 3D viewer。</p></div>
      <div class="filters">
        <input id="search" type="search" placeholder="搜尋 test 編號或實驗名稱">
        <select id="experiment"><option value="all">全部實驗</option>{options}</select>
      </div>
    </header>
{chr(10).join(sections)}
    <p class="note">Time 以事件時間著色；Depth 以 AE 點進入試體內部的深度著色。</p>
  </main>
  <script>
    const search = document.querySelector("#search");
    const experiment = document.querySelector("#experiment");
    const cards = [...document.querySelectorAll(".result")];
    const sections = [...document.querySelectorAll("section[data-experiment]")];
    function filterResults() {{
      const query = search.value.trim().toLowerCase();
      cards.forEach(card => {{
        const section = card.closest("section");
        const showExperiment = experiment.value === "all" || section.dataset.experiment === experiment.value;
        const showSearch = !query || card.dataset.search.toLowerCase().includes(query);
        card.classList.toggle("hidden", !(showExperiment && showSearch));
      }});
      sections.forEach(section => section.classList.toggle("hidden", !section.querySelector(".result:not(.hidden)")));
    }}
    search.addEventListener("input", filterResults);
    experiment.addEventListener("change", filterResults);
  </script>
</body>
</html>'''


def build_viewers() -> list[dict[str, str]]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    VIEWER_DIR.mkdir(parents=True, exist_ok=True)
    cards = []

    for experiment in EXPERIMENTS:
        experiment_id = experiment["id"]
        module = load_figure_module(experiment["folder"] / "concrete_Figure_Output.py")
        module["OUTPUT_DIR"] = VIEWER_DIR
        module["PLOTLY_JS_MODE"] = "cdn"
        module["MAX_TIME_STEPS"] = MAX_GITHUB_FRAMES
        module["MAX_TIME_STEPS_BY_TEST"] = {}
        for results_path in module["RESULTS_PATHS"]:
            if not Path(results_path).exists():
                print(f"Skip missing pkl: {results_path}")
                continue

            data = module["load_results"](results_path)
            module["apply_results_data"](data)

            for result in module["target_results"]():
                source_stem = Path(result.get("file", "")).stem
                label = experiment["labels"].get(source_stem)
                if label is None or not result.get("events"):
                    continue

                base_name = f"{experiment_id}_{label}"
                for color_mode in ("time", "depth"):
                    module["POINT_COLOR_MODE"] = color_mode
                    generated_path = Path(module["write_interactive_html"](result, method="hypodd"))
                    target_path = VIEWER_DIR / f"{safe_name(base_name + '_' + color_mode)}.html"
                    if target_path.exists() and target_path != generated_path:
                        target_path.unlink()
                    if generated_path != target_path:
                        generated_path.replace(target_path)

                cards.append(
                    {
                        "experiment": experiment_id,
                        "name": base_name,
                        "source": source_stem,
                        "search": f"{experiment_id} {label} {source_stem} time depth".lower(),
                        "time_href": f"viewers/{safe_name(base_name + '_time')}.html",
                        "depth_href": f"viewers/{safe_name(base_name + '_depth')}.html",
                    }
                )
    return cards


def main() -> None:
    cards = build_viewers()
    (OUT_DIR / "index.html").write_text(build_index(cards), encoding="utf-8")
    (OUT_DIR / ".nojekyll").touch()
    print(f"Built {len(cards)} original-format result groups in {OUT_DIR}")


if __name__ == "__main__":
    main()
