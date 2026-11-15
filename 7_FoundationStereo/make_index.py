#!/usr/bin/env python3
"""Build output/index.html: one page showing every result in output/, newest comparison first.

    python make_index.py            # writes output/index.html
    python make_index.py --open     # ...and opens it in the Windows browser (WSL)

For each folder in output/ it shows the summary table (summary.md / methods.md), the plots (click to open full size),
and links to the 3D viewers (clouds.html, cloud_denoise.html). Images and viewers are linked, not copied, so the page
is small and always up to date; re-run after new results. Open it from disk (it uses relative links).
"""

import argparse
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "output"

ORDER = [  # (folder-name pattern, section title)
    (r"_methods$", "All methods on one frame (compare_methods.py)"),
    (r"_frame\d+_", "One method vs WASS (compare_frame.py)"),
    (r"^demo_", "Upstream demos"),
]
IMAGES = ["methods.png", "comparison.png", "summary.png", "rectified.jpg", "disp_vis.png", "vis.png"]
VIEWERS = ["clouds.html", "cloud_denoise.html"]
DOCS = ["methods.md", "summary.md"]


def md_to_html(text):
    """Just enough Markdown for the summaries: headings, tables, paragraphs, **bold**, `code`."""
    def inline(s):
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
        s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
        return re.sub(r"`(.+?)`", r"<code>\1</code>", s)

    out, table = [], []
    def flush():
        if table:
            rows = [r for r in table if not re.fullmatch(r"\|[\s:|-]+\|", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            h = "".join(f"<th>{inline(c)}</th>" for c in cells[0])
            b = "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in cells[1:])
            out.append(f"<div class=tw><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>")
            table.clear()
    for line in text.splitlines():
        if line.startswith("|"):
            table.append(line)
            continue
        flush()
        if line.startswith("#"):
            lvl = min(len(line) - len(line.lstrip("#")) + 2, 5)
            out.append(f"<h{lvl}>{inline(line.lstrip('#').strip())}</h{lvl}>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush()
    return "\n".join(out)


def section_of(name):
    for i, (pat, _) in enumerate(ORDER):
        if re.search(pat, name):
            return i
    return len(ORDER)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", action="store_true", help="open in the Windows default browser")
    args = ap.parse_args()

    dirs = sorted((d for d in OUT.iterdir() if d.is_dir()), key=lambda d: (section_of(d.name), d.name))
    body, toc, current = [], [], None
    for d in dirs:
        imgs = [f for f in IMAGES if (d / f).exists()]
        views = [f for f in VIEWERS if (d / f).exists()]
        docs = [f for f in DOCS if (d / f).exists()]
        if not (imgs or views or docs):
            continue
        sec = section_of(d.name)
        if sec != current:
            current = sec
            title = ORDER[sec][1] if sec < len(ORDER) else "Other"
            body.append(f"<h2>{html.escape(title)}</h2>")
        anchor = re.sub(r"[^a-z0-9]+", "-", d.name.lower())
        toc.append(f'<a href="#{anchor}">{html.escape(d.name)}</a>')
        links = " ".join(f'<a class=btn href="{d.name}/{v}" target=_blank>3D viewer: {v}</a>' for v in views)
        links += " ".join(f'<a class=lnk href="{d.name}/{f}" target=_blank>{f}</a>' for f in docs)
        body.append(f'<section id="{anchor}"><h3>{html.escape(d.name)}</h3><div class=links>{links}</div>')
        for f in docs[:1]:
            body.append(f"<div class=doc>{md_to_html((d / f).read_text())}</div>")
        for f in imgs:
            body.append(f'<figure><a href="{d.name}/{f}" target=_blank><img loading=lazy src="{d.name}/{f}" '
                        f'alt="{html.escape(f)}"></a><figcaption>{f}</figcaption></figure>')
        body.append("</section>")

    page = f"""<!doctype html><html><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1"><title>FoundationStereo results</title>
<style>
:root{{--bg:#fbfbfa;--fg:#1b1b1a;--mut:#666;--line:#e3e2de;--card:#fff;--acc:#2a78d6}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151515;--fg:#e8e8e6;--mut:#9a9a96;--line:#333;--card:#1e1e1e;--acc:#6aa8f0}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif}}
main{{max-width:1500px;margin:auto;padding:16px}}
nav{{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:13px;margin:8px 0 20px}}
a{{color:var(--acc)}} h2{{margin-top:40px;border-bottom:2px solid var(--line);padding-bottom:4px}}
section{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:16px 0}}
h3{{margin:4px 0 8px;font-family:ui-monospace,monospace;font-size:15px}}
.links{{display:flex;flex-wrap:wrap;gap:8px;margin-bottom:8px}}
.btn{{background:var(--acc);color:#fff;text-decoration:none;padding:4px 10px;border-radius:5px;font-size:13px}}
.lnk{{font-size:13px;padding:4px 0}} .doc{{font-size:13px}} .doc p{{margin:4px 0;color:var(--mut)}}
.tw{{overflow-x:auto}} table{{border-collapse:collapse;margin:6px 0}}
th,td{{border-bottom:1px solid var(--line);padding:3px 8px;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} code{{font-size:12px}}
figure{{margin:10px 0}} img{{max-width:100%;border:1px solid var(--line);border-radius:4px;background:#fff}}
figcaption{{font-size:12px;color:var(--mut)}}
</style></head><body><main>
<h1>FoundationStereo vs WASS: results</h1>
<p style="color:var(--mut)">Generated by <code>make_index.py</code> from <code>7_FoundationStereo/output/</code>.
Click a plot to open it full size; the 3D viewers open in a new tab (drag to orbit, scroll to zoom).</p>
<nav>{''.join(toc)}</nav>
{''.join(body)}
</main></body></html>"""
    (OUT / "index.html").write_text(page)
    print(f"wrote {OUT / 'index.html'} ({len(toc)} result folders)")
    if args.open:
        from cloud_to_html import open_in_windows
        open_in_windows(OUT / "index.html")


if __name__ == "__main__":
    main()
