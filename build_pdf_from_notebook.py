#!/usr/bin/env python3
"""
Build a management-friendly PDF directly from main.ipynb outputs.
No CSV read is performed in this script.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import textwrap
from datetime import date
from pathlib import Path
from typing import Any

# Ensure plotting works in headless environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch


def load_notebook(notebook_path: Path) -> dict[str, Any]:
    return json.loads(notebook_path.read_text())


def first_code_comment(source_text: str) -> str | None:
    for line in source_text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            title = s.lstrip("#").strip()
            if title:
                return title
    return None


CELL_TITLE_OVERRIDES: dict[int, str] = {
    25: "20) Countries with Shortest Re-order Cycle",
}


CELL_EXPLANATIONS: dict[int, tuple[str, str]] = {
    3: (
        "Annual volume trend by financial year using Total Qty.",
        "Compare each year bar to explain growth/decline and overall demand direction.",
    ),
    4: (
        "Quarter-level seasonality pattern for total shipped units.",
        "Highlight recurring peak quarters and weak quarters for inventory/dispatch planning.",
    ),
    5: (
        "Strength-wise volume distribution across years (125/250/500/1000/2000 mg).",
        "Explain where demand is concentrated and whether strength preference is shifting.",
    ),
    6: (
        "Geographic concentration view for top countries over time.",
        "Use line trends + heatmap intensity to show concentration risk and country momentum.",
    ),
    7: (
        "Department leaderboard by shipped volume.",
        "Explain top contributors and discuss resource allocation around high-volume teams.",
    ),
    8: (
        "Overall market channel split: Tender vs Private vs Other.",
        "Use percentage shares to explain channel dependence and diversification scope.",
    ),
    9: (
        "Volume distribution by pharmacopoeia standard and year.",
        "Show compliance mix and where quality/regulatory load is highest.",
    ),
    13: (
        "Country split for I.B.(VPG-GERMANY) strategic/project volume.",
        "Use this to explain where that department's portfolio is concentrated geographically.",
    ),
    17: (
        "India channel transition trend by year.",
        "Discuss whether India is moving between Tender and Private and the commercial implication.",
    ),
    19: (
        "Department efficiency comparison (units delivered per kg API).",
        "Higher values imply better API conversion efficiency; compare leaders vs laggards.",
    ),
    24: (
        "Countries ranked by longest average re-order cycle (days between orders).",
        "Higher bars mean slower repeat demand; use this for safety-stock and replenishment strategy.",
    ),
    25: (
        "Counter-view of re-order behavior: countries with shortest cycle.",
        "Lower cycle means faster repeat demand; these markets need tighter service and stock discipline.",
    ),
    26: (
        "Average order volume by pharmacopoeia as a logistics-cost proxy.",
        "Explain which standards are associated with larger batches and planning impact.",
    ),
    27: (
        "Drill-down of the 'Other' market bucket by department/country.",
        "Use it to identify if 'Other' is meaningful business or just fragmented tail activity.",
    ),
    28: (
        "Stacked quarter-wise operational load by department.",
        "Explain which teams carry workload each quarter and where balancing is needed.",
    ),
    32: (
        "Country-level contribution to latest YoY change (decliners vs growers).",
        "Read negative bars as drag and positive bars as support for latest-year performance.",
    ),
    34: (
        "Concentration risk trend using Top-3 share and HHI.",
        "Higher values indicate greater dependence on fewer countries; use this as risk signal.",
    ),
    36: (
        "Monthly heatmap of volume by financial year.",
        "Darker cells indicate heavier demand months; use this for production calendar planning.",
    ),
    38: (
        "Market-mix share trend across years.",
        "Area movement shows channel model shift over time (Tender/Private/Other).",
    ),
    40: (
        "New-market stickiness: repeat order rate after first entry.",
        "Explain market quality: entries that repeat are scalable, non-repeat entries need review.",
    ),
    42: (
        "Country demand stability segmentation based on re-order pattern.",
        "Use segments to separate predictable markets from volatile ones for planning discipline.",
    ),
    44: (
        "Year-end dependency: share of annual volume shipped on 31-Mar.",
        "Higher share means last-day concentration risk; lower share means smoother operations.",
    ),
    46: (
        "Monthly API utilization ratio vs expected benchmark.",
        "Values near benchmark indicate process stability; sustained deviation suggests control issue.",
    ),
    48: (
        "Top 1% transaction outlier composition by department and market.",
        "Shows where extreme order concentration exists; useful for risk and capacity planning.",
    ),
    50: (
        "Country opportunity matrix: latest scale vs 2-year growth segmentation.",
        "Use quadrants to prioritize defend-core, defend-grow, build, or monitor actions.",
    ),
    55: (
        "Year-quarter comparison views (heatmap + quarter bars).",
        "Chart 1 shows intensity by year/quarter; Chart 2 compares quarter totals across years.",
    ),
    57: (
        "Quarter-level YoY % change for latest year.",
        "Positive bars show improvement; negative bars flag quarter-specific correction areas.",
    ),
    59: (
        "Department trend across years with growth/decline view.",
        "Use this to explain structural winners and pressure points by department.",
    ),
    61: (
        "Latest-year quarter split by department (stacked).",
        "Shows which team dominates each quarter and potential concentration in execution.",
    ),
    63: (
        "Strength share evolution across years.",
        "Highlights portfolio shift between strengths and implications for manufacturing mix.",
    ),
    65: (
        "Department vs strength concentration (latest year heatmap).",
        "Identify specialization: which teams are tied to specific strengths.",
    ),
    67: (
        "SKU trend comparison across years.",
        "Explain top SKU growth/decline and discuss portfolio resilience.",
    ),
    69: (
        "Quarter pattern for top SKUs in latest year.",
        "Shows whether leading SKUs are steady or quarter-spike dependent.",
    ),
}


def nearest_markdown_context(cells: list[dict[str, Any]], idx: int) -> tuple[str | None, str | None]:
    for j in range(idx - 1, -1, -1):
        cell = cells[j]
        if cell.get("cell_type") != "markdown":
            continue
        text = "".join(cell.get("source", [])).strip()
        if not text:
            continue
        lines = text.splitlines()
        heading = None
        body_lines: list[str] = []
        for line in lines:
            if heading is None and line.strip().startswith("#"):
                heading = re.sub(r"^#+\s*", "", line.strip())
                continue
            if line.strip():
                body_lines.append(line.strip())
        body = " ".join(body_lines).strip() if body_lines else None
        if heading or body:
            return heading, body
    return None, None


def clean_stream_text(text: str) -> str:
    cleaned_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "warning" in line.lower():
            continue
        if line.startswith("/var/folders/"):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def choose_title(md_heading: str | None, code_heading: str | None, idx: int, image_number: int, image_count: int) -> str:
    base = CELL_TITLE_OVERRIDES.get(idx) or md_heading or code_heading or f"Notebook Chart (Cell {idx})"
    if image_count > 1:
        return f"{base} - Chart {image_number}/{image_count}"
    return base


def explanation_for(cell_idx: int, title: str, md_body: str | None) -> tuple[str, str]:
    if cell_idx in CELL_EXPLANATIONS:
        return CELL_EXPLANATIONS[cell_idx]

    t = title.lower()
    if "new markets" in t or "sticking" in t:
        return (
            "This evaluates whether newly entered markets place follow-up orders.",
            "Higher repeat rates indicate stronger market quality and scalability.",
        )
    if "year-over-year" in t or "yoy" in t:
        return (
            "This compares annual shipped quantity across financial years.",
            "Read left to right: higher bars/points mean higher yearly volume.",
        )
    if "quarter" in t and "yoy" not in t:
        return (
            "This shows quarter-level demand pattern and seasonality.",
            "Compare Q1-Q4 to identify strong and weak periods for planning.",
        )
    if "strength" in t and "mix" in t:
        return (
            "This shows how product strengths (mg) contribute to total demand.",
            "Look for shifts in share over years to adjust product planning.",
        )
    if "geographic" in t or "country" in t:
        return (
            "This shows country concentration and movement in demand.",
            "Large gaps between top countries indicate concentration risk.",
        )
    if "department" in t:
        return (
            "This compares department-level contribution and change.",
            "Use rank and trend direction to identify leaders and laggards.",
        )
    if "market" in t:
        return (
            "This shows channel mix across Tender, Private, and Other.",
            "A rising channel share means strategy is shifting toward that channel.",
        )
    if "pharmacopoeia" in t or "compliance" in t:
        return (
            "This shows regulatory standard mix and its volume impact.",
            "Use it to align quality/compliance planning with actual demand mix.",
        )
    if "re-order" in t or "reorder" in t or "lead time" in t:
        return (
            "This tracks order recurrence and market stability.",
            "Higher variability means less predictable dispatch and inventory flow.",
        )
    if "api" in t:
        return (
            "This tracks API use efficiency against expected conversion.",
            "Values around the benchmark indicate process stability.",
        )
    if "outlier" in t or "largest transactions" in t:
        return (
            "This isolates very large transactions (top percentile).",
            "It highlights which departments/channels dominate extreme orders.",
        )
    if "opportunity matrix" in t:
        return (
            "This classifies countries by current scale and recent growth.",
            "Use quadrants to decide where to defend, grow, or monitor.",
        )
    if md_body:
        return (
            md_body,
            "Read axes first, then compare category heights/trends to decide action.",
        )
    return (
            "This chart is part of the notebook analysis output.",
            "Read title and axes first, then compare category size or trend direction.",
        )


def extract_sections(nb: dict[str, Any]) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    cells = nb.get("cells", [])

    for idx, cell in enumerate(cells):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        image_outputs = []
        stream_texts = []

        for out in outputs:
            if out.get("output_type") == "stream":
                stream = "".join(out.get("text", ""))
                if stream:
                    stream_texts.append(clean_stream_text(stream))

            data = out.get("data", {})
            if isinstance(data, dict) and "image/png" in data:
                image_b64 = data["image/png"]
                if isinstance(image_b64, list):
                    image_b64 = "".join(image_b64)
                image_outputs.append(image_b64)

        if not image_outputs:
            continue

        md_heading, md_body = nearest_markdown_context(cells, idx)
        code_heading = first_code_comment("".join(cell.get("source", [])))
        stream_text = "\n".join([s for s in stream_texts if s]).strip()

        for image_idx, image_b64 in enumerate(image_outputs, start=1):
            title = choose_title(
                md_heading=md_heading,
                code_heading=code_heading,
                idx=idx,
                image_number=image_idx,
                image_count=len(image_outputs),
            )
            what, how = explanation_for(idx, title, md_body)

            sections.append(
                {
                    "cell_index": idx,
                    "title": title,
                    "md_body": md_body,
                    "what": what,
                    "how": how,
                    "stream_text": stream_text,
                    "image_b64": image_b64,
                }
            )

    return sections


def page_number(fig: plt.Figure, current: int, total: int) -> None:
    fig.text(0.98, 0.015, f"Page {current}/{total}", ha="right", va="bottom", fontsize=8, color="#64748b")


def wrap_and_limit(text: str, width: int, max_lines: int) -> str:
    lines: list[str] = []
    for paragraph in text.splitlines():
        p = paragraph.strip()
        if not p:
            lines.append("")
            continue
        lines.extend(
            textwrap.wrap(
                p,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    return "\n".join(lines)


def draw_info_box(ax: plt.Axes, title: str, body: str, body_fontsize: float = 9.4) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    card = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#cbd5e1",
        facecolor="white",
        transform=ax.transAxes,
    )
    ax.add_patch(card)
    ax.text(0.03, 0.92, title, fontsize=10.2, weight="bold", color="#0f172a", va="top")
    ax.text(
        0.03,
        0.84,
        body,
        fontsize=body_fontsize,
        color="#334155",
        va="top",
        linespacing=1.45,
        clip_on=True,
    )


def render_cover(pdf: PdfPages, current: int, total: int, notebook_name: str, section_count: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#0b132b")

    fig.text(0.06, 0.86, "Management Explanation Report", fontsize=28, weight="bold", color="white")
    fig.text(0.06, 0.79, "Generated directly from main.ipynb", fontsize=14, color="#cbd5e1")
    fig.text(0.06, 0.72, f"Notebook source: {notebook_name}", fontsize=11, color="#cbd5e1")
    fig.text(0.06, 0.68, f"Charts extracted: {section_count}", fontsize=11, color="#cbd5e1")
    fig.text(0.06, 0.64, f"Generated on: {date.today().isoformat()}", fontsize=11, color="#cbd5e1")

    note = (
        "This version is built from notebook outputs only.\n"
        "It does not re-read the CSV during PDF generation."
    )
    fig.text(0.06, 0.53, note, fontsize=12, color="#e2e8f0", linespacing=1.6)

    fig.text(0.06, 0.40, "Flow", fontsize=14, weight="bold", color="white")
    fig.text(
        0.06,
        0.32,
        "1. Data meaning (business terms)\n2. Graph-by-graph explanation\n3. Example output lines from notebook",
        fontsize=12,
        color="#cbd5e1",
        linespacing=1.6,
    )

    page_number(fig, current, total)
    pdf.savefig(fig)
    plt.close(fig)


def render_data_explanation(pdf: PdfPages, current: int, total: int) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#f8fafc")
    ax = fig.add_axes([0.05, 0.06, 0.9, 0.9])
    ax.axis("off")

    ax.text(0.0, 0.97, "Data First: What Each Field Means", fontsize=20, weight="bold", va="top")

    intro = (
        "Volume in this report means: Total Qty (shipment units) aggregated over a group "
        "(year, quarter, country, department, etc.)."
    )
    ax.text(0.0, 0.88, textwrap.fill(intro, width=120), fontsize=12, color="#0f172a")

    fields = [
        "`Invoice Date` = shipment/invoice date",
        "`Financial Year` = reporting year bucket (FY)",
        "`Quarter` = fiscal quarter (Q1 to Q4)",
        "`Final Country` = destination market/country",
        "`Department` = internal commercial/operational owner",
        "`Strength In MG` = dosage strength (e.g., 500 mg, 1000 mg)",
        "`Market` = channel type (Tender, Private, Other)",
        "`Pharmacopoeia` = regulatory standard (EP/USP/IP)",
        "`Total Qty` = units shipped (this is the core volume measure)",
        "`API Requirement in KG` = API needed for produced/shipped quantity",
        "`Wastage` = process factor used with API/quantity calculations",
    ]

    y = 0.80
    for line in fields:
        ax.text(0.02, y, f"- {line}", fontsize=11, color="#1e293b", va="top")
        y -= 0.055

    guide = (
        "How to explain each graph to management:\n"
        "1) Start with what the graph measures.\n"
        "2) Point to trend or concentration.\n"
        "3) Give one concrete example number.\n"
        "4) End with business action."
    )
    ax.text(0.0, 0.14, guide, fontsize=11.5, color="#0f172a", linespacing=1.6)

    page_number(fig, current, total)
    pdf.savefig(fig)
    plt.close(fig)


def render_chart_page(
    pdf: PdfPages,
    section: dict[str, Any],
    current: int,
    total: int,
) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor("#f8fafc")

    title = section["title"]
    what = section["what"]
    how = section["how"]
    stream_text = section.get("stream_text") or ""
    md_body = section.get("md_body") or ""

    wrapped_title = textwrap.fill(title, width=82)
    title_lines = wrapped_title.count("\n") + 1
    fig.text(0.05, 0.955, wrapped_title, fontsize=14, weight="bold", va="top", color="#0f172a")
    cell_info_y = 0.905 - max(0, title_lines - 1) * 0.034
    fig.text(0.05, cell_info_y, f"Notebook cell: {section['cell_index']}", fontsize=9, color="#64748b")

    narrative_parts = [
        f"What this shows: {what}",
        f"How to read it: {how}",
    ]
    if md_body:
        narrative_parts.append(f"Business context from notebook: {md_body}")

    if stream_text:
        # Keep example concise and fit into a bounded panel.
        short_example = "\n".join(stream_text.splitlines()[:8]).strip()
        if len(short_example) > 900:
            short_example = short_example[:900].rstrip() + "..."
    else:
        short_example = "No textual output for this chart in notebook (visual-only output)."

    narrative = "\n\n".join(narrative_parts)
    narrative = wrap_and_limit(narrative, width=78, max_lines=14)
    short_example = wrap_and_limit(short_example, width=48, max_lines=16)

    ax_left = fig.add_axes([0.05, 0.66, 0.60, 0.21])
    draw_info_box(ax_left, "How To Explain This Graph", narrative, body_fontsize=9.3)

    ax_right = fig.add_axes([0.67, 0.66, 0.28, 0.21])
    draw_info_box(ax_right, "Example From Notebook Output", short_example, body_fontsize=8.9)

    image_bytes = base64.b64decode(section["image_b64"])
    image = mpimg.imread(io.BytesIO(image_bytes), format="png")

    ax_img = fig.add_axes([0.05, 0.07, 0.90, 0.55])
    ax_img.set_facecolor("white")
    ax_img.imshow(image)
    ax_img.axis("off")

    page_number(fig, current, total)
    pdf.savefig(fig)
    plt.close(fig)


def build_pdf_from_notebook(notebook_path: Path, output_path: Path) -> None:
    nb = load_notebook(notebook_path)
    sections = extract_sections(nb)
    total_pages = 2 + len(sections)

    with PdfPages(output_path) as pdf:
        page = 1
        render_cover(
            pdf=pdf,
            current=page,
            total=total_pages,
            notebook_name=notebook_path.name,
            section_count=len(sections),
        )
        page += 1

        render_data_explanation(pdf=pdf, current=page, total=total_pages)
        page += 1

        for section in sections:
            render_chart_page(pdf=pdf, section=section, current=page, total=total_pages)
            page += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Create management PDF directly from notebook outputs.")
    parser.add_argument("--notebook", default="main.ipynb", help="Path to source notebook")
    parser.add_argument(
        "--output",
        default="Meropenem_Management_Report.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    notebook_path = Path(args.notebook)
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook not found: {notebook_path}")

    output_path = Path(args.output)
    build_pdf_from_notebook(notebook_path=notebook_path, output_path=output_path)
    print(f"PDF generated from notebook: {output_path.resolve()}")


if __name__ == "__main__":
    main()
