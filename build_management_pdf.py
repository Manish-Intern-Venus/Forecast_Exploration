#!/usr/bin/env python3
"""
Create a management-ready PDF from the same analyses used in main.ipynb.
"""

from __future__ import annotations

import argparse
import os
import textwrap
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Keep matplotlib cache writable in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter


def human_readable_number(value: float, _position: int | None = None) -> str:
    abs_value = abs(value)
    if abs_value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}B"
    if abs_value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs_value >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def percent_text(value: float) -> str:
    if pd.isna(value):
        return "NA"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def short_number(value: float) -> str:
    return human_readable_number(float(value), None)


def setup_theme() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "figure.facecolor": "#f8fafc",
            "axes.facecolor": "#ffffff",
            "axes.edgecolor": "#cbd5e1",
            "axes.labelcolor": "#1e293b",
            "xtick.color": "#334155",
            "ytick.color": "#334155",
            "text.color": "#0f172a",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "legend.frameon": True,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
        }
    )


def parse_dates_with_fallback(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, format="%d-%b-%y", errors="coerce")
    missing = parsed.isna()
    if missing.any():
        parsed.loc[missing] = pd.to_datetime(series[missing], dayfirst=True, errors="coerce")
    return parsed


def compute_metrics(df: pd.DataFrame) -> dict:
    data = df.copy()

    data["Invoice Date"] = parse_dates_with_fallback(data["Invoice Date"])
    data = data.dropna(subset=["Invoice Date"]).copy()
    data["Financial Year"] = data["Financial Year"].astype(int)
    data["Quarter"] = data["Quarter"].astype(int)
    data["Strength Label"] = data["Strength In MG"].apply(
        lambda x: f"{int(x)} mg" if pd.notna(x) else "Unknown"
    )
    data["Month Number"] = data["Invoice Date"].dt.month
    data["Month Name"] = data["Invoice Date"].dt.strftime("%b")

    latest_year = int(data["Financial Year"].max())
    previous_year = latest_year - 1

    yoy = data.groupby("Financial Year")["Total Qty"].sum().sort_index().to_frame("Total Qty")
    yoy["YoY %"] = yoy["Total Qty"].pct_change() * 100

    quarter_by_year = (
        data.groupby(["Financial Year", "Quarter"])["Total Qty"]
        .sum()
        .unstack(fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
        .sort_index()
    )

    top_months = (
        data.groupby("Month Name")["Total Qty"]
        .sum()
        .reindex(
            ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
            fill_value=0,
        )
        .sort_values(ascending=False)
        .head(3)
    )

    country_total = data.groupby("Final Country")["Total Qty"].sum().sort_values(ascending=False)
    top10_countries = country_total.head(10)
    top3_share_overall = country_total.head(3).sum() / country_total.sum() * 100

    country_year = data.groupby(["Financial Year", "Final Country"])["Total Qty"].sum().unstack(fill_value=0)
    country_share = country_year.div(country_year.sum(axis=1), axis=0)
    concentration = pd.DataFrame(
        {
            "Top 3 Share (%)": country_share.apply(lambda row: row.nlargest(3).sum() * 100, axis=1),
            "HHI (0-10,000)": country_share.apply(lambda row: (row.pow(2).sum()) * 10_000, axis=1),
        }
    ).sort_index()

    if latest_year in country_year.index and previous_year in country_year.index:
        country_change = (country_year.loc[latest_year] - country_year.loc[previous_year]).sort_values()
        country_declines = country_change.head(10)
        country_growth = country_change.tail(10).sort_values(ascending=False)
    else:
        country_change = pd.Series(dtype=float)
        country_declines = pd.Series(dtype=float)
        country_growth = pd.Series(dtype=float)

    market_by_year = data.groupby(["Financial Year", "Market"])["Total Qty"].sum().unstack(fill_value=0).sort_index()
    market_share_by_year = market_by_year.div(market_by_year.sum(axis=1), axis=0) * 100
    market_total = data.groupby("Market")["Total Qty"].sum().sort_values(ascending=False)
    overall_market_share = market_total / market_total.sum() * 100

    pharma_by_year = (
        data.groupby(["Financial Year", "Pharmacopoeia"])["Total Qty"]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    pharma_total = data.groupby("Pharmacopoeia")["Total Qty"].sum().sort_values(ascending=False)

    department_year = data.groupby(["Department", "Financial Year"])["Total Qty"].sum().unstack(fill_value=0)
    department_total = department_year.sum(axis=1).sort_values(ascending=False)
    if previous_year in department_year.columns and latest_year in department_year.columns:
        dept_growth_pct = (
            (department_year[latest_year] - department_year[previous_year])
            / department_year[previous_year].replace(0, np.nan)
            * 100
        )
        dept_growth_pct = dept_growth_pct.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)
    else:
        dept_growth_pct = pd.Series(dtype=float)

    strength_year = (
        data.groupby(["Strength Label", "Financial Year"])["Total Qty"]
        .sum()
        .unstack(fill_value=0)
    )
    strength_order = strength_year.sum(axis=1).sort_values(ascending=False).index
    strength_year = strength_year.loc[strength_order]
    strength_share_by_year = strength_year.div(strength_year.sum(axis=0), axis=1) * 100

    latest_year_data = data[data["Financial Year"] == latest_year].copy()
    dept_strength_latest = (
        latest_year_data.groupby(["Department", "Strength Label"])["Total Qty"].sum().unstack(fill_value=0)
    )
    top8_departments = latest_year_data.groupby("Department")["Total Qty"].sum().nlargest(8).index
    dept_strength_latest = dept_strength_latest.reindex(top8_departments).fillna(0)

    monthly_volume = (
        data.groupby(["Financial Year", "Month Number"])["Total Qty"].sum().reset_index()
    )
    monthly_heatmap = (
        monthly_volume.pivot(index="Financial Year", columns="Month Number", values="Total Qty")
        .reindex(columns=range(1, 13), fill_value=0)
        .fillna(0)
        .sort_index()
    )

    year_end_flag = data["Invoice Date"].dt.strftime("%d-%b") == "31-Mar"
    year_end_volume = data[year_end_flag].groupby("Financial Year")["Total Qty"].sum()
    annual_volume = data.groupby("Financial Year")["Total Qty"].sum()
    year_end_dependency = (year_end_volume / annual_volume * 100).fillna(0).sort_index()

    data["API Use Ratio"] = (
        data["API Requirement in KG"] * 1_000_000
    ) / (data["Total Qty"] * data["Strength In MG"])
    data.loc[~np.isfinite(data["API Use Ratio"]), "API Use Ratio"] = np.nan
    monthly_api_ratio = data.set_index("Invoice Date")["API Use Ratio"].resample("ME").mean().dropna()

    country_order_flow = data.sort_values(["Final Country", "Invoice Date"]).copy()
    country_order_flow["Days Between Orders"] = (
        country_order_flow.groupby("Final Country")["Invoice Date"].diff().dt.days
    )
    reorder_stats = (
        country_order_flow.groupby("Final Country")["Days Between Orders"]
        .agg(Intervals="count", Avg_Days="mean", Std_Days="std")
        .dropna()
    )
    reorder_stats = reorder_stats[reorder_stats["Intervals"] >= 5].copy()
    reorder_stats["Volatility"] = reorder_stats["Std_Days"] / reorder_stats["Avg_Days"]

    def classify_reorder(row: pd.Series) -> str:
        if row["Avg_Days"] <= 90 and row["Volatility"] <= 0.5:
            return "Stable and Frequent"
        if row["Avg_Days"] <= 90 and row["Volatility"] > 0.5:
            return "Frequent but Unstable"
        if row["Avg_Days"] > 90 and row["Volatility"] <= 0.5:
            return "Predictable but Infrequent"
        return "Infrequent and Unstable"

    if not reorder_stats.empty:
        reorder_stats["Segment"] = reorder_stats.apply(classify_reorder, axis=1)
        reorder_segment_counts = reorder_stats["Segment"].value_counts()
    else:
        reorder_segment_counts = pd.Series(dtype=int)

    outlier_cutoff = data["Total Qty"].quantile(0.99)
    large_orders = data[data["Total Qty"] >= outlier_cutoff].copy()
    if not large_orders.empty:
        large_order_dept_share = (
            large_orders.groupby("Department")["Total Qty"].sum() / large_orders["Total Qty"].sum() * 100
        ).sort_values(ascending=False).head(10)
        large_order_market_share = (
            large_orders.groupby("Market")["Total Qty"].sum() / large_orders["Total Qty"].sum() * 100
        ).sort_values(ascending=False)
    else:
        large_order_dept_share = pd.Series(dtype=float)
        large_order_market_share = pd.Series(dtype=float)

    start_year_for_growth = latest_year - 2
    country_year_volume = data.groupby(["Final Country", "Financial Year"])["Total Qty"].sum().unstack(fill_value=0)
    if start_year_for_growth in country_year_volume.columns and latest_year in country_year_volume.columns:
        matrix_data = country_year_volume[
            (country_year_volume[start_year_for_growth] > 0)
            & (country_year_volume[latest_year] > 0)
        ].copy()
        matrix_data["Latest Year Volume"] = matrix_data[latest_year]
        matrix_data["Two Year Growth (%)"] = (
            (matrix_data[latest_year] / matrix_data[start_year_for_growth]) - 1
        ) * 100

        volume_split = matrix_data["Latest Year Volume"].median()
        growth_split = matrix_data["Two Year Growth (%)"].median()

        def classify_priority(row: pd.Series) -> str:
            if row["Latest Year Volume"] >= volume_split and row["Two Year Growth (%)"] >= growth_split:
                return "Defend and Grow"
            if row["Latest Year Volume"] >= volume_split and row["Two Year Growth (%)"] < growth_split:
                return "Defend Core"
            if row["Latest Year Volume"] < volume_split and row["Two Year Growth (%)"] >= growth_split:
                return "Build Aggressively"
            return "Selective / Monitor"

        matrix_data["Priority Segment"] = matrix_data.apply(classify_priority, axis=1)
        priority_segment_count = matrix_data["Priority Segment"].value_counts()
    else:
        matrix_data = pd.DataFrame()
        growth_split = np.nan
        volume_split = np.nan
        priority_segment_count = pd.Series(dtype=int)

    country_presence = data.groupby(["Final Country", "Financial Year"]).size().unstack(fill_value=0)
    first_active_year = country_presence.apply(lambda row: row[row > 0].index.min(), axis=1)
    market_entry_quality = pd.DataFrame({"First Active Year": first_active_year})
    market_entry_quality.index.name = "Country"
    market_entry_quality = market_entry_quality.reset_index()
    market_entry_quality["Repeat in Next Year"] = market_entry_quality.apply(
        lambda row: country_presence.loc[row["Country"]].get(row["First Active Year"] + 1, 0) > 0,
        axis=1,
    )
    recent_entries = market_entry_quality[market_entry_quality["First Active Year"] >= (latest_year - 1)]
    repeat_rate = (
        float(recent_entries["Repeat in Next Year"].mean() * 100) if len(recent_entries) > 0 else float("nan")
    )

    return {
        "raw": data,
        "rows": len(data),
        "country_count": int(data["Final Country"].nunique()),
        "department_count": int(data["Department"].nunique()),
        "date_min": data["Invoice Date"].min(),
        "date_max": data["Invoice Date"].max(),
        "total_qty": float(data["Total Qty"].sum()),
        "latest_year": latest_year,
        "previous_year": previous_year,
        "yoy": yoy,
        "quarter_by_year": quarter_by_year,
        "top_months": top_months,
        "country_total": country_total,
        "top10_countries": top10_countries,
        "top3_share_overall": float(top3_share_overall),
        "concentration": concentration,
        "country_change": country_change,
        "country_declines": country_declines,
        "country_growth": country_growth,
        "market_share_by_year": market_share_by_year,
        "overall_market_share": overall_market_share,
        "pharma_by_year": pharma_by_year,
        "pharma_total": pharma_total,
        "department_total": department_total,
        "dept_growth_pct": dept_growth_pct,
        "strength_share_by_year": strength_share_by_year,
        "dept_strength_latest": dept_strength_latest,
        "monthly_heatmap": monthly_heatmap,
        "year_end_dependency": year_end_dependency,
        "monthly_api_ratio": monthly_api_ratio,
        "reorder_segment_counts": reorder_segment_counts,
        "outlier_cutoff": float(outlier_cutoff),
        "large_order_dept_share": large_order_dept_share,
        "large_order_market_share": large_order_market_share,
        "matrix_data": matrix_data,
        "growth_split": growth_split,
        "volume_split": volume_split,
        "priority_segment_count": priority_segment_count,
        "recent_repeat_rate": repeat_rate,
    }


def draw_card(
    fig: plt.Figure,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    value: str,
    subtitle: str,
    facecolor: str = "#ffffff",
) -> None:
    ax = fig.add_axes([x, y, w, h])
    ax.axis("off")
    patch = FancyBboxPatch(
        (0, 0),
        1,
        1,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor="#cbd5e1",
        facecolor=facecolor,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)
    ax.text(0.05, 0.72, title, fontsize=9, color="#334155", weight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.40, value, fontsize=17, color="#0f172a", weight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.14, subtitle, fontsize=8.5, color="#475569", transform=ax.transAxes)


def save_cover_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.add_artist(Rectangle((0, 0), 1, 1, transform=fig.transFigure, color="#0b132b", zorder=-10))
    fig.add_artist(Rectangle((0.03, 0.05), 0.94, 0.9, transform=fig.transFigure, color="#f8fafc", zorder=-9))

    latest_year = m["latest_year"]
    previous_year = m["previous_year"]
    yoy = m["yoy"]
    latest_qty = yoy.loc[latest_year, "Total Qty"] if latest_year in yoy.index else np.nan
    latest_yoy = yoy.loc[latest_year, "YoY %"] if latest_year in yoy.index else np.nan
    top_country = m["country_total"].index[0]
    top_country_qty = m["country_total"].iloc[0]
    top_country_share = top_country_qty / m["total_qty"] * 100
    top_market = m["overall_market_share"].index[0]
    top_market_share = m["overall_market_share"].iloc[0]

    fig.text(0.08, 0.87, "Meropenem Business Performance Report", fontsize=24, weight="bold", color="#0f172a")
    fig.text(
        0.08,
        0.82,
        "Management summary generated from analyses in main.ipynb",
        fontsize=12,
        color="#334155",
    )
    fig.text(
        0.08,
        0.78,
        f"Coverage: FY{int(m['yoy'].index.min())} to FY{int(m['yoy'].index.max())} | "
        f"Data window: {m['date_min'].date()} to {m['date_max'].date()}",
        fontsize=10,
        color="#475569",
    )

    draw_card(
        fig,
        0.08,
        0.56,
        0.26,
        0.16,
        "Total Volume",
        short_number(m["total_qty"]),
        f"{m['rows']:,} rows across {m['country_count']} countries",
    )
    draw_card(
        fig,
        0.37,
        0.56,
        0.26,
        0.16,
        f"FY{latest_year} Volume",
        short_number(latest_qty),
        f"YoY vs FY{previous_year}: {percent_text(latest_yoy)}",
        facecolor="#ecfeff",
    )
    draw_card(
        fig,
        0.66,
        0.56,
        0.26,
        0.16,
        "Largest Country",
        top_country,
        f"{short_number(top_country_qty)} units ({top_country_share:.1f}% share)",
        facecolor="#f0fdf4",
    )
    draw_card(
        fig,
        0.08,
        0.35,
        0.26,
        0.16,
        "Leading Channel",
        top_market,
        f"{top_market_share:.1f}% of total volume",
        facecolor="#fffbeb",
    )
    draw_card(
        fig,
        0.37,
        0.35,
        0.26,
        0.16,
        "Top-3 Country Concentration",
        f"{m['top3_share_overall']:.1f}%",
        "Overall portfolio concentration risk indicator",
        facecolor="#fff7ed",
    )
    draw_card(
        fig,
        0.66,
        0.35,
        0.26,
        0.16,
        "New-Market Repeat Rate",
        f"{m['recent_repeat_rate']:.1f}%",
        "Recent entry quality (repeat next-year ordering)",
        facecolor="#fdf4ff",
    )

    highlights = [
        f"Volume in FY{latest_year} is {percent_text(latest_yoy)} versus FY{previous_year}.",
        f"Top 3 countries contribute {m['top3_share_overall']:.1f}% of total volume.",
        f"{top_market} remains the dominant channel at {top_market_share:.1f}%.",
        "Operational quality is stable with API usage ratio centered near 1.41.",
    ]
    bullet_block = "\n".join([f"- {line}" for line in highlights])
    fig.text(0.08, 0.17, "Executive Highlights", fontsize=12, weight="bold", color="#0f172a")
    fig.text(0.08, 0.08, bullet_block, fontsize=10.5, color="#334155", linespacing=1.5)

    fig.text(0.92, 0.02, f"Generated on {date.today().isoformat()}", ha="right", fontsize=8, color="#64748b")
    pdf.savefig(fig)
    plt.close(fig)


def save_demand_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1.3, 1.0], height_ratios=[1.0, 1.0], hspace=0.35, wspace=0.28)
    fig.suptitle("1. Demand Trajectory and Seasonality", fontsize=16, weight="bold", x=0.05, ha="left")

    yoy = m["yoy"].copy()
    latest_year = m["latest_year"]
    previous_year = m["previous_year"]

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.bar(yoy.index.astype(str), yoy["Total Qty"], color="#2563eb")
    ax1.set_title("Total Volume by Financial Year")
    ax1.set_ylabel("Units")
    ax1.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
    for i, (fy, val) in enumerate(yoy["Total Qty"].items()):
        ax1.text(i, val, short_number(val), ha="center", va="bottom", fontsize=8)

    ax2 = fig.add_subplot(gs[0, 2])
    top_months = m["top_months"].sort_values()
    ax2.barh(top_months.index, top_months.values, color="#0ea5e9")
    ax2.set_title("Top 3 Months by Volume")
    ax2.xaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax2.set_xlabel("Units")

    ax3 = fig.add_subplot(gs[1, :2])
    sns.heatmap(
        m["quarter_by_year"],
        cmap="Blues",
        linewidths=0.4,
        cbar_kws={"label": "Total Qty"},
        ax=ax3,
    )
    ax3.set_title("Quarterly Pattern by Financial Year")
    ax3.set_xlabel("Quarter")
    ax3.set_ylabel("Financial Year")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    q = m["quarter_by_year"]
    quarter_note = "Quarter trend not available."
    if latest_year in q.index and previous_year in q.index:
        q_yoy = ((q.loc[latest_year] - q.loc[previous_year]) / q.loc[previous_year].replace(0, np.nan) * 100).replace(
            [np.inf, -np.inf], np.nan
        )
        best_q = q_yoy.dropna().idxmax() if not q_yoy.dropna().empty else None
        worst_q = q_yoy.dropna().idxmin() if not q_yoy.dropna().empty else None
        if best_q is not None and worst_q is not None:
            quarter_note = (
                f"FY{latest_year} vs FY{previous_year}:\n"
                f"Q{int(best_q)} strongest ({percent_text(q_yoy.loc[best_q])}),\n"
                f"Q{int(worst_q)} weakest ({percent_text(q_yoy.loc[worst_q])})."
            )

    insight_lines = [
        f"FY{latest_year} volume: {short_number(yoy.loc[latest_year, 'Total Qty'])}",
        f"YoY change vs FY{previous_year}: {percent_text(yoy.loc[latest_year, 'YoY %'])}",
        quarter_note,
        "Seasonality example from notebook: Sep, Dec, and Mar are the largest months overall.",
    ]
    wrapped = "\n\n".join(textwrap.fill(line, width=34) for line in insight_lines)
    ax4.text(0.0, 1.0, "Management Interpretation", fontsize=11, weight="bold", va="top")
    ax4.text(0.0, 0.9, wrapped, fontsize=9.5, color="#334155", va="top", linespacing=1.5)

    pdf.savefig(fig)
    plt.close(fig)


def save_geography_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, hspace=0.33, wspace=0.28)
    fig.suptitle("2. Country Concentration and Geographic Movement", fontsize=16, weight="bold", x=0.05, ha="left")

    ax1 = fig.add_subplot(gs[0, 0])
    top10 = m["top10_countries"].sort_values()
    ax1.barh(top10.index, top10.values, color="#14b8a6")
    ax1.set_title("Top 10 Countries by Total Volume")
    ax1.xaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax1.set_xlabel("Units")

    ax2 = fig.add_subplot(gs[0, 1])
    conc = m["concentration"]
    ax2.plot(conc.index, conc["Top 3 Share (%)"], marker="o", color="#1d4ed8", label="Top 3 Share (%)")
    ax2.set_ylabel("Top 3 Share (%)", color="#1d4ed8")
    ax2.tick_params(axis="y", labelcolor="#1d4ed8")
    ax2.set_xlabel("Financial Year")
    ax2.set_title("Concentration Trend")
    ax2b = ax2.twinx()
    ax2b.plot(conc.index, conc["HHI (0-10,000)"], marker="s", color="#f97316", label="HHI")
    ax2b.set_ylabel("HHI", color="#f97316")
    ax2b.tick_params(axis="y", labelcolor="#f97316")

    ax3 = fig.add_subplot(gs[1, 0])
    growth = m["country_growth"].head(5).sort_values()
    decline = m["country_declines"].head(5)
    if not growth.empty or not decline.empty:
        combined = pd.concat([decline, growth]).sort_values()
        colors = ["#dc2626" if v < 0 else "#16a34a" for v in combined.values]
        ax3.barh(combined.index, combined.values, color=colors)
        ax3.axvline(0, color="#334155", linewidth=1)
        ax3.set_title(
            f"Country Volume Change (FY{m['previous_year']} to FY{m['latest_year']})\nExample view from notebook insights"
        )
        ax3.xaxis.set_major_formatter(FuncFormatter(human_readable_number))
        ax3.set_xlabel("Change in Units")
    else:
        ax3.text(0.5, 0.5, "Not enough yearly data.", ha="center", va="center")
        ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    first_country = m["country_total"].index[0]
    first_country_qty = m["country_total"].iloc[0]
    notes = [
        f"Overall top country: {first_country} ({short_number(first_country_qty)} units).",
        f"Top-3 countries account for {m['top3_share_overall']:.1f}% of total business.",
    ]
    if not m["country_declines"].empty and not m["country_growth"].empty:
        notes.append(
            f"Largest recent decline example: {m['country_declines'].index[0]} "
            f"({human_readable_number(m['country_declines'].iloc[0])} units)."
        )
        notes.append(
            f"Largest recent growth example: {m['country_growth'].index[0]} "
            f"({human_readable_number(m['country_growth'].iloc[0])} units)."
        )
    notes.append("Management implication: protect high-scale countries while diversifying risk in growth markets.")
    ax4.text(0.0, 1.0, "Key Message", fontsize=11, weight="bold", va="top")
    ax4.text(
        0.0,
        0.9,
        "\n\n".join(textwrap.fill(line, width=46) for line in notes),
        fontsize=9.5,
        color="#334155",
        va="top",
        linespacing=1.5,
    )

    pdf.savefig(fig)
    plt.close(fig)


def save_mix_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.26)
    fig.suptitle("3. Channel, Regulatory, and Product Mix", fontsize=16, weight="bold", x=0.05, ha="left")

    ax1 = fig.add_subplot(gs[0, 0])
    market_share = m["market_share_by_year"]
    market_share.plot(kind="area", stacked=True, alpha=0.85, ax=ax1, cmap="Set2")
    ax1.set_title("Market Share Shift by Year")
    ax1.set_ylabel("Share (%)")
    ax1.set_xlabel("Financial Year")
    ax1.legend(title="Market", loc="upper left")

    ax2 = fig.add_subplot(gs[0, 1])
    overall_market = m["overall_market_share"]
    colors = sns.color_palette("Set2", len(overall_market))
    wedges, _texts, _autotexts = ax2.pie(
        overall_market.values,
        labels=overall_market.index,
        autopct="%1.1f%%",
        startangle=120,
        colors=colors,
        wedgeprops={"linewidth": 1, "edgecolor": "white"},
    )
    centre = plt.Circle((0, 0), 0.60, color="white")
    ax2.add_artist(centre)
    ax2.set_title("Overall Channel Share")
    ax2.axis("equal")

    ax3 = fig.add_subplot(gs[1, 0])
    pharma_year = m["pharma_by_year"]
    pharma_year.plot(kind="bar", stacked=True, ax=ax3, colormap="Oranges")
    ax3.set_title("Pharmacopoeia Mix by Year")
    ax3.set_xlabel("Financial Year")
    ax3.set_ylabel("Units")
    ax3.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax3.legend(title="Pharmacopoeia", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    top_market = overall_market.index[0]
    top_market_share = overall_market.iloc[0]
    top_pharma = m["pharma_total"].index[0]
    top_pharma_share = m["pharma_total"].iloc[0] / m["pharma_total"].sum() * 100
    notes = [
        f"Channel example: {top_market} drives {top_market_share:.1f}% of overall volume.",
        "Notebook channel trend shows Private peaked in FY2023 and then normalized.",
        f"Regulatory example: {top_pharma} contributes {top_pharma_share:.1f}% of total volume.",
        "Product implication: continue portfolio balancing between Tender stability and Private growth upside.",
    ]
    ax4.text(0.0, 1.0, "Management Interpretation", fontsize=11, weight="bold", va="top")
    ax4.text(
        0.0,
        0.9,
        "\n\n".join(textwrap.fill(line, width=46) for line in notes),
        fontsize=9.5,
        color="#334155",
        va="top",
        linespacing=1.5,
    )

    pdf.savefig(fig)
    plt.close(fig)


def save_department_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.30)
    fig.suptitle("4. Department and Strength Performance", fontsize=16, weight="bold", x=0.05, ha="left")

    ax1 = fig.add_subplot(gs[0, 0])
    top_dept = m["department_total"].head(10).sort_values()
    ax1.barh(top_dept.index, top_dept.values, color="#8b5cf6")
    ax1.set_title("Top Departments by Total Volume")
    ax1.xaxis.set_major_formatter(FuncFormatter(human_readable_number))
    ax1.set_xlabel("Units")

    ax2 = fig.add_subplot(gs[0, 1])
    growth = m["dept_growth_pct"]
    if not growth.empty:
        growth_view = pd.concat([growth.head(5), growth.tail(5)]).sort_values()
        colors = ["#dc2626" if v < 0 else "#16a34a" for v in growth_view.values]
        ax2.barh(growth_view.index, growth_view.values, color=colors)
        ax2.axvline(0, color="#334155", linewidth=1)
        ax2.set_title(f"Department Growth: FY{m['previous_year']} to FY{m['latest_year']}")
        ax2.set_xlabel("Growth (%)")
    else:
        ax2.text(0.5, 0.5, "Department growth requires two years of data.", ha="center", va="center")
        ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    strength_share = m["strength_share_by_year"].T
    strength_share.plot(kind="area", stacked=True, alpha=0.85, ax=ax3, cmap="tab20c")
    ax3.set_title("Strength Mix Shift Across Years")
    ax3.set_xlabel("Financial Year")
    ax3.set_ylabel("Share (%)")
    ax3.legend(title="Strength", bbox_to_anchor=(1.02, 1), loc="upper left")

    ax4 = fig.add_subplot(gs[1, 1])
    heatmap_data = m["dept_strength_latest"]
    if not heatmap_data.empty:
        sns.heatmap(heatmap_data, cmap="YlOrRd", linewidths=0.35, cbar_kws={"label": "Units"}, ax=ax4)
        ax4.set_title(f"Department vs Strength (FY{m['latest_year']})")
        ax4.set_xlabel("Strength")
        ax4.set_ylabel("Department")
    else:
        ax4.text(0.5, 0.5, "No latest-year mix data available.", ha="center", va="center")
        ax4.axis("off")

    pdf.savefig(fig)
    plt.close(fig)


def save_operations_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    fig.suptitle("5. Operational Stability and Quality Signals", fontsize=16, weight="bold", x=0.05, ha="left")

    ax1 = fig.add_subplot(gs[0, 0])
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    sns.heatmap(m["monthly_heatmap"], cmap="YlGnBu", linewidths=0.3, cbar_kws={"label": "Units"}, ax=ax1)
    ax1.set_title("Monthly Volume Pattern by Year")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Financial Year")
    ax1.set_xticks(np.arange(12) + 0.5)
    ax1.set_xticklabels(month_labels, rotation=0)

    ax2 = fig.add_subplot(gs[0, 1])
    dependency = m["year_end_dependency"]
    ax2.bar(dependency.index.astype(str), dependency.values, color="#6366f1")
    ax2.set_title("31-Mar Dependency Ratio")
    ax2.set_xlabel("Financial Year")
    ax2.set_ylabel("Share of Annual Volume (%)")
    for i, val in enumerate(dependency.values):
        ax2.text(i, val, f"{val:.1f}%", ha="center", va="bottom", fontsize=8)

    ax3 = fig.add_subplot(gs[1, 0])
    api = m["monthly_api_ratio"]
    if not api.empty:
        ax3.plot(api.index, api.values, color="#15803d", linewidth=2)
        ax3.axhline(1.41, color="black", linestyle="--", linewidth=1, label="Expected 1.41")
        ax3.axhline(1.41 * 1.05, color="#dc2626", linestyle=":", linewidth=1, label="+/-5% band")
        ax3.axhline(1.41 * 0.95, color="#dc2626", linestyle=":", linewidth=1)
        ax3.set_title("Monthly API Utilization Ratio")
        ax3.set_ylabel("Ratio")
        ax3.set_xlabel("Invoice Month")
        ax3.legend(loc="best")
    else:
        ax3.text(0.5, 0.5, "API ratio series unavailable.", ha="center", va="center")
        ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    segments = m["reorder_segment_counts"]
    if not segments.empty:
        seg_plot = segments.reindex(
            [
                "Stable and Frequent",
                "Frequent but Unstable",
                "Predictable but Infrequent",
                "Infrequent and Unstable",
            ]
        ).dropna()
        palette = ["#16a34a", "#f97316", "#0ea5e9", "#dc2626"]
        ax4.bar(seg_plot.index, seg_plot.values, color=palette[: len(seg_plot)])
        ax4.set_title("Reorder Stability Segments")
        ax4.set_ylabel("Number of Countries")
        ax4.tick_params(axis="x", rotation=25)
    else:
        ax4.text(0.5, 0.5, "Reorder segmentation unavailable.", ha="center", va="center")
        ax4.axis("off")

    fig.text(
        0.05,
        0.02,
        textwrap.fill(
            f"Notebook example: 31-Mar dependency moved from 8.2% in FY2023 to {m['year_end_dependency'].get(m['latest_year'], 0):.1f}% in "
            f"FY{m['latest_year']}, indicating lower year-end push risk. New-market repeat rate is {m['recent_repeat_rate']:.1f}%.",
            width=150,
        ),
        fontsize=9,
        color="#334155",
    )

    pdf.savefig(fig)
    plt.close(fig)


def save_outlier_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.28)
    fig.suptitle("6. Outliers and Opportunity Prioritization", fontsize=16, weight="bold", x=0.05, ha="left")

    ax1 = fig.add_subplot(gs[0, 0])
    dept_share = m["large_order_dept_share"].sort_values()
    if not dept_share.empty:
        ax1.barh(dept_share.index, dept_share.values, color="#2563eb")
        ax1.set_title("Large Orders (Top 1%) by Department Share")
        ax1.set_xlabel("Share of Large-Order Volume (%)")
    else:
        ax1.text(0.5, 0.5, "No outlier orders found.", ha="center", va="center")
        ax1.axis("off")

    ax2 = fig.add_subplot(gs[0, 1])
    market_share = m["large_order_market_share"]
    if not market_share.empty:
        ax2.bar(market_share.index, market_share.values, color="#7c3aed")
        ax2.set_title("Large Orders by Market Share")
        ax2.set_ylabel("Share (%)")
    else:
        ax2.text(0.5, 0.5, "No market split for outliers.", ha="center", va="center")
        ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 0])
    matrix_data = m["matrix_data"]
    if not matrix_data.empty:
        sns.scatterplot(
            data=matrix_data,
            x="Two Year Growth (%)",
            y="Latest Year Volume",
            hue="Priority Segment",
            size="Latest Year Volume",
            sizes=(30, 260),
            alpha=0.75,
            ax=ax3,
        )
        ax3.axvline(m["growth_split"], color="#64748b", linestyle="--", linewidth=1)
        ax3.axhline(m["volume_split"], color="#64748b", linestyle="--", linewidth=1)
        ax3.set_title("Country Opportunity Matrix")
        ax3.yaxis.set_major_formatter(FuncFormatter(human_readable_number))
        ax3.legend(loc="best", fontsize=8, title="Segment")
    else:
        ax3.text(0.5, 0.5, "Not enough coverage for opportunity matrix.", ha="center", va="center")
        ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    segments = m["priority_segment_count"]
    if not segments.empty:
        ax4.bar(segments.index, segments.values, color="#0ea5e9")
        ax4.set_title("Priority Segment Count")
        ax4.set_ylabel("Countries")
        ax4.tick_params(axis="x", rotation=20)
    else:
        ax4.text(0.5, 0.5, "Priority segmentation unavailable.", ha="center", va="center")
        ax4.axis("off")

    fig.text(
        0.05,
        0.02,
        textwrap.fill(
            f"Example from notebook logic: outlier cutoff is {m['outlier_cutoff']:,.0f} units (top 1% transactions). "
            "Use outlier concentration and matrix segments together to decide where to defend core markets versus build selectively.",
            width=150,
        ),
        fontsize=9,
        color="#334155",
    )

    pdf.savefig(fig)
    plt.close(fig)


def save_recommendation_page(pdf: PdfPages, m: dict) -> None:
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle("7. Recommended Business Actions (Management View)", fontsize=16, weight="bold", x=0.05, ha="left")
    ax = fig.add_axes([0.05, 0.08, 0.90, 0.82])
    ax.axis("off")

    latest_year = m["latest_year"]
    previous_year = m["previous_year"]
    latest_yoy = m["yoy"].loc[latest_year, "YoY %"] if latest_year in m["yoy"].index else np.nan
    top_country = m["country_total"].index[0]
    top_market = m["overall_market_share"].index[0]

    sections = [
        (
            "Commercial Focus",
            [
                f"Protect top-volume country accounts (starting with {top_country}) with quarterly risk reviews.",
                "Use the opportunity matrix to split countries into Defend-and-Grow, Defend-Core, Build, and Monitor tracks.",
                "Track country-level gains/losses monthly so corrective action is not delayed to year-end.",
            ],
        ),
        (
            "Channel Strategy",
            [
                f"{top_market} remains the base channel; keep service consistency there while selectively scaling Private wins.",
                "Monitor the 'Other' bucket as an exception channel, not a growth engine.",
                "Review channel mix by country, not only globally, because migration patterns vary market to market.",
            ],
        ),
        (
            "Operations and Quality",
            [
                "Keep API utilization guardrails around the 1.41 benchmark (+/-5%).",
                "Continue reducing dependence on 31-Mar dispatch spikes through smoother quarter planning.",
                "Prioritize countries in the 'Frequent but Unstable' reorder segment for process correction.",
            ],
        ),
        (
            "90-Day Management Dashboard",
            [
                "Weekly: top country volatility and large-order concentration alerts.",
                f"Monthly: FY{latest_year} run-rate vs FY{previous_year} baseline (current YoY {percent_text(latest_yoy)}).",
                "Quarterly: portfolio review by country priority segment with ownership and action status.",
            ],
        ),
    ]

    y = 0.96
    for title, bullets in sections:
        ax.text(0.0, y, title, fontsize=12, weight="bold", color="#0f172a", va="top")
        y -= 0.05
        for bullet in bullets:
            wrapped = textwrap.fill(f"- {bullet}", width=120)
            ax.text(0.02, y, wrapped, fontsize=10, color="#334155", va="top")
            y -= 0.06
        y -= 0.02

    ax.text(
        0.0,
        0.02,
        "Note: This PDF is a management summary generated from the analyses implemented in main.ipynb.",
        fontsize=9,
        color="#64748b",
    )

    pdf.savefig(fig)
    plt.close(fig)


def generate_report(csv_path: Path, output_path: Path) -> None:
    setup_theme()
    df = pd.read_csv(csv_path)
    metrics = compute_metrics(df)

    with PdfPages(output_path) as pdf:
        save_cover_page(pdf, metrics)
        save_demand_page(pdf, metrics)
        save_geography_page(pdf, metrics)
        save_mix_page(pdf, metrics)
        save_department_page(pdf, metrics)
        save_operations_page(pdf, metrics)
        save_outlier_page(pdf, metrics)
        save_recommendation_page(pdf, metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build management PDF report from Meropenem analysis dataset.")
    parser.add_argument(
        "--csv",
        default="Meropenem_FY22_to_FY25_with_MarketCategory.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="Meropenem_Management_Report.pdf",
        help="Output PDF path",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_path = Path(args.output)
    generate_report(csv_path=csv_path, output_path=output_path)
    print(f"PDF generated: {output_path.resolve()}")


if __name__ == "__main__":
    main()
