from pathlib import Path

import altair as alt
import polars as pl

# from PIL import Image

alt.data_transformers.enable("json")
alt.themes.enable("opaque")


def create_and_save_perf_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "performance_chart.png"):
    output_path = save_dir / file_name

    perf_chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X(
                "format:N",
                title="Format",
                sort="-y",
                axis=alt.Axis(
                    labelAngle=-45,
                    labelAlign="right",
                    labelBaseline="middle",
                    labelLimit=150,
                    labelOverlap=False,
                ),
            ),
            y=alt.Y("mean_time:Q", title="Mean Time (seconds)").scale(type="linear"),
            color=alt.Color("format:N", title="Format").scale(scheme="tableau20"),
            column=alt.Column(
                "operation:N",
                title="Operation Type",
                header=alt.Header(labelFontSize=14, titleFontSize=14, titleFontWeight="bold"),
            ),
            tooltip=["format:N", "mean_time:Q", "std_time:Q", "memory_usage_bytes:Q", "file_size_bytes:Q"],
        )
        .resolve_scale(y="independent", x="independent")
        .properties(
            width=500,
            height=300,
            title={"text": "Performance Comparison by Format", "subtitle": "Lower is better (logarithmic scale)"},
        )
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridOpacity=0.2)
    )

    perf_chart.save(output_path, ppi=400)
    # Image.open(output_path).show()


def create_and_save_file_size_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "file_size_comparison.png"):
    output_path = save_dir / file_name
    write_df = df.filter(pl.col("operation") == "write").filter(pl.col("file_size_bytes") > 0)

    file_size_chart = (
        alt.Chart(write_df)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("format:N", title="Format", sort="-y"),
            y=alt.Y("file_size_bytes:Q", title="File Size (bytes)"),
            color=alt.Color("format:N", title="Format").scale(scheme="tableau20"),
            tooltip=["format:N", "file_size_bytes:Q"],
        )
        .properties(width=400, height=200, title="File Size by Format")
    )

    file_size_chart.save(output_path, ppi=400)
    # Image.open(output_path).show()


def create_and_save_memory_usage_chart(df: pl.DataFrame, save_dir: Path, file_name: str = "memory_usage.png"):
    output_path = save_dir / file_name
    memory_chart = (
        alt.Chart(df)
        .mark_bar(cornerRadius=2)
        .encode(
            x=alt.X("format:N", title="Format", sort="-y"),
            y=alt.Y("memory_usage_bytes:Q", title="Memory Usage (bytes)"),
            color=alt.Color("format:N", title="Format").scale(scheme="tableau20"),
            column=alt.Column(
                "operation:N",
                title="Operation Type",
                header=alt.Header(labelFontSize=14, titleFontSize=14, titleFontWeight="bold"),
            ),
            tooltip=["format:N", "operation:N", "memory_usage_bytes:Q"],
        )
        .resolve_scale(y="independent", x="independent")
        .properties(width=400, height=200, title="Memory Usage by Format and Operation")
    )

    memory_chart.save(output_path, ppi=400)
    # Image.open(output_path).show()
