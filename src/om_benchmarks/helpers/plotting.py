from pathlib import Path

import altair as alt
import polars as pl

# from PIL import Image


def create_benchmark_charts(df: pl.DataFrame, save_dir: str = "benchmark_plots") -> None:
    """Create benchmark visualization charts using Altair with proper column names"""

    save_path = Path(save_dir)
    save_path.mkdir(exist_ok=True)

    alt.data_transformers.enable("json")
    alt.themes.enable("opaque")

    _create_and_save_perf_chart(df, save_path)
    _create_and_save_file_size_chart(df, save_path)
    _create_and_save_memory_usage_chart(df, save_path)

    print(f"Charts saved to {save_path}")


def _create_and_save_perf_chart(df, save_path):
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

    perf_chart.save(str(save_path / "performance_comparison.png"), ppi=400)
    # Image.open(str(save_path / "performance_comparison.png")).show()


def _create_and_save_file_size_chart(df, save_path):
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

    file_size_chart.save(str(save_path / "file_size_comparison.png"), ppi=400)
    # Image.open(str(save_path / "file_size_comparison.png")).show()


def _create_and_save_memory_usage_chart(df, save_path):
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

    memory_chart.save(str(save_path / "memory_usage.png"), ppi=400)
    # Image.open(str(save_path / "memory_usage.png")).show()
