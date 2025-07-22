import os

import cdsapi
import typer

from om_benchmarks.AsyncTyper import AsyncTyper
from om_benchmarks.era5 import configure_era5_request

app = AsyncTyper()


@app.command()
async def main(
    download_again: bool = typer.Option(False, help="Whether to download the dataset again even if it already exists."),
    target_download: str = typer.Option("downloaded_data.nc", help="Path where downloaded dataset will be saved."),
):
    if os.path.exists(target_download) and not download_again:
        print(f"Dataset already exists at {target_download}. Skipping download.")
    else:
        dataset, request = configure_era5_request()
        client = cdsapi.Client()
        client.retrieve(dataset, request).download(target_download)
        print(f"Downloaded dataset to {target_download}")


if __name__ == "__main__":
    app()
