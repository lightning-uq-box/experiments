from huggingface_hub import snapshot_download
from torchgeo.datasets.utils import extract_archive
from glob import glob

snapshot_download(
    repo_id="torchgeo/tropical_cyclone",
    repo_type="dataset",
    local_dir="./",
    cache_dir=None,
)

# Extract the downloaded files
for file in glob("*.tar.gz"):
    extract_archive(file)
