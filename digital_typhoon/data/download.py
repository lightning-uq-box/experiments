from huggingface_hub import snapshot_download
from torchgeo.datasets.utils import extract_archive

snapshot_download(
    repo_id="torchgeo/digital_typhoon",
    repo_type="dataset",
    local_dir="./",
    cache_dir=None,
)


chunk_size = 2**15  # same as torchvision
path = "WP.tar.gz"
with open(path, "wb") as f:
    for suffix in ["aa", "ab"]:
        with open(path + suffix, "rb") as g:
            while chunk := g.read(chunk_size):
                f.write(chunk)

# Extract the concatenated tarball
extract_archive(path)
