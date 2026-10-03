"""Pinned public research assets; never extract an archive-controlled destination."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import tempfile
import time
import urllib.request
import zipfile

from PIL import Image
from scripts import fit_mpi_pcb_normal_bank as fit

ASSETS = {
    "teacher": {
        "url": "https://github.com/open-edge-platform/anomalib/releases/download/efficientad_pretrained_weights/efficientad_pretrained_weights.zip",
        "sha256": "c09aeaa2b33f244b3261a5efdaeae8f8284a949470a4c5a526c61275fe62684a",
        "filename": "efficientad_pretrained_weights.zip", "maximum_bytes": 1024**3,
    },
    "imagenette": {
        "url": "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2.tgz",
        "sha256": "6cbfac238434d89fe99e651496f0812ebc7a10fa62bd42d6874042bf01de4efd",
        "filename": "imagenette2.tgz", "maximum_bytes": 2 * 1024**3,
    },
}


def download(root, spec, allow_download, opener=urllib.request.urlopen):
    root.mkdir(parents=True, exist_ok=True)
    final = root / spec["filename"]
    if final.exists():
        if fit.sha256_file(final) != spec["sha256"]:
            raise ValueError(f"Existing asset hash differs; preserved: {final}")
        return final
    if not allow_download:
        raise ValueError("Missing research asset; download requires --allow-download")
    attempt = Path(tempfile.mkdtemp(prefix="download-", dir=root)) / spec["filename"]
    print(f"Downloading {spec['filename']}; incomplete files are preserved.", flush=True)
    size, last = 0, 0
    start = time.monotonic()
    with opener(spec["url"], timeout=60) as response, attempt.open("xb") as stream:
        if not response.geturl().startswith("https://"):
            raise ValueError("Non-HTTPS asset redirect")
        while block := response.read(1024**2):
            size += len(block)
            if size > spec["maximum_bytes"] or time.monotonic() - start > 7200:
                raise RuntimeError("Asset size/download time limit exceeded; partial file preserved")
            stream.write(block)
            if size - last >= 100 * 1024**2:
                print(f"  {size / 1024**2:.0f} MiB received", flush=True)
                last = size
    if fit.sha256_file(attempt) != spec["sha256"]:
        raise ValueError(f"Downloaded asset hash differs; preserved: {attempt}")
    if final.exists():
        raise FileExistsError(final)
    attempt.rename(final)
    return final


def auxiliary_names(members):
    """Only train JPEGs. Pick 96 names by hash before decoding; never use val."""
    names = []
    seen = set()
    for member in members:
        p = PurePosixPath(member.name)
        if (p.is_absolute() or ".." in p.parts or "\\" in member.name
                or member.issym() or member.islnk()):
            raise ValueError("Unsafe auxiliary archive member")
        if member.name in seen:
            raise ValueError("Duplicate auxiliary archive member")
        seen.add(member.name)
        if (len(p.parts) == 4 and p.parts[:2] == ("imagenette2", "train")
                and p.suffix.lower() in (".jpg", ".jpeg") and not p.name.startswith(".")):
            if not member.isfile() or not 0 < member.size <= 16 * 1024**2:
                raise ValueError("Invalid auxiliary image member")
            names.append(member.name)
    if len(names) < 96:
        raise ValueError("Too few auxiliary train images")
    return sorted(names, key=lambda n: hashlib.sha256(("mpi-efficientad-probe-v1\0" + n).encode()).hexdigest())[:96]


def prepare(root, allow_download):
    """Returns immutable local teacher and 96 auxiliary files; no redistribution."""
    started = time.perf_counter()
    paths = {k: download(root, v, allow_download) for k, v in ASSETS.items()}
    signature = fit.digest(ASSETS)
    names = ["assets.json", "teacher.pth"] + [f"aux-{i:03d}.jpg" for i in range(96)]
    if (root / "prepared").exists():
        fit.read_phase(root, "prepared", signature, names)
    else:
        with fit.phase_stage(root, "prepared") as stage:
            with zipfile.ZipFile(paths["teacher"]) as z:
                selected = [i for i in z.infolist() if PurePosixPath(i.filename).name == "pretrained_teacher_small.pth"]
                if len(selected) != 1 or not 0 < selected[0].file_size < 100 * 1024**2:
                    raise ValueError("Expected one small teacher weights file")
                (stage / "teacher.pth").write_bytes(z.read(selected[0]))
            with tarfile.open(paths["imagenette"], "r:gz") as source:
                chosen = auxiliary_names(source.getmembers())
            chosen_indices = {name: i for i, name in enumerate(chosen)}
            rows = []
            # Stream the gzip once, rather than seeking backwards for every selected JPEG.
            with tarfile.open(paths["imagenette"], "r|gz") as source:
                for member in source:
                    if member.name not in chosen_indices:
                        continue
                    raw = source.extractfile(member).read()
                    with Image.open(io.BytesIO(raw)) as image:
                        if image.format != "JPEG":
                            raise ValueError("Auxiliary train image is not JPEG")
                        image.load()
                    i = chosen_indices[member.name]
                    target = f"aux-{i:03d}.jpg"
                    (stage / target).write_bytes(raw)
                    rows.append({"index": i, "source": member.name, "file": target,
                                 "sha256": hashlib.sha256(raw).hexdigest()})
            if len(rows) != 96:
                raise ValueError("Incomplete auxiliary extraction")
            fit.write_json(stage / "assets.json", {"sources": ASSETS, "auxiliary": sorted(rows, key=lambda r: r["index"]),
                "teacher_sha256": fit.sha256_file(stage / "teacher.pth"),
                "scope": "non-commercial research probe only; images and weights excluded from handoff"})
            fit.seal_phase(stage, "prepared", signature, {}, names)
    manifest = json.loads((root / "prepared/assets.json").read_bytes())
    return root / "prepared", manifest, time.perf_counter() - started
