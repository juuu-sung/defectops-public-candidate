"""Export 120 pinned normal tiles only; no model, scoring or test decoding."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

from PIL import Image, ImageDraw, ImageFont, __version__ as PILLOW_VERSION

NAME = "mpi-normal-visual-review-v1"
CONFIG_SHA = "27d773bbb8d70788948d59365352d0aa1670b411dfcab5b150c552aac9930807"
PHOTO_IDS = ["1554", "1681", "1570", "1582", "0432", "1671", "0616", "0705", "1365", "0290"]
REGIONS = [f"grid{r}-{c}_s1024" for r in range(1, 5) for c in range(1, 4)]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def load_config(path):
    raw = Path(path).read_bytes().replace(b"\r\n", b"\n")
    if sha(raw) != CONFIG_SHA:
        raise ValueError("Pinned review configuration does not match; do not edit the selection")
    config = json.loads(raw)
    if config["regions"] != REGIONS or [p["photo_id"] for p in config["photos"]] != PHOTO_IDS:
        raise ValueError("Unexpected review inventory")
    for i, photo in enumerate(config["photos"]):
        if photo["cohort"] != ("development" if i < 5 else "calibration"):
            raise ValueError("Unexpected normal cohort")
        if [t["region"] for t in photo["tiles"]] != REGIONS:
            raise ValueError("Unexpected tile order")
        for tile in photo["tiles"]:
            expected = f"extracted/{tile['region']}/train/good/{photo['photo_id']}.png"
            if (tile["member"] != expected or tile["photo_id"] != photo["photo_id"]
                    or (tile["width"], tile["height"]) != (256, 256)):
                raise ValueError("Only pinned 256x256 normal members are allowed")
    return config, sha(raw)


def decode(raw, tile):
    if sha(raw) != tile["member_sha256"]:
        raise ValueError(f"Original PNG checksum mismatch: {tile['member']}")
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
            raise ValueError(f"Unexpected PNG encoding: {tile['member']}")
        image.verify()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        rgb = image.convert("RGB")
    if sha(rgb.tobytes()) != tile["decoded_rgb_sha256"]:
        raise ValueError(f"Decoded pixel checksum mismatch: {tile['member']}")
    return rgb


@contextmanager
def locked(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".run.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another export is running; existing results were not changed") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def inventory(root):
    return {p.relative_to(root).as_posix(): file_sha(p)
            for p in sorted(root.rglob("*")) if p.is_file() and p != root / "receipt.json"}


def seal(root, identity):
    write(root / "receipt.json", json_bytes({"identity": identity, "files": inventory(root)}))


def verify(root, identity):
    receipt = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity or receipt["files"] != inventory(root):
        raise ValueError(f"Existing evidence differs; preserved without overwrite: {root}")


def contact_sheet(photo, images):
    # A labeled contact sheet, NOT geometric reconstruction of overlapping source crops.
    card = Image.new("RGB", (824, 1224), "#f4f5f7")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=15)
    draw.text((12, 12), f"NORMAL {photo['photo_id']} | {photo['cohort']}", fill="black", font=font)
    draw.text((12, 36), "Original 256x256 pixels | no heatmap | not a stitched board", fill="black", font=font)
    for i, (region, image) in enumerate(zip(REGIONS, images)):
        x, y = 12 + (i % 3) * 272, 68 + (i // 3) * 288
        label = region + (" [focus]" if region == photo["focus_region"] else "")
        draw.text((x, y), label, fill="black", font=font)
        card.paste(image, (x, y + 24))
    return card


def export_photo(z, photo, root, identity):
    target = root / photo["photo_id"]
    photo_identity = {**identity, "photo": photo}
    if target.exists():
        verify(target, photo_identity)
        # Compare cached original bytes to pinned source hashes, not just their receipt.
        for tile in photo["tiles"]:
            if file_sha(target / (tile["region"] + ".png")) != tile["member_sha256"]:
                raise ValueError("Cached original differs from pinned fingerprint")
        return False
    stage = Path(tempfile.mkdtemp(prefix=f".{photo['photo_id']}-attempt-", dir=root))
    images = []
    for tile in photo["tiles"]:
        info = z.getinfo(tile["member"])
        if info.is_dir() or info.flag_bits & 1 or not 0 < info.file_size < 2 * 1024 * 1024:
            raise ValueError(f"Invalid selected ZIP member: {info.filename}")
        raw = z.read(info)  # ZIP CRC checked here; never read any test or mask member.
        images.append(decode(raw, tile))
        write(stage / (tile["region"] + ".png"), raw)
    buffer = io.BytesIO()
    contact_sheet(photo, images).save(buffer, format="PNG")
    write(stage / "contact.png", buffer.getvalue())
    write(stage / "photo.json", json_bytes(photo))
    seal(stage, photo_identity)
    os.rename(stage, target)
    return True


def viewer(config):
    e = html.escape
    cards = []
    for photo in config["photos"]:
        pid = photo["photo_id"]
        links = " · ".join(f'<a href="photos/{pid}/{r}.png">{r}</a>' for r in REGIONS)
        cards.append(f'<section><h2>{pid} · {photo["cohort"]}</h2><p>{e(photo["reason"])}</p>'
                     f'<p>우선 확인: {e(photo["focus_region"] or "12영역 전체; 직접 경계로 선정된 사진 아님")}</p>'
                     f'<img src="photos/{pid}/contact.png" alt="정상 {pid}의 12영역 원본">'
                     f'<p>{links}</p></section>')
    return ('''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MPI-PCB 정상 사진 검토</title><style>
body{font:17px/1.6 system-ui,sans-serif;max-width:1000px;margin:32px auto;padding:0 20px;color:#202632}
section{margin:36px 0;border-top:1px solid #ccd2da}img{max-width:100%;height:auto}a{color:#1749a5}
</style><h1>MPI-PCB 정상 사진 10장 검토</h1>
<p>정상 경고 기준이 표본 구성에 따라 흔들리는 이유를 외관에서 확인하기 위한 자료입니다.
정상으로 등록된 사진이며, 점수가 높다는 이유만으로 불량 또는 오류라고 판단하지 않습니다.</p>
<p>색 보정·히트맵·재추론은 없습니다. 각 타일은 압축파일의 PNG 원본 그대로이며,
3열 4행 배열은 검토용 목록입니다. 겹치는 영역을 포함하므로 전체 기판을 복원한 이미지가 아닙니다.
화면에서는 축소될 수 있습니다. 아래 영역 링크로 원본을 열어 확인하십시오.</p>
<h2>확인할 내용</h2><ol><li>정상 부품·패턴의 차이, 위치나 밝기 차이가 보이는지 확인합니다.</li>
<li>흐림·잘림·파일 이상이 의심되면 위치와 관찰 사실만 기록합니다.</li>
<li>개발 사진은 grid3-2, 보정 사진은 표시된 영역을 우선 보되 다른 11영역과도 비교합니다.</li></ol>
<p>선정은 기존 결과를 본 뒤 이루어졌습니다. 대표성·인과관계·성능 개선을 입증하지 않습니다.
라벨 변경, 자료 제외, 임곗값 수정, 자동 합격, 생산 조치는 하지 않습니다.
검토 결과 저장 버튼은 없습니다. 사진 ID·영역·관찰 내용을 별도 기록하십시오.</p>
<p>데이터 이용 조건이 명확해지기 전 원본·시각화를 공개하거나 Git·Notion에 올리지 마십시오.</p>
''' + "\n".join(cards) + "</html>").encode("utf-8")


def bundle(result, handoff):
    files = {p.relative_to(result).as_posix(): p for p in result.rglob("*") if p.is_file()}
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(files) or set(z.namelist()) != set(files):
                raise ValueError("Existing handoff inventory differs; preserved")
            for name, path in files.items():
                if sha(z.read(name)) != file_sha(path):
                    raise ValueError("Existing handoff content differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".normal-review-", suffix=".zip", dir=handoff.parent)
    os.close(fd)
    temp = Path(temp)
    try:
        with zipfile.ZipFile(temp, "w", zipfile.ZIP_DEFLATED) as z:
            for name in sorted(files):
                z.write(files[name], name)
        os.link(temp, handoff)  # Atomic publication; never overwrite an existing handoff.
    finally:
        temp.unlink(missing_ok=True)


def run(project, archive):
    project, archive = Path(project).resolve(), Path(archive).resolve()
    config, config_sha = load_config(project / "configs" / f"{NAME}.json")
    print("Verifying original archive SHA256; no test images are decoded...", flush=True)
    if file_sha(archive) != config["archive_sha256"]:
        raise ValueError("Original archive SHA256 mismatch; no image exported")
    identity = {"config_sha256": config_sha, "archive_sha256": config["archive_sha256"],
                "code_sha256": sha(Path(__file__).read_bytes().replace(b"\r\n", b"\n")),
                "pillow_version": PILLOW_VERSION}
    root = project / "artifacts" / "analyses" / NAME
    handoff = project / "artifacts" / "handoff" / f"{NAME}.zip"
    fresh = 0
    with locked(root), zipfile.ZipFile(archive) as z:
        selected = {t["member"] for p in config["photos"] for t in p["tiles"]}
        names = z.namelist()  # Directory metadata only, including names; no test payload read.
        if any(names.count(n) != 1 for n in selected):
            raise ValueError("Selected normal member missing or duplicated")
        photos_root = root / "photos"
        photos_root.mkdir(exist_ok=True)
        for i, photo in enumerate(config["photos"], 1):
            fresh += int(export_photo(z, photo, photos_root, identity))
            print(f"Verified normal photo {i}/10: {photo['photo_id']}", flush=True)
        result = root / "result"
        report = {"status": "normal_visual_material_prepared_not_reviewed", "identity": identity,
                  "photo_ids": PHOTO_IDS, "verified_normal_photos": 10, "verified_normal_tiles": 120,
                  "test_images_decoded": 0, "test_masks_decoded": 0, "model_scores_computed": 0,
                  "training": False, "threshold_changed": False, "data_excluded": False,
                  "automatic_pass_allowed": False, "appearance_cause_confirmed": False,
                  "rights_note": "No public redistribution or Git/Notion image upload; private review only."}
        if result.exists():
            verify(result, identity)
            if json.loads((result / "report.json").read_text(encoding="utf-8")) != report:
                raise ValueError("Existing report differs; preserved")
        else:
            stage = Path(tempfile.mkdtemp(prefix=".result-attempt-", dir=root))
            for photo in config["photos"]:
                shutil.copytree(photos_root / photo["photo_id"], stage / "photos" / photo["photo_id"])
            write(stage / "index.html", viewer(config))
            write(stage / "selection.json", json_bytes(config))
            write(stage / "report.json", json_bytes(report))
            seal(stage, identity)
            os.rename(stage, result)
        bundle(result, handoff)
    return {**report, "source_tiles_decoded_this_invocation": fresh * 12,
            "viewer": str(result / "index.html"), "handoff": str(handoff),
            "handoff_sha256": file_sha(handoff)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.project, args.archive), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, zipfile.BadZipFile, RuntimeError, SyntaxError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}\nOriginal data and previous output preserved.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
