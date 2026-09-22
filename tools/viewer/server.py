"""Read-only, dependency-free browser for preprocessed JSCAS frames and predictions."""

import argparse
from bisect import bisect_right
import csv
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
import shutil
import threading
from urllib.parse import parse_qs, urlsplit
import webbrowser


PHASE_SOURCE = "https://github.com/modafone/jscaschallenge2026/blob/main/basic_challenge/readme.md"
# Chinese translations are display aids; Japanese descriptions retain the official definitions.
PHASES = {
    "0": ("手术开始至阶段操作开始", "手術開始～工程開始まで", "#8593ad"),
    "1": ("骶岬尾侧内侧游离／正中右侧 ME", "岬角より尾側の内側授動。正中より右側のMesenteric excision（ME）", "#4e92ef"),
    "2": ("骶岬头侧、IMA/SRA 右侧内侧游离", "岬角より頭側かつIMA/SRAより右側の内側授動", "#56b8ef"),
    "3": ("IMA 处理", "IMA処理", "#69c9bc"),
    "4": ("IMA/SRA 左侧内侧游离", "IMA/SRAより左側の内側授動", "#a7cf75"),
    "5": ("SDJ 头侧外侧游离", "SDJより頭側の外側授動", "#dcc45c"),
    "6": ("SDJ 尾侧外侧游离／正中左侧 ME", "SDJより尾側の外側授動。正中より左側のME", "#eaa45c"),
    "7": ("预定离断线上的直肠系膜处理", "切離予定線上の直腸間膜処理", "#df827d"),
    "81": ("直肠离断至吻合的其他场景", "直腸切離～吻合で工程82, 83に属さないシーン", "#bba1eb"),
    "82": ("直肠离断", "直腸切離", "#d883be"),
    "83": ("吻合", "吻合", "#8d93ed"),
    "9": ("IMV（LCA）处理", "IMV（LCA）処理", "#c2a080"),
    "13": ("阶段操作结束至手术结束", "工程終了～手術終了まで", "#93a79b"),
    "x": ("体外", "体外", "#4c5567"),
    "e": ("视频结束标记", "終了マーカー", "#394150"),
}
LABELS = tuple(k for k in PHASES if k not in {"x", "e"})
FRAME_NAME = re.compile(r"(\d{2})_(\d{6,})(?:\.png)?\Z")


def read_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def csv_rows(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError(f"{path}: CSV 表头缺失或重复")
        for line, row in enumerate(reader, 2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{line}: CSV 列数不一致")
            yield line, row


def frame_identity(name, video=None, frame=None):
    match = FRAME_NAME.fullmatch(name)
    if not match:
        raise ValueError(f"无效帧文件名: {name!r}")
    vid, number = match[1], int(match[2])
    if (video is not None and str(video) != vid) or (frame not in (None, "") and int(frame) != number):
        raise ValueError(f"帧身份冲突: {name}, video_id={video}, frame_index={frame}")
    return vid, number


def probability(value, context):
    if value in (None, ""):
        return None
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{context}: 概率必须是 0 到 1 的有限数值")
    return value


def read_predictions(path, explicit=False):
    """Normalize existing formats without treating prediction labels as ground truth."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        document = read_json(path)
        if not isinstance(document, dict) or not isinstance(document.get("predictions"), list):
            raise ValueError(f"{path}: 不是 predict-head 的预测结果 JSON")
        entries = []
        for index, item in enumerate(document["predictions"], 1):
            if not isinstance(item, dict):
                raise ValueError(f"{path}: 第 {index} 项预测必须是对象")
            scores = item.get("scores", {})
            if not isinstance(scores, dict):
                raise ValueError(f"{path}: 第 {index} 项 scores 必须是对象")
            label = str(item.get("phase_id", ""))
            entries.append((index, {"filename": str(item.get("image", "")).replace("\\", "/").split("/")[-1],
                                   "predicted_phase_id": label, "top_score": scores.get(label),
                                   **{f"probability_{k}": v for k, v in scores.items()}}))
    else:
        entries = csv_rows(path)
    result = {}
    for line, row in entries:
        context = f"{path}:{line}"
        try:
            video, frame = frame_identity(row.get("filename", ""), row.get("video_id"), row.get("frame_index"))
            official = "predicted_phase_id" not in row and "phase-id" in row
            if official and not explicit:
                raise ValueError("两列预测 CSV 必须通过 --prediction 显式指定")
            label = row.get("phase-id") if official else row.get("predicted_phase_id")
            if label not in LABELS:
                raise ValueError(f"未知预测阶段: {label!r}")
            truth = row.get("phase_id") or None
            if truth is not None and truth not in PHASES:
                raise ValueError(f"未知真值阶段: {truth!r}")
            scores = {}
            for key, value in row.items():
                if key.startswith("probability_"):
                    phase = key.removeprefix("probability_")
                    if phase not in LABELS:
                        raise ValueError(f"未知概率阶段: {phase}")
                    parsed = probability(value, context)
                    if parsed is not None:
                        scores[phase] = parsed
            score = probability(row.get("top_score"), context)
            rows = result.setdefault(video, {})
            if frame in rows:
                raise ValueError(f"重复帧: {video}_{frame:06d}")
            rows[frame] = {"predicted": label, "prediction_truth": truth,
                           "confidence": score, "probabilities": scores}
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{context}: {exc}") from exc
    if not result:
        raise ValueError(f"{path}: 没有预测记录")
    return result


def frame_segments(rows, key, step, end):
    result = []
    for row in rows:
        label, start = row.get(key), row["frame"]
        if label is None:
            continue
        stop = min(start + step, end)
        if result and result[-1]["phase"] == label and result[-1]["end"] == start:
            result[-1]["end"] = stop
        else:
            result.append({"phase": label, "start": start, "end": stop})
    return result


class Catalog:
    def __init__(self, root, experiments, predictions=()):
        self.root = Path(root).resolve(strict=True)
        self.experiments = Path(experiments).resolve()
        index = self.root / "labeled_1fps_v1" / "dataset.json"
        self.dataset = read_json(index) if index.is_file() else {}
        self.metadata = {item["video_id"]: item for item in self.dataset.get("videos", [])}
        self.cases = {p.name[5:7]: p for p in self.root.glob("video[0-9][0-9]_1fps") if p.is_dir()}
        self.sources = {}
        candidates = [(p, False) for p in self.experiments.rglob("validation_predictions*.csv")]
        # Most result.json files are training metrics: do not advertise them as predictions.
        for path in self.experiments.rglob("result.json"):
            try:
                with path.open(encoding="utf-8-sig") as stream:
                    if re.search(r'"predictions"\s*:', stream.read(4096)):
                        candidates.append((path, False))
            except OSError:
                continue
        candidates.extend((Path(p).resolve(strict=True), True) for p in predictions)
        merged = {}
        for path, explicit in candidates:
            path = path.resolve()
            merged[path] = merged.get(path, False) or explicit
        for i, (path, explicit) in enumerate(sorted(merged.items(), key=lambda pair: str(pair[0]))):
            name = str(path.relative_to(self.experiments)) if path.is_relative_to(self.experiments) else str(path)
            self.sources[str(i)] = {"path": path, "explicit": explicit, "name": name}

    def summary(self):
        videos = sorted(set(self.cases) | set(self.metadata) | set(self.dataset.get("official_test_ids_not_processed", [])))
        return {"videos": [{"id": v, "labeled": v in self.metadata, "available": v in self.cases} for v in videos],
                "sources": [{"id": k, "name": v["name"]} for k, v in self.sources.items()],
                "phases": {k: {"name": v[0], "ja": v[1], "color": v[2]} for k, v in PHASES.items()},
                "phase_order": list(PHASES), "phase_source": PHASE_SOURCE,
                "root": str(self.root), "experiments": str(self.experiments)}

    @lru_cache(maxsize=1)
    def samples(self):
        path = self.root / "labeled_1fps_v1" / "samples.csv"
        result = {}
        if not path.is_file():
            if self.metadata:
                raise ValueError(f"缺少有效帧清单: {path}")
            return result
        for line, row in csv_rows(path):
            try:
                video, frame = frame_identity(row["filename"], row["video_id"], row["frame_index"])
                label = row["phase-id"]
                if label not in LABELS:
                    raise ValueError(f"无效分类阶段: {label}")
                image = Path(row["image_path"]).resolve()
                if not image.is_relative_to(self.root) or image.name != row["filename"] + ".png":
                    raise ValueError(f"图片路径与预处理根目录或文件名不符: {image}")
                rows = result.setdefault(video, {})
                if frame in rows:
                    raise ValueError(f"重复帧: {row['filename']}")
                rows[frame] = {"frame": frame, "filename": image.name, "path": image,
                               "truth": label, "excluded": None}
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line}: {exc}") from exc
        return result

    @lru_cache(maxsize=6)
    def video(self, video):
        if video not in self.cases:
            raise ValueError(f"视频 {video} 缺少预处理图片目录")
        case = self.cases[video]
        processing = case / "processing.json"
        meta = self.metadata.get(video) or (read_json(processing) if processing.is_file() else {})
        fps = float(meta.get("fps", 30))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"视频 {video}: 无效 fps")
        step = int(meta.get("sample_every_frames", round(fps)))
        if step <= 0:
            raise ValueError(f"视频 {video}: 无效抽帧间隔")
        directory = case / "basic_challenge" / "data" / video
        labeled = video in self.metadata
        rows = {frame: dict(row) for frame, row in self.samples().get(video, {}).items()} if labeled else {}
        if labeled:
            excluded = directory / f"excluded_{video}.csv"
            if not excluded.is_file():
                raise ValueError(f"缺少排除清单: {excluded}")
            for line, row in csv_rows(excluded):
                _, frame = frame_identity(row["filename"], video)
                if frame in rows or row["phase-id"] not in PHASES:
                    raise ValueError(f"{excluded}:{line}: 重复帧或未知阶段")
                if not row.get("reason"):
                    raise ValueError(f"{excluded}:{line}: 缺少排除原因")
                image = directory / "images" / (row["filename"] + ".png")
                rows[frame] = {"frame": frame, "filename": image.name, "path": image,
                               "truth": row["phase-id"], "excluded": row["reason"]}
        else:
            for image in (directory / "images").glob("*.png"):
                _, frame = frame_identity(image.name, video)
                if frame in rows:
                    raise ValueError(f"视频 {video}: 重复帧 {frame}")
                rows[frame] = {"frame": frame, "filename": image.name, "path": image,
                               "truth": None, "excluded": None}
        if not rows:
            raise ValueError(f"视频 {video}: 没有可查看的抽帧图片记录")
        end = int(meta.get("source_frame_count", max(rows) + step))
        if end <= max(rows) or min(rows) < 0:
            raise ValueError(f"视频 {video}: 帧号超出原视频范围")
        segments = []
        if labeled:
            annotation = case / "source" / f"{video}.csv"
            if not annotation.is_file():
                annotation = Path(meta.get("source_annotation", ""))
            if not annotation.is_file():
                raise ValueError(f"视频 {video}: 缺少原始阶段分段 CSV")
            starts = []
            for line, row in csv_rows(annotation):
                start, label = int(row["start-frame"]), row["phase-id"]
                if label not in PHASES or start < 0 or start > end or (starts and start <= starts[-1]):
                    raise ValueError(f"{annotation}:{line}: 非法阶段或阶段边界")
                starts.append(start)
                segments.append({"phase": label, "start": start, "end": end})
            for a, b in zip(segments, segments[1:]):
                a["end"] = b["start"]
            for frame, row in rows.items():
                pos = bisect_right(starts, frame) - 1
                if pos < 0 or segments[pos]["phase"] != row["truth"]:
                    raise ValueError(f"视频 {video} 帧 {frame}: 清单标签与原始标注冲突")
        return {"id": video, "fps": fps, "step": step, "end": end, "labeled": labeled,
                "rows": rows, "truth_segments": segments,
                "warnings": [] if meta else ["缺少 processing.json，按当前数据约定使用 30 fps、每 30 帧一张。"]}

    @lru_cache(maxsize=2)
    def prediction(self, source):
        if source not in self.sources:
            raise ValueError("未知预测结果")
        item = self.sources[source]
        return read_predictions(item["path"], item["explicit"])

    def view(self, video, source=None):
        base = self.video(video)
        predictions = self.prediction(source).get(video, {}) if source is not None else {}
        unknown = set(predictions) - set(base["rows"])
        if unknown:
            raise ValueError(f"视频 {video}: {len(unknown)} 个预测帧无法对应预处理图片清单，如 {min(unknown)}")
        rows = []
        for frame, row in sorted(base["rows"].items()):
            pred = predictions.get(frame, {})
            reported_truth = pred.get("prediction_truth")
            if reported_truth is not None and row["truth"] is not None and reported_truth != row["truth"]:
                raise ValueError(f"视频 {video} 帧 {frame}: 预测文件真值 {reported_truth} 与数据标注 {row['truth']} 冲突")
            rows.append({k: v for k, v in row.items() if k != "path"} | {
                "predicted": pred.get("predicted"), "confidence": pred.get("confidence"),
                "probabilities": pred.get("probabilities", {})})
        return {k: v for k, v in base.items() if k != "rows"} | {
            "rows": rows, "source": source, "prediction_count": len(predictions),
            "prediction_segments": frame_segments(rows, "predicted", base["step"], base["end"])}

    def image_path(self, video, frame):
        row = self.video(video)["rows"].get(frame)
        if row is None:
            raise ValueError("未登记的图片")
        path = row["path"].resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() != ".png":
            raise ValueError("图片不在允许的预处理目录内")
        if not path.is_file():
            raise FileNotFoundError(f"图片缺失: {row['filename']}")
        return path


def make_handler(catalog):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            # Avoid printing one line per image while browsing or playing.
            if len(args) > 1 and str(args[1]) not in {"200", "304"}:
                super().log_message(format, *args)

        def send_body(self, data, content_type, status=200):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, value, status=200):
            self.send_body(json.dumps(value, ensure_ascii=False, allow_nan=False).encode(),
                           "application/json; charset=utf-8", status)

        def do_GET(self):
            try:
                # Local-only service: reject cross-origin requests and DNS rebinding hosts.
                host = self.headers.get("Host", "").split(":")[0]
                if host not in {"127.0.0.1", "localhost"}:
                    self.send_json({"error": "仅支持本机访问"}, 403)
                    return
                origin = self.headers.get("Origin")
                if origin and origin != f"http://{self.headers.get('Host')}":
                    self.send_json({"error": "不允许跨来源访问"}, 403)
                    return
                request = urlsplit(self.path)
                query = parse_qs(request.query, keep_blank_values=True)
                def arg(name, default=None):
                    return query.get(name, [default])[0]
                if request.path == "/":
                    self.send_body(Path(__file__).with_name("viewer.html").read_bytes(), "text/html; charset=utf-8")
                elif request.path == "/api/catalog":
                    self.send_json(catalog.summary())
                elif request.path == "/api/prediction":
                    result = catalog.prediction(arg("source"))
                    self.send_json({"videos": [{"id": v, "frames": len(rows), "available": v in catalog.cases}
                                               for v, rows in sorted(result.items())]})
                elif request.path == "/api/video":
                    self.send_json(catalog.view(arg("video"), arg("source")))
                elif request.path == "/api/image":
                    path = catalog.image_path(arg("video"), int(arg("frame", "-1")))
                    with path.open("rb") as stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "image/png")
                        self.send_header("Content-Length", str(path.stat().st_size))
                        self.send_header("Cache-Control", "private, max-age=3600")
                        self.send_header("X-Content-Type-Options", "nosniff")
                        self.end_headers()
                        shutil.copyfileobj(stream, self.wfile)
                else:
                    self.send_json({"error": "未找到接口"}, 404)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.send_json({"error": str(exc)}, 400)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description="只读 JSCAS 手术阶段数据查看器（无需 GPU）")
    parser.add_argument("--data-root", type=Path, default=Path(r"H:\dataset\JSCAS_preprocessed"))
    parser.add_argument("--experiments-root", type=Path, default=Path(r"H:\dataset\JSCAS_experiments"))
    parser.add_argument("--prediction", type=Path, action="append", default=[])
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)
    try:
        catalog = Catalog(args.data_root, args.experiments_root, args.prediction)
        server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(catalog))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"无法启动查看器: {exc}\n")
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"JSCAS 查看器: {url}\n数据只读，Ctrl+C 退出。", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
