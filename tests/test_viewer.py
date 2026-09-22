"""Small offline checks; run with python -m unittest discover -s tests -p test_viewer.py."""

import base64
import csv
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import threading
import unittest

from tools.viewer.server import Catalog, frame_segments, make_handler, read_predictions


PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=")


def write_csv(path, columns, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


class ViewerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "data"
        self.experiments = Path(self.temporary.name) / "experiments"
        self.experiments.mkdir()
        self.index = self.root / "labeled_1fps_v1"
        self.index.mkdir(parents=True)
        self.case = self.root / "video01_1fps"
        self.directory = self.case / "basic_challenge" / "data" / "01"
        self.images = self.directory / "images"
        self.images.mkdir(parents=True)
        for frame in [0, 30, 60, 90, 120]:
            (self.images / f"01_{frame:06d}.png").write_bytes(PNG)
        (self.index / "dataset.json").write_text(json.dumps({
            "videos": [{"video_id": "01", "fps": 30, "source_frame_count": 150}],
            "official_test_ids_not_processed": ["02", "07", "11", "15"],
        }), encoding="utf-8")
        write_csv(self.index / "samples.csv", ["video_id", "filename", "frame_index", "time_seconds",
                                               "phase-id", "image_path", "image_sha256"],
                  [["01", f"01_{f:06d}", f, f / 30, label, self.images / f"01_{f:06d}.png", "0" * 64]
                   for f, label in [(30, "0"), (60, "1"), (120, "0")]])
        write_csv(self.directory / "excluded_01.csv", ["filename", "phase-id", "reason", "max_pixel"],
                  [["01_000000", "x", "phase_x", 0], ["01_000090", "1", "all_pixels_at_most_1", 1]])
        write_csv(self.case / "source" / "01.csv", ["start-frame", "phase-id"],
                  [[0, "x"], [15, "0"], [60, "1"], [100, "0"], [150, "e"]])

    def catalog(self, *extra):
        catalog = Catalog(self.root, self.experiments, extra)
        # The decorators are deliberately bounded; release temporary catalog instances after each test.
        self.addCleanup(catalog.samples.cache_clear)
        self.addCleanup(catalog.video.cache_clear)
        self.addCleanup(catalog.prediction.cache_clear)
        return catalog

    def test_annotation_boundaries_repeated_phases_and_exclusions(self):
        catalog = self.catalog()
        view = catalog.view("01")
        self.assertEqual([(r["frame"], r["truth"]) for r in view["rows"]],
                         [(0, "x"), (30, "0"), (60, "1"), (90, "1"), (120, "0")])
        self.assertEqual(view["truth_segments"][1], {"phase": "0", "start": 15, "end": 60})
        self.assertEqual(view["truth_segments"][3]["phase"], "0")
        self.assertEqual(view["rows"][3]["excluded"], "all_pixels_at_most_1")
        self.assertEqual(sum(r["excluded"] is None for r in view["rows"]), 3)
        self.assertIsNone(view["rows"][1]["predicted"])
        self.assertFalse(next(v for v in catalog.summary()["videos"] if v["id"] == "02")["available"])

    def test_old_and_new_predictions_join_by_identity_not_order(self):
        old = self.experiments / "old" / "validation_predictions.csv"
        new = self.experiments / "new" / "validation_predictions.csv"
        write_csv(old, ["video_id", "filename", "phase_id", "predicted_phase_id", "top_score"],
                  [["01", "01_000060.png", "1", "0", 0.8], ["01", "01_000030.png", "0", "0", 0.9]])
        write_csv(new, ["fold", "video_id", "filename", "frame_index", "phase_id", "predicted_phase_id",
                        "top_score", "probability_0", "probability_1"],
                  [[1, "01", "01_000030.png", 30, "0", "0", 0.9, 0.9, 0.1]])
        catalog = self.catalog()
        for source, item in catalog.sources.items():
            view = catalog.view("01", source)
            self.assertEqual(view["rows"][1]["predicted"], "0")
            self.assertEqual(view["rows"][1]["confidence"], 0.9)
            if item["path"] == new:
                self.assertEqual(view["rows"][1]["probabilities"], {"0": 0.9, "1": 0.1})

    def test_official_predictions_are_explicit_and_no_fabricated_scores(self):
        path = self.experiments / "submission.csv"
        write_csv(path, ["filename", "phase-id"], [["01_000030", "82"]])
        with self.assertRaisesRegex(ValueError, "显式指定"):
            read_predictions(path)
        self.assertEqual(self.catalog().summary()["sources"], [])
        view = self.catalog(path).view("01", "0")
        self.assertEqual(view["rows"][1]["predicted"], "82")
        self.assertIsNone(view["rows"][1]["confidence"])
        self.assertEqual(view["rows"][1]["truth"], "0")

    def test_predict_head_json_and_unlabeled_images(self):
        images = self.root / "video02_1fps" / "basic_challenge" / "data" / "02" / "images"
        images.mkdir(parents=True)
        image = images / "02_000030.png"
        image.write_bytes(PNG)
        (images / "02_000060.png").write_bytes(PNG)
        path = self.experiments / "result.json"
        path.write_text(json.dumps({"predictions": [{"image": str(image), "phase_id": "83",
                                                     "scores": {"83": 0.7, "82": 0.3}}]}), encoding="utf-8")
        view = self.catalog().view("02", "0")
        self.assertFalse(view["labeled"])
        self.assertIsNone(view["rows"][0]["truth"])
        self.assertEqual(view["rows"][0]["predicted"], "83")
        self.assertEqual(view["rows"][0]["confidence"], 0.7)
        self.assertIsNone(view["rows"][1]["predicted"])
        self.assertEqual(view["truth_segments"], [])
        self.assertEqual(view["prediction_segments"], [{"phase": "83", "start": 30, "end": 60}])

    def test_prediction_segments_preserve_missing_seconds(self):
        rows = [{"frame": f, "predicted": label} for f, label in [(0, "0"), (30, "0"), (60, None), (90, "0")]]
        self.assertEqual(frame_segments(rows, "predicted", 30, 115),
                         [{"phase": "0", "start": 0, "end": 60}, {"phase": "0", "start": 90, "end": 115}])

    def test_bad_prediction_inputs_have_actionable_errors(self):
        path = self.experiments / "validation_predictions.csv"
        fields = ["video_id", "filename", "frame_index", "predicted_phase_id", "top_score"]
        good = ["01", "01_000030.png", "30", "0", "0.5"]
        cases = [(good + ["extra"], "列数"), (["01", "../01_000030.png", "30", "0", "0.5"], "文件名"),
                 (["02", *good[1:]], "身份冲突"), ([*good[:2], "60", "0", "0.5"], "身份冲突"),
                 ([*good[:3], "8", "0.5"], "未知预测阶段"), ([*good[:4], "NaN"], "有限数值")]
        for row, error in cases:
            with self.subTest(error=error):
                write_csv(path, fields, [row])
                with self.assertRaisesRegex(ValueError, error):
                    read_predictions(path)
        write_csv(path, fields, [good, good])
        with self.assertRaisesRegex(ValueError, "重复帧"):
            read_predictions(path)
        result = self.experiments / "result.json"
        result.write_text('{"predictions":[null]}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "必须是对象"):
            read_predictions(result)

    def test_truth_conflicts_and_unmatched_prediction_frames(self):
        path = self.experiments / "validation_predictions.csv"
        write_csv(path, ["filename", "phase_id", "predicted_phase_id"], [["01_000030", "1", "0"]])
        with self.assertRaisesRegex(ValueError, "真值.*冲突"):
            self.catalog().view("01", "0")
        write_csv(path, ["filename", "predicted_phase_id"], [["01_000031", "0"]])
        with self.assertRaisesRegex(ValueError, "无法对应"):
            self.catalog().view("01", "0")
        write_csv(self.case / "source" / "01.csv", ["start-frame", "phase-id"], [[0, "x"], [15, "1"], [150, "e"]])
        with self.assertRaisesRegex(ValueError, "原始标注冲突"):
            self.catalog().view("01")

    def test_image_registration_missing_files_and_path_escape(self):
        catalog = self.catalog()
        with self.assertRaisesRegex(ValueError, "未登记"):
            catalog.image_path("01", 31)
        with self.assertRaisesRegex(ValueError, "缺少预处理"):
            catalog.image_path("../../", 0)
        (self.images / "01_000030.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "图片缺失"):
            catalog.image_path("01", 30)
        # A poisoned manifest must never expose a file outside the configured root.
        manifest = self.index / "samples.csv"
        manifest.write_text(manifest.read_text(encoding="utf-8-sig").replace(str(self.images), str(self.root.parent)), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "图片路径"):
            self.catalog().view("01")

    def test_http_read_only_routes(self):
        server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(self.catalog()))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
            for route, expected in [("/", 200), ("/api/catalog", 200), ("/api/video?video=01", 200),
                                    ("/api/image?video=01&frame=30", 200), ("/api/image?video=01&frame=31", 400),
                                    ("/api/image?path=../../secret", 400), ("/../server.py", 404)]:
                with self.subTest(route=route):
                    client.request("GET", route)
                    response = client.getresponse()
                    self.assertEqual(response.status, expected)
                    body = response.read()
                    if route == "/api/image?video=01&frame=30":
                        self.assertEqual(body, PNG)
            client.request("GET", "/api/catalog", headers={"Origin": "https://unrelated.example"})
            response = client.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            client.request("GET", "/api/catalog", headers={"Host": "unrelated.example"})
            response = client.getresponse()
            self.assertEqual(response.status, 403)
            response.read()
            client.request("POST", "/api/video?video=01")
            response = client.getresponse()
            self.assertEqual(response.status, 501)
            response.read()
            client.close()
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
