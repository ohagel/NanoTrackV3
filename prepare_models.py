"""One-time V3 download/preparation; requires onnx and onnxsim, not at runtime."""
from pathlib import Path
import hashlib
import json
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from onnxsim import simplify

COMMIT = "248663fde6bf7c40190cf10ee396d5662919ecd3"
BASE = f"https://raw.githubusercontent.com/HonglinChu/SiamTrackers/{COMMIT}/NanoTrack/models/nanotrackv3"


def main():
    root = Path(__file__).resolve().parent / "models"
    root.mkdir(exist_ok=True)
    manifest = {"upstream_commit": COMMIT, "models": {}}
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    rng = np.random.default_rng(42)
    for kind in ("backbone", "head"):
        name = f"nanotrack_{kind}"
        raw = root / f"{name}.onnx"
        if not raw.exists():
            urllib.request.urlretrieve(f"{BASE}/{name}.onnx", raw)
        model = onnx.load(raw)
        if kind == "backbone":
            # Fully convolutional graph: only spatial interface metadata changes.
            for value, labels in ((model.graph.input[0], ("height", "width")),
                                  (model.graph.output[0], ("feature_h", "feature_w"))):
                for dim, label in zip(value.type.tensor_type.shape.dim[2:], labels):
                    dim.dim_param = label
        baseline = ort.InferenceSession(model.SerializeToString(), options,
                                        providers=["CPUExecutionProvider"])
        result, checked = simplify(model)
        if not checked:
            raise RuntimeError("ONNX simplifier validation failed")
        onnx.checker.check_model(result)
        prepared = ort.InferenceSession(result.SerializeToString(), options,
                                        providers=["CPUExecutionProvider"])
        shapes = [[(1, 3, s, s)] for s in (127, 255)] if kind == "backbone" else [[(1, 96, 8, 8), (1, 96, 16, 16)]]
        for case in shapes:
            feeds = {v.name: rng.uniform(0, 255 if kind == "backbone" else 1, shape).astype(np.float32)
                     for v, shape in zip(baseline.get_inputs(), case)}
            for a, b in zip(baseline.run(None, feeds), prepared.run(None, feeds)):
                np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)
        destination = root / f"{name}_sim.onnx"
        onnx.save(result, destination)
        manifest["models"][destination.name] = {
            "source": f"{BASE}/{name}.onnx",
            "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "prepared_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "change": "dynamic spatial metadata + simplification" if kind == "backbone" else "simplification",
        }
        print(f"Validated {destination.name}")
    (root / "provenance.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
