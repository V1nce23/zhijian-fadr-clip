"""Offline pixel comparison against pinned official DeepfakeBench preprocessing.

The reference file must be downloaded explicitly from the recorded revision.
Only its alignment functions and the release's resize helper are evaluated.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from skimage import transform as trans

from app.detectors.face_media import FaceMedia
from app.detectors.faces import _align_crop
from app.detectors.fadr_clip.adapter import FADRClipDetector
from app.detectors.catalog import detector_specs

REFERENCE_SHA256 = "d85eae2ee2e3e6d05d1393e5d3f837333446539e2c728e0d67f456928770d4d5"


def function(tree, name):
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = args.reference.read_bytes()
    assert hashlib.sha256(reference).hexdigest() == REFERENCE_SHA256
    tree = ast.parse(reference)
    scope = {"np": np, "cv2": cv2, "trans": trans}
    # Execute only these standalone mathematical / crop functions, never script startup.
    module = ast.Module(body=[function(tree, name) for name in
                            ("get_keypts", "extract_aligned_face_dlib", "img_align_crop")], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "pinned-deepfakebench-functions", "exec"), scope)
    rgb = np.random.default_rng(123).integers(0, 256, (480, 640, 3), dtype=np.uint8)
    landmarks = np.array([[220, 165], [350, 172], [288, 238], [236, 300], [335, 305]])
    expected, _ = scope["img_align_crop"](rgb, landmarks, (256, 256))
    actual = _align_crop(rgb, landmarks)
    assert np.array_equal(actual, expected)
    report = {"synthetic_crop_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
              "synthetic_crop_pixel_max_error": int(np.abs(actual.astype(int)-expected.astype(int)).max())}

    import dlib
    scope["face_utils"] = SimpleNamespace(shape_to_np=lambda shape: np.array(
        [[shape.part(i).x, shape.part(i).y] for i in range(shape.num_parts)]))
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(str(args.models / "face/shape_predictor_81_face_landmarks.dat"))
    original_bgr = cv2.imread(str(args.image))
    official_bgr, _, _ = scope["extract_aligned_face_dlib"](detector, predictor, original_bgr)
    assert official_bgr is not None, "Reference found no valid face in the chosen sample"
    expected_rgb = cv2.cvtColor(official_bgr, cv2.COLOR_BGR2RGB)
    actual_samples, _, _ = FaceMedia(args.models).extract(args.image, "image")
    assert np.array_equal(actual_samples[0].image, expected_rgb)
    report["real_face_crop_pixel_max_error"] = 0
    cv2.imwrite(str(args.output.parent / "reference-aligned.png"), official_bgr)

    resize_tree = ast.parse((args.release / "dataset/albu.py").read_text(encoding="utf-8"))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function(resize_tree, "isotropically_resize_image")],
         type_ignores=[])), "original-release-resize", "exec"), scope)
    resized = scope["isotropically_resize_image"](expected_rgb, 224)
    # The input is a square 256 face crop, so pad and center crop are identities.
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]
    expected_tensor = torch.from_numpy(resized.copy()).permute(2, 0, 1).float() / 255
    expected_tensor = (expected_tensor - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]
    deployed = FADRClipDetector(detector_specs("image")[0], args.models)
    deployed.config = {"mean": mean, "std": std, "clip_input_resolution": 224}
    actual_tensor = deployed._input_tensor(actual_samples[0].image)
    assert torch.equal(expected_tensor, actual_tensor)
    report["model_input_tensor_max_error"] = float((expected_tensor-actual_tensor).abs().max())
    report["model_input_shape"] = list(actual_tensor.shape)
    report["opencv_version"] = cv2.__version__
    report["passed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
