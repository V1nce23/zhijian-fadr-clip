from __future__ import annotations

import numpy as np
import cv2
from skimage import transform as trans


# Five-point alignment targets, same scheme as DeepfakeBench preprocessing
# (InsightFace-style destination coordinates scaled to the output size).
def _align_crop(image: np.ndarray, landmarks: np.ndarray, size: int = 256) -> np.ndarray:
    dst = np.array(
        [
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041],
        ],
        dtype=np.float32,
    )
    dst[:, 0] += 8.0
    dst = dst * size / 112.0
    margin_rate = 1.3 - 1
    x_margin = size * margin_rate / 2.0
    y_margin = size * margin_rate / 2.0
    dst[:, 0] += x_margin
    dst[:, 1] += y_margin
    dst[:, 0] *= size / (size + 2 * x_margin)
    dst[:, 1] *= size / (size + 2 * y_margin)

    tform = trans.SimilarityTransform()
    tform.estimate(landmarks.astype(np.float32), dst)
    # Match DeepfakeBench interpolation and rounding exactly.
    aligned = cv2.warpAffine(image, tform.params[0:2, :], (size, size))
    return cv2.resize(aligned, (size, size))


class DlibFaceAligner:
    """Detects the largest face and aligns it with the DeepfakeBench 5-point scheme."""

    def __init__(self, predictor_path) -> None:
        import dlib

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(str(predictor_path))

    def keypoints(self, image: np.ndarray):
        import dlib

        faces = self._detector(image, 1)
        if not faces:
            return None
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self._predictor(image, face)
        indices = (37, 44, 30, 49, 55)
        return np.array([[shape.part(index).x, shape.part(index).y] for index in indices])

    def align(self, image: np.ndarray, size: int = 256) -> np.ndarray | None:
        landmarks = self.keypoints(image)
        if landmarks is None:
            return None
        return _align_crop(image, landmarks, size=size)

    def align_with_region(self, image: np.ndarray, size: int = 256):
        faces = self._detector(image, 1)
        if not faces:
            return None
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        shape = self._predictor(image, face)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in (37, 44, 30, 49, 55)])
        height, width = image.shape[:2]
        left, top = max(face.left(), 0), max(face.top(), 0)
        right, bottom = min(face.right() + 1, width), min(face.bottom() + 1, height)
        region = {"x": left / width, "y": top / height,
                  "width": (right - left) / width, "height": (bottom - top) / height}
        aligned = _align_crop(image, landmarks, size=size)
        # Official preprocessing checks for a detectable face in the saved BGR crop.
        if not self._detector(cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR), 1):
            return None
        return aligned, region
