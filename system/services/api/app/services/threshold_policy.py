"""Versioned decision presets; scores and model inference remain unchanged."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ThresholdMode:
    mode: str
    label: str
    description: str
    video_threshold: float
    frame_threshold: float


MODES = (
    ThresholdMode("balanced", "均衡", "兼顾误报与漏报，适合日常检测", 0.67, 0.85),
    ThresholdMode("low_false_positive", "低误报", "减少将真实素材报为伪造，可能漏掉更多伪造", 0.93, 0.99),
    ThresholdMode("low_false_negative", "低漏报", "优先筛出可疑素材，可能产生更多误报", 0.50, 0.50),
)
CUSTOM_MODE = ThresholdMode("custom", "自定义", "自行设置视频与图片／单帧的判断阈值", 0.67, 0.85)


def make_policy(mode: str, media_type: str, custom_thresholds: dict | None = None) -> dict:
    preset = CUSTOM_MODE if mode == "custom" else next(item for item in MODES if item.mode == mode)
    if mode == "custom":
        if custom_thresholds is None:
            raise ValueError("自定义模式缺少阈值")
        video, frame = custom_thresholds["video_threshold"], custom_thresholds["frame_threshold"]
    else:
        video, frame = preset.video_threshold, preset.frame_threshold
    return {
        "version": "fadr-threshold-v1", "mode": preset.mode, "label": preset.label,
        "operator": ">", "threshold": video if media_type == "video" else frame,
        "video_threshold": video, "frame_threshold": frame,
        "score_unit": "video_mean" if media_type == "video" else "image",
        "selection_status": "user_defined" if mode == "custom" else "exploratory_test_set_candidates_not_validated",
    }


def frame_thresholds(policy: dict) -> dict:
    # The gallery displays individual frames; never compare a single frame to
    # the threshold selected for a video mean.
    threshold = policy["frame_threshold"]
    return {"authentic_max": threshold, "manipulated_min": threshold, "operator": ">"}


def policy_verdict(score: float, policy: dict) -> str:
    # Strict > matches the original threshold analysis, including exact ties.
    return "likely_manipulated" if score > policy["threshold"] else "likely_authentic"


def policy_summary(policy: dict, verdict: str) -> str:
    unit = "有效人脸帧的平均分数" if policy["score_unit"] == "video_mean" else "人脸检测分数"
    comparison = "超过" if verdict == "likely_manipulated" else "未超过"
    number = f"{policy['threshold']:.6f}".rstrip("0").rstrip(".")
    whole, _, fraction = number.partition(".")
    threshold = whole + "." + fraction.ljust(2, "0")
    return f"{unit}{comparison}{policy['label']}模式阈值 {threshold}。"
