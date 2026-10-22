from collections import Counter
from dataclasses import dataclass

from app.detectors.base import DetectionResult
from app.models import Verdict


def aggregate_verdict(verdicts: list[Verdict]) -> Verdict:
    ordered = Counter(verdicts).most_common()
    if not ordered:
        return Verdict.UNCERTAIN
    if len(ordered) == 1 or ordered[0][1] > ordered[1][1]:
        return ordered[0][0]
    return Verdict.UNCERTAIN


@dataclass(frozen=True)
class AnalysisSummary:
    score: float
    verdict: Verdict
    summary: str


def aggregate_results(results: list[DetectionResult]) -> AnalysisSummary:
    if not results:
        raise ValueError("没有成功的模型结果")
    if len(results) == 1:
        result = results[0]
        verdict = Verdict(result.verdict)
        descriptions = {
            Verdict.LIKELY_MANIPULATED: "FADR-CLIP 检测到疑似合成或篡改特征。",
            Verdict.LIKELY_AUTHENTIC: "FADR-CLIP 未发现明显合成或篡改特征。",
            Verdict.UNCERTAIN: "FADR-CLIP 分数处于不确定区间，建议人工复核。",
        }
        return AnalysisSummary(round(result.score, 6), verdict, descriptions[verdict])
    verdicts = [Verdict(result.verdict) for result in results]
    verdict = aggregate_verdict(verdicts)
    counts = Counter(verdicts)
    if verdict == Verdict.LIKELY_MANIPULATED:
        summary = f"{counts[verdict]} 个检查器判定为合成或篡改。"
    elif verdict == Verdict.LIKELY_AUTHENTIC:
        summary = f"{counts[verdict]} 个检查器未发现明显合成或篡改特征。"
    else:
        summary = "检查器结论不一致或证据不足，建议人工复核。"
    return AnalysisSummary(
        score=round(sum(result.score for result in results) / len(results), 6),
        verdict=verdict,
        summary=summary,
    )
