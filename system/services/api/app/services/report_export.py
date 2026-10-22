"""Portable reports made solely from authorized existing records and cached assets."""

import base64
import csv
import hashlib
import html
import io
import json
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from app.detectors.base import verdict_from_score
from app.schemas import AnalysisPublic
from app.services.evidence_gallery import gallery_prefix
from app.services.routing_evidence import report_frames, routing_for_frames, validated_routing

VERDICTS = {"likely_manipulated": "疑似伪造", "likely_authentic": "疑似真实", "uncertain": "不确定"}
STATUSES = {"completed": "已完成", "partially_completed": "部分完成", "failed": "失败", "cancelled": "已取消",
            "succeeded": "已完成", "queued": "等待检测", "running": "检测中", "preprocessing": "预处理中",
            "aggregating": "正在汇总", "created": "已创建", "timed_out": "超时"}
FRAME_STATUSES = {"scored": "已检测", "no_face": "无人脸，已跳过", "pending": "待处理", "sampled": "已读取，待对齐", "ready": "已对齐，待检测"}
NOTICE = "检测分数不是已校准的伪造概率。路由权重表示各层特征的融合比例，不代表伪造概率或检测准确率。注意力颜色仅表示当前帧内相对关注，不是伪造像素定位或分数的因果解释。"
BASIC_NOTICE = "检测分数不是已校准的伪造概率。注意力颜色仅表示当前帧内相对关注，不是伪造像素定位或分数的因果解释。"


def frame_decision(score, thresholds):
    if score is None:
        return None
    low, high = thresholds.get("authentic_max"), thresholds.get("manipulated_min")
    if high is None or low is None:
        return None
    if thresholds.get("operator") == ">":
        return "likely_manipulated" if score > high else "likely_authentic"
    return verdict_from_score(score, authentic_threshold=low, manipulated_threshold=high)


def snapshot(analysis, storage, *, include_technical=True):
    record = AnalysisPublic.model_validate(analysis).model_dump(mode="json")
    # Hash the original public report before attaching supplementary export data.
    digest = hashlib.sha256(json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    runs = []
    for run in analysis.detector_runs:
        evidence = run.evidence or {}
        frames = report_frames(run)
        routes = routing_for_frames(analysis.id, run, storage if analysis.upload.status != "deleted" else None, frames) if include_technical else {}
        thresholds = evidence.get("thresholds") or {}
        runs.append({"run_id": str(run.id), "routing_mean": validated_routing(evidence.get("routing_weights"), evidence.get("routing_layers")),
            "thresholds": thresholds,
            "recorded_sampled_frames": evidence.get("sampled_frames"),
            "frames": [{"frame_index": item["frame_index"], "frame_number": item["frame_index"] + 1,
                "timestamp_seconds": item.get("timestamp_seconds"), "status": item.get("status", "scored"),
                "score": item.get("score"), "verdict": frame_decision(item.get("score"), thresholds),
                "routing": routes.get(item["frame_index"])} for item in frames]})
    return {"schema_version": "zhijian-report-v1", "exported_at": datetime.now(timezone.utc).isoformat(),
        "record_sha256": digest, "record": record, "supplementary": runs,
        "technical_details_included": include_technical, "interpretation": NOTICE if include_technical else BASIC_NOTICE}


def _safe_csv(value):
    if value is None:
        return ""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def csv_report(data):
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    headers = ["记录ID", "文件名", "媒体类型", "模型", "模型版本", "预处理版本", "任务状态", "总体分数（0–1）", "总体结论",
        "画面索引（从0起）", "画面序号（从1起）", "时间（秒）", "帧状态", "帧分数（0–1）", "帧结论", "阈值比较符", "单帧真实上界", "单帧伪造下界", "总体判断阈值",
        "L6路由权重", "L12路由权重", "L18路由权重", "L24路由权重", "逐帧路由来源"]
    columns = list(range(len(headers))) if data.get("technical_details_included", True) else [index for index in range(19) if index not in (4, 5)]
    writer.writerow([headers[index] for index in columns])
    record = data["record"]
    for run, extra in zip(record["detector_runs"], data["supplementary"]):
        thresholds = extra["thresholds"]
        for frame in extra["frames"] or [{}]:
            weights = (frame.get("routing") or {}).get("weights", [None] * 4)
            origin = (frame.get("routing") or {}).get("origin")
            row = [record["id"], record["upload"]["original_name"], record["upload"]["media_type"], run["detector_id"], run["model_version"], run["preprocessing_version"],
                STATUSES.get(record["status"], record["status"]), record["score"], VERDICTS.get(record["verdict"], "未生成"),
                frame.get("frame_index"), frame.get("frame_number"), frame.get("timestamp_seconds"), FRAME_STATUSES.get(frame.get("status"), "未记录逐帧数据"), frame.get("score"),
                VERDICTS.get(frame.get("verdict"), ""), thresholds.get("operator", ">= / <="), thresholds.get("authentic_max"), thresholds.get("manipulated_min"),
                (record.get("threshold_policy") or {}).get("threshold"), *weights,
                "原检测记录" if origin == "original_detection" else "补充生成" if origin == "supplementary_replay" else "未记录"]
            writer.writerow([_safe_csv(row[index]) for index in columns])
    return stream.getvalue().encode("utf-8-sig")


def _text(value):
    return html.escape(str(value if value is not None else "未记录"), quote=True)


def _number(value):
    return "未记录" if value is None else f"{value:.6f}".rstrip("0").rstrip(".")


def _score(value):
    return "—" if value is None else f"{value * 100:.2f}%"


def _date(value):
    if not value:
        return "未记录"
    moment = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S UTC+8")


def _routing_chart(routing):
    if not routing:
        return '<p class="muted">本记录未保存可用路由权重，不推测缺失数据。</p>'
    return '<div class="routes">' + "".join(f'<div class="route"><span>L{layer}</span><div class="track"><i style="width:{weight * 100:.6f}%"></i></div><b>{weight * 100:.2f}%</b></div>'
        for layer, weight in zip(routing["layers"], routing["weights"])) + '</div>'


def _image_data(storage, key):
    try:
        if not storage.exists(key):
            return None
        with tempfile.TemporaryDirectory(prefix="zhijian-report-") as directory:
            path = Path(directory) / "cached.png"
            storage.download(key, path)
            with path.open("rb") as stream:
                content = stream.read(512 * 1024 + 1)
        if len(content) > 512 * 1024 or not content.startswith(b"\x89PNG\r\n\x1a\n"):
            return None
        return "data:image/png;base64," + base64.b64encode(content).decode("ascii")
    except Exception:
        return None


def html_report(data, storage):
    technical = data.get("technical_details_included", True)
    record = data["record"]
    upload = record["upload"]
    policy = record.get("threshold_policy")
    media_label = {"video": "视频", "image": "图片", "audio": "历史音频"}.get(upload["media_type"], upload["media_type"])
    selected_device = {"cpu": "CPU", "cuda": "GPU"}.get(record["requested_device"])
    executed_device = {"cpu": "CPU", "cuda": "GPU"}.get(record["executed_device"])
    sections = []
    for run, extra in zip(record["detector_runs"], data["supplementary"]):
        rows = []
        for frame in extra["frames"]:
            routing = frame.get("routing")
            cells = [frame["frame_number"], _number(frame["timestamp_seconds"]), FRAME_STATUSES.get(frame["status"], frame["status"]),
                _score(frame["score"]), VERDICTS.get(frame["verdict"], "—"),
                *([f"{value * 100:.2f}%" for value in routing["weights"]] if routing else ["—"] * 4),
                "补充生成" if routing and routing["origin"] == "supplementary_replay" else "原检测" if routing else "未记录"]
            if not technical:
                cells = cells[:5]
            rows.append("<tr>" + "".join(f"<td>{_text(cell)}</td>" for cell in cells) + "</tr>")
        thresholds = extra["thresholds"]
        if thresholds.get("operator") == ">":
            decision = f"图片／单帧分数 > {_number(thresholds.get('manipulated_min'))} 判为疑似伪造，否则为疑似真实。"
        elif thresholds.get("authentic_max") is not None and thresholds.get("manipulated_min") is not None:
            decision = f"历史规则：单帧分数 ≥ {_number(thresholds.get('manipulated_min'))} 为疑似伪造，≤ {_number(thresholds.get('authentic_max'))} 为疑似真实，其间为不确定。"
        else:
            decision = "单帧阈值未记录，保留原报告结论，不推算单帧结论。"
        examples = []
        attention_assets = {}
        try:
            from app.services.attention_gallery import read_manifest
            if upload["status"] != "deleted":
                attention_assets = {item["frame_index"]: item for item in read_manifest(storage, record["id"], run["id"])["frames"]}
        except Exception:
            pass
        # Reuse at most three actual cached faces, sorted by original score.
        if upload["status"] != "deleted":
            for frame in sorted((item for item in extra["frames"] if item["score"] is not None), key=lambda item: item["score"], reverse=True)[:3]:
                face = _image_data(storage, f"{gallery_prefix(record['id'], run['id'])}frame-{frame['frame_index']}.png")
                if not face:
                    continue
                from app.services.attention_gallery import prefix
                heatmap = _image_data(storage, f"{prefix(record['id'], run['id'])}frame-{frame['frame_index']}.png") if frame['frame_index'] in attention_assets else None
                pair = f'<figure><img src="{face}" alt="对齐后检测人脸"><figcaption>检测人脸</figcaption></figure>'
                if heatmap:
                    origin = "补充生成" if attention_assets[frame['frame_index']].get("origin") == "supplementary_replay" else "原检测"
                    pair += f'<figure><div class="overlay"><img src="{face}" alt=""><img class="heatmap" src="{heatmap}" alt="路由加权空间注意力"></div><figcaption>路由加权空间注意力 · {origin} · 55% 叠加</figcaption></figure>'
                frame_route = _routing_chart(frame.get("routing")) if technical else ""
                examples.append(f'<article class="example"><h4>{"第 " + str(frame["frame_number"]) + " 帧" if upload["media_type"] == "video" else "检测人脸"} · {_score(frame["score"])}</h4><div class="pair">{pair}</div>{frame_route}</article>')
        model_label = "FADR-CLIP" if run["detector_id"] == "fadr-clip" else run["detector_id"]
        mean_note = '<p class="muted">权重来自原检测记录，四层之和约为 100%。</p>' if extra["routing_mean"] else ''
        technical_section = (f'<h3>技术详情</h3><dl><dt>模型版本</dt><dd>{_text(run["model_version"])}</dd><dt>预处理版本</dt><dd>{_text(run["preprocessing_version"])}</dd></dl><h3>{"有效人脸帧平均路由权重" if upload["media_type"] == "video" else "图片路由权重"}</h3>{_routing_chart(extra["routing_mean"])}{mean_note}') if technical else ""
        columns = ['帧号', '时间（秒）', '状态', '分数', '结论'] + (['L6', 'L12', 'L18', 'L24', '路由来源'] if technical else [])
        sections.append(f'<section><h2>{_text(model_label)} · 模型结果</h2><p>{_text(VERDICTS.get(run["verdict"], "未生成结论"))} · {_score(run["score"])} · {_text(STATUSES.get(run["status"], run["status"]))}</p>{technical_section}<p>{_text(decision)}</p>'
            + (f'<p class="muted">模型记录：{_text(run["error"])}</p>' if run.get("error") else "")
            + ('<h3>分数最高的有效人脸样本（最多 3 张，使用已缓存素材）</h3><div class="examples">' + "".join(examples) + '</div>' if examples else '<p class="muted">未附加人脸图片：本次导出只读取现有缓存，不重新抽帧或检测。</p>')
            + (f'<h3>完整逐帧记录</h3><p class="muted">记录抽样数：{_text(extra["recorded_sampled_frames"])}；已保存明细：{len(extra["frames"])}。帧号从 1 起计；无人脸／未完成帧的分数留空，不计作 0 分。时间戳为近似值。</p>' if upload["media_type"] == "video" else '<h3>图片检测明细</h3>')
            + '<div class="table-wrap"><table><thead><tr>'
            + ''.join(f'<th>{label}</th>' for label in columns)
            + '</tr></thead><tbody>' + ''.join(rows) + '</tbody></table></div>'
            + ('<p class="muted">此记录没有保存逐帧明细。</p>' if not rows else '') + '</section>')
    policy_text = (f'{_text(policy["label"])} · {"视频均值" if policy["score_unit"] == "video_mean" else "图片分数"} &gt; {_number(policy["threshold"])}'
                   if policy else '历史模式，沿用原检测记录的结论与阈值')
    css = """
    *{box-sizing:border-box}body{font:15px/1.65 system-ui,'Microsoft YaHei',sans-serif;color:#233247;background:#eef2f7;margin:0}main{max-width:1080px;margin:24px auto;background:white;padding:32px 40px}h1{font-size:28px;margin:0}h2{font-size:20px;border-bottom:1px solid #d6e0ec;padding-bottom:8px;margin-top:28px}h3{font-size:16px;margin:22px 0 10px}h4{margin:0 0 8px}.muted,dt,figcaption{color:#58677b;font-size:13px}.summary{border:1px solid #c6d7ed;border-radius:10px;padding:16px 20px;margin:18px 0}.summary strong{font-size:26px;color:#195ca8}dl{display:grid;grid-template-columns:120px 1fr;gap:5px 12px}dd{margin:0;overflow-wrap:anywhere}table{width:100%;border-collapse:collapse;font-size:11px;font-variant-numeric:tabular-nums}th,td{padding:7px 5px;border:1px solid #d6e0ec;text-align:left;overflow-wrap:anywhere}th{background:#edf3fc}.table-wrap{overflow-x:auto}.routes{max-width:560px}.route{display:grid;grid-template-columns:40px 1fr 70px;align-items:center;gap:10px;margin:7px 0;font-size:13px}.track{height:10px;background:#e4ebf5;border-radius:5px;overflow:hidden}.track i{display:block;height:100%;background:#2879c9}.route b{text-align:right}.examples{display:grid;gap:16px;grid-template-columns:repeat(auto-fit,minmax(240px,1fr))}.example{border:1px solid #d6e0ec;border-radius:8px;padding:14px;break-inside:avoid}.pair{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}figure{margin:0}figure img{width:100%;display:block;aspect-ratio:1;object-fit:contain}.overlay{position:relative}.overlay .heatmap{position:absolute;inset:0;opacity:.55}figcaption{font-size:11px}.notice{border-top:1px solid #d6e0ec;margin-top:24px;padding-top:16px}button{padding:9px 14px;border:0;border-radius:6px;background:#195ca8;color:white;cursor:pointer}.toolbar{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin-bottom:20px}.hash{font:11px/1.6 monospace;overflow-wrap:anywhere}@page{size:A4;margin:14mm}@media print{body{background:white}main{padding:0;margin:0;max-width:none}.toolbar{display:none}table{font-size:9px}thead{display:table-header-group}tr{break-inside:avoid}h2,h3,h4{break-after:avoid}.summary,.routes{break-inside:avoid}a{color:inherit}.examples{grid-template-columns:repeat(3,minmax(0,1fr))}.example{padding:8px}}@media(max-width:600px){main{padding:22px 16px;margin:0}dl{grid-template-columns:90px 1fr}}
    """
    return ('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; script-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        f'<title>智鉴分析报告 · {_text(upload["original_name"])}</title><style>{css}</style></head><body><main><div class="toolbar"><button onclick="window.print()">打印／另存为 PDF</button><span class="muted">此报告可离线打开；打印时可关闭浏览器页眉页脚。</span></div><header><h1>智鉴 · 伪造人脸分析报告</h1><p>{_text(upload["original_name"])}</p></header><div class="summary"><strong>{_text(VERDICTS.get(record["verdict"], "未生成结论"))} · {_score(record["score"])}</strong><p>{_text(record["summary"])}</p><p>状态：{_text(STATUSES.get(record["status"], record["status"]))} · 判断模式：{policy_text}</p></div>'
        f'<dl><dt>记录 ID</dt><dd>{_text(record["id"])}</dd><dt>原文件 SHA-256</dt><dd class="hash">{_text(upload["sha256"])}</dd><dt>媒体类型／大小</dt><dd>{_text(media_label)} ／ {upload["size_bytes"]} 字节</dd><dt>提交时间</dt><dd>{_date(record["created_at"])}</dd><dt>完成时间</dt><dd>{_date(record["completed_at"])}</dd><dt>所选／实际设备</dt><dd>{_text(selected_device)} ／ {_text(executed_device)}</dd></dl>'
        + ''.join(sections) + f'<footer class="notice"><p>{_text(data["interpretation"])}</p><p class="muted">导出时间：{_date(data["exported_at"])}。此文件是导出时的记录快照；后补数据明确标注来源，不改写原检测结果。</p><p class="hash">原报告数据校验值（SHA-256）：{data["record_sha256"]}</p></footer></main></body></html>').encode("utf-8")


def export_report(analysis, storage, format, *, include_technical=False):
    data = snapshot(analysis, storage, include_technical=include_technical)
    if format == "json":
        if not include_technical:
            for run in data["record"]["detector_runs"]:
                for key in ("model_version", "preprocessing_version", "evidence"):
                    run.pop(key, None)
            for extra in data["supplementary"]:
                extra.pop("routing_mean", None)
                for frame in extra["frames"]:
                    frame.pop("routing", None)
        return json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False).encode(), "application/json"
    if format == "csv":
        return csv_report(data), "text/csv"
    if format == "pdf":
        from app.services.office_report import pdf_report
        return pdf_report(data, storage), "application/pdf"
    if format == "docx":
        from app.services.office_report import docx_report
        return docx_report(data, storage), "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    return html_report(data, storage), "text/html"
