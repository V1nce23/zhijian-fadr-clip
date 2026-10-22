"use client";

import { useEffect, useRef, useState, type CSSProperties } from "react";
import { Check, ChevronLeft, ChevronRight, Layers3, LoaderCircle, RefreshCw, ScanFace } from "lucide-react";
import { formatThreshold } from "@/lib/format";
import { apiFetch } from "@/lib/api";
import type { DetectorRun, MediaType } from "@/lib/types";
import { RoutingWeights, type RoutingEvidence } from "@/components/routing-weights";

type EvidenceFrame = { frame_index: number; timestamp_seconds: number; score: number | null; image_url: string | null; image_kind?: string; status?: string; heatmap_url?: string | null; attention_uniform?: boolean; attention_origin?: string; routing?: RoutingEvidence | null };
type Progress = { sampled_frames?: number; extracted_frames?: number; processed_frames?: number; valid_frames?: number; scored_frames?: number };
type Attention = { status: string; completed_frames?: number; total_frames?: number; can_generate?: boolean; detail?: string };
type Gallery = { status: "pending" | "waiting" | "streaming" | "ready" | "stopped" | "failed" | "unavailable"; frames?: EvidenceFrame[]; stage?: string; progress?: Progress; detail?: string; attention?: Attention };
const videoStageLabels: Record<string, string> = { queued: "等待检测开始", reading: "正在读取原文件", preprocessing: "正在抽帧与对齐人脸", model_loading: "人脸已准备好，正在加载模型", detecting: "正在逐帧检测", reporting: "正在生成报告", completed: "检测完成", cancelled: "检测已取消，保留已完成帧", failed: "检测已停止，保留已完成帧" };

function scoreTone(score: number | null, low: number, high: number, strict = false) {
  return score == null ? "muted" : (strict ? score > high : score >= high) ? "danger" : score <= low ? "success" : "warning";
}

function frameLabel(frame: EvidenceFrame, stopped = false, video = true) {
  if (frame.score != null) return "已检测";
  if (frame.status === "no_face") return "无人脸，已跳过";
  if (stopped) return "未完成检测";
  return frame.status === "pending" ? (video ? "待抽帧" : "待读取图片") : frame.status === "sampled" ? "待对齐" : "待检测";
}

export function FrameGallery({ analysisId, run, mediaType }: { analysisId: string; run: DetectorRun; mediaType: MediaType }) {
  const [gallery, setGallery] = useState<Gallery>({ status: "pending" });
  const [attempt, setAttempt] = useState(0);
  const [active, setActive] = useState(0);
  const [width, setWidth] = useState(600);
  const [failedImage, setFailedImage] = useState<string | null>(null);
  const [heatmap, setHeatmap] = useState(true);
  const [opacity, setOpacity] = useState(55);
  const [attentionAttempt, setAttentionAttempt] = useState(0);
  const [submittingAttention, setSubmittingAttention] = useState(false);
  const [attentionError, setAttentionError] = useState<string | null>(null);
  const [failedHeatmap, setFailedHeatmap] = useState<string | null>(null);
  const [following, setFollowing] = useState(true);
  const manual = useRef(false);
  const attempted = useRef(0);
  const urls = useRef(new Map<string, { url: string; time: number }>());
  const deck = useRef<HTMLDivElement>(null);
  const cards = useRef<(HTMLButtonElement | null)[]>([]);
  const live = run.evidence?.visualization as { revision?: number } | undefined;
  const revision = live?.revision ?? 0;

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    const retry = attempt > attempted.current;
    attempted.current = attempt;
    let first = true;
    const refresh = async () => {
      try {
        const next = await apiFetch<Gallery>(`/analyses/${analysisId}/runs/${run.id}/gallery`, {
          signal: controller.signal, method: first && retry ? "POST" : "GET",
        });
        first = false;
        if (controller.signal.aborted) return;
        // Keep the same signed URL while a frame is unchanged, avoiding repeated
        // image downloads on each real progress update. Refresh before expiry.
        next.frames = next.frames?.map(item => {
          const result = { ...item };
          for (const field of ["image_url", "heatmap_url"] as const) {
            const url = item[field];
            if (!url) continue;
            const key = `${item.frame_index}:${item.image_kind ?? "aligned_face"}:${field}`;
            const cached = urls.current.get(key);
            if (cached && Date.now() - cached.time < 230_000) result[field] = cached.url;
            else urls.current.set(key, { url, time: Date.now() });
          }
          return result;
        });
        setGallery(next);
        if (!manual.current && next.status === "streaming" && next.frames?.length) {
          const scored = next.frames.map(item => item.score != null).lastIndexOf(true);
          const prepared = next.frames.map(item => !!item.image_url).lastIndexOf(true);
          setActive(scored >= 0 ? scored : Math.max(prepared, 0));
        }
        if (["pending", "waiting", "streaming", "ready"].includes(next.status)) {
          timer = window.setTimeout(() => void refresh(), next.status === "ready" && next.attention?.status !== "pending" ? 240_000 : 1000);
        }
      } catch (error) {
        if (!controller.signal.aborted) setGallery(current => ({ ...current, status: "failed", detail: error instanceof Error ? error.message : "检测帧暂时无法加载" }));
      }
    };
    void refresh();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [analysisId, run.id, attempt, revision, attentionAttempt]);

  const frames = gallery.frames ?? [];
  const frame = frames[Math.min(active, frames.length - 1)];
  const video = mediaType === "video";
  const stageLabels = video ? videoStageLabels : { ...videoStageLabels, reading: "正在读取图片", preprocessing: "正在提取并对齐人脸", detecting: "正在检测人脸", cancelled: "检测已取消，保留已完成结果", failed: "检测已停止，保留已完成结果" };
  const thresholds = run.evidence?.thresholds as { authentic_max?: number; manipulated_min?: number; operator?: string } | undefined;
  const low = thresholds?.authentic_max ?? 0.35;
  const high = thresholds?.manipulated_min ?? 0.65;
  const strict = thresholds?.operator === ">";
  const tone = frame ? scoreTone(frame.score, low, high, strict) : "muted";
  const step = Math.max(width < 480 ? 46 : 27, Math.min(62, (width - 138) / Math.max(frames.length - 1, 1)));
  const busy = ["pending", "waiting", "streaming"].includes(gallery.status);
  const stopped = gallery.status === "stopped";
  const stage = gallery.stage ?? (gallery.status === "ready" ? "completed" : gallery.status === "waiting" ? "queued" : "reading");
  const counts = gallery.progress ?? {};
  const total = counts.sampled_frames ?? Number(run.evidence?.sampled_frames ?? frames.length);
  const valid = counts.valid_frames ?? frames.filter(item => item.status !== "no_face").length;
  const extracted = counts.extracted_frames ?? (busy ? 0 : total);
  const processed = counts.processed_frames ?? (busy ? 0 : total);
  const scored = counts.scored_frames ?? frames.filter(item => item.score != null).length;
  const steps = [
    { label: video ? "抽帧" : "读取图片", value: video ? (total ? `${extracted} / ${total}` : "等待开始") : extracted > 0 ? "已读取" : "等待读取", done: total > 0 && extracted === total },
    { label: "人脸对齐", value: video ? (total ? `${processed} / ${total}` : "等待开始") : processed > 0 ? valid > 0 ? "已对齐" : "未检出人脸" : "等待对齐", done: total > 0 && processed === total },
    { label: video ? "逐帧检测" : "人脸检测", value: video ? (valid > 0 ? `${scored} / ${valid}` : "等待人脸") : scored > 0 ? "已检测" : "等待检测", done: valid > 0 && scored === valid },
    { label: "生成报告", value: stage === "completed" ? "已完成" : stopped ? "已停止" : "等待汇总", done: stage === "completed" },
  ];

  useEffect(() => {
    if (!deck.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(deck.current);
    return () => observer.disconnect();
  }, [frames.length, gallery.status]);

  const select = (index: number, focus = false) => {
    const next = Math.max(0, Math.min(index, frames.length - 1));
    manual.current = true;
    setFollowing(false);
    setActive(next);
    if (focus) cards.current[next]?.focus();
  };
  const retry = () => { setFailedImage(null); setFailedHeatmap(null); urls.current.clear(); setAttempt(value => value + 1); };
  const generateAttention = async () => {
    setSubmittingAttention(true);
    setAttentionError(null);
    try {
      const attention = await apiFetch<Attention>(`/analyses/${analysisId}/runs/${run.id}/attention`, { method: "POST" });
      setGallery(current => ({ ...current, attention }));
      setAttentionAttempt(value => value + 1);
    } catch (error) {
      setAttentionError(error instanceof Error ? error.message : "注意力图暂时无法生成，请重试");
    } finally { setSubmittingAttention(false); }
  };
  const attention = gallery.attention;
  const paired = heatmap && attention?.status !== "unavailable";

  return <section className={`surface frame-gallery ${video ? "is-video" : "is-image"}`} aria-label="检测结果可视化">
    <div className="frame-gallery-heading">
      <div><h2><Layers3 size={18} />{video ? busy ? "检测过程可视化" : "逐帧检测可视化" : "人脸检测可视化"}</h2><p>{busy ? video ? "卡片随真实处理进度更新，完成检测后才显示分数" : "直接读取图片，展示人脸对齐与检测结果" : video ? "悬停或点击卡片，查看这一帧的画面与检测分数" : "查看实际检测的人脸区域与分数"}</p></div>
      {frames.length > 0 && <span className="frame-count">{video ? `${valid} / ${total} 帧有效` : "256 × 256 RGB"}</span>}
    </div>
    <div className={`frame-process ${stopped ? "is-stopped" : ""}`}>
      <div className="frame-process-status" role="status">{busy ? <LoaderCircle size={15} className="animate-spin" /> : stage === "completed" ? <Check size={15} /> : <ScanFace size={15} />}<span>{stageLabels[stage] ?? gallery.detail ?? "正在准备可视化"}</span>{busy && <button aria-pressed={following} onClick={() => { manual.current = false; setFollowing(true); }}>跟随进度{following ? " · 开" : " · 关"}</button>}</div>
      <div className="frame-process-steps">{steps.map((item, index) => <div key={item.label} className={item.done ? "is-done" : busy && (index < 2 ? stage === "preprocessing" : index === 2 ? ["model_loading", "detecting"].includes(stage) : stage === "reporting") ? "is-working" : ""}><span className="frame-step-dot">{item.done ? <Check size={11} /> : index + 1}</span><span>{item.label}<small>{item.value}</small></span></div>)}</div>
    </div>
    {gallery.status === "failed" && <div className="frame-load-notice" role="alert">{gallery.detail}<button className="secondary-button" onClick={retry}><RefreshCw size={13} />重试</button></div>}
    {!frame ? <div className="frame-gallery-empty" role="status">{busy ? <LoaderCircle size={24} className="animate-spin" /> : <ScanFace size={28} />}<p>{gallery.detail || (busy ? stageLabels[stage] : "暂无可展示的人脸帧")}</p><span>{busy ? video ? "抽样帧读取后会逐张展开，无需等待全部检测完成" : "读取图片后展示人脸，无需抽帧" : "原检测结果会保留在报告中"}</span></div>
      : <>
        <div className={`frame-spotlight ${paired ? "with-attention" : ""}`}>
          <div className={`frame-visual-pair ${paired ? "is-paired" : ""}`}>
          <div className="frame-face-stage">
            {!frame.image_url ? <div className="frame-image-error">{busy && frame.status === "pending" ? <LoaderCircle size={28} className="animate-spin" /> : <ScanFace size={28} />}<span>{frameLabel(frame, stopped, video)}</span></div>
              : failedImage === frame.image_url ? <div className="frame-image-error"><ScanFace size={28} /><span>帧图暂时无法加载</span><button className="secondary-button" onClick={retry}>重新加载</button></div>
                // eslint-disable-next-line @next/next/no-img-element
                : <img src={frame.image_url} alt={video ? `第 ${frame.frame_index + 1} 帧${frame.image_kind === "source_frame" ? "原始画面" : "检测人脸"}` : "检测人脸裁剪图"} onError={() => setFailedImage(frame.image_url)} className="frame-face-image" />}
            <span className="frame-image-caption"><ScanFace size={13} />{frame.image_kind === "source_frame" ? video ? "原始抽样画面" : "原始图片" : frame.image_url ? "对齐后检测人脸" : "等待画面"}</span>
          </div>
          {paired && <div className="frame-face-stage frame-attention-stage">
            {frame.heatmap_url && frame.image_url && failedHeatmap !== frame.heatmap_url ? <>
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img src={frame.image_url} alt="" className="frame-face-image" />
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img key={frame.heatmap_url} src={frame.heatmap_url} alt={video ? `第 ${frame.frame_index + 1} 帧路由加权空间注意力` : "人脸路由加权空间注意力"} className="frame-attention-layer" style={{ opacity: opacity / 100 }} onError={() => setFailedHeatmap(frame.heatmap_url ?? null)} />
            </> : <div className="frame-image-error">{attention?.status === "pending" || busy ? <LoaderCircle size={24} className="animate-spin" /> : <Layers3 size={26} />}<span>{frame.status === "no_face" ? "无人脸，不生成注意力图" : failedHeatmap === frame.heatmap_url && frame.heatmap_url ? "注意力图加载失败" : frame.score == null ? "检测后显示注意力图" : attention?.status === "pending" ? "正在生成注意力图" : "尚未生成注意力图"}</span>{failedHeatmap === frame.heatmap_url && frame.heatmap_url && <button className="secondary-button" onClick={retry}>重新加载</button>}</div>}
            <span className="frame-image-caption">{frame.attention_uniform ? "关注分布较均匀" : "路由加权空间注意力"}</span>
          </div>}
          </div>
          <div className="frame-inspector" style={{ "--frame-tone": `var(--${tone})` } as CSSProperties}>
            <div className="frame-inspector-top"><span className="frame-eyebrow">{video ? "当前抽样帧" : "当前检测人脸"}</span>{video && <span className="frame-position">{Math.min(active + 1, frames.length).toString().padStart(2, "0")} / {frames.length}</span>}</div>
            <h3>{video ? `第 ${frame.frame_index + 1} 帧` : "人脸区域"}</h3>
            <p className="frame-time">{video ? `约 ${frame.timestamp_seconds.toFixed(2)} 秒 · 均匀抽样` : "最大人脸 · 对齐裁剪"}</p>
            <div className="frame-score" aria-live="polite"><span className="frame-eyebrow">伪造倾向分数</span><strong>{frame.score == null ? "—" : (frame.score * 100).toFixed(1)}{frame.score != null && <small>%</small>}</strong>{frame.score != null ? <span className={`frame-verdict frame-verdict-${tone}`}>{tone === "danger" ? "疑似伪造" : tone === "success" ? "疑似真实" : "不确定"}</span> : <span className="frame-pending-label">{frameLabel(frame, stopped, video)}</span>}</div>
            <div className="frame-score-meter"><span style={{ width: frame.score == null ? "0%" : `${frame.score * 100}%` }} /></div>
            <p className="frame-score-note">{frame.status === "no_face" ? "未检测到可对齐人脸，此帧不参与总体分数。" : frame.score == null ? "仅在模型实际完成该帧检测后显示分数。" : strict ? `${video ? "单帧" : "图片"}分数 > ${formatThreshold(high)} 判为疑似伪造；分数不是已校准的概率。` : "分数越高，模型判断人脸存在伪造的倾向越强。"}</p>
            <details className="technical-details frame-technical"><summary>{video ? "当前帧技术详情" : "人脸技术详情"}</summary><div className="technical-details-body"><RoutingWeights title={video ? "当前帧路由权重" : "图片路由权重"} weights={frame.routing?.weights} layers={frame.routing?.layers} origin={frame.routing?.origin} emptyText={frame.status === "no_face" ? "此帧无人脸，不计算路由权重。" : frame.score == null ? "该帧完成检测后显示路由权重。" : "旧记录未保存逐帧路由；可通过生成注意力图补充，平均权重不能代替当前帧权重。"} /></div></details>
            {video && <div className="frame-navigation"><button className="icon-button" aria-label="上一检测帧" disabled={active === 0} onClick={() => select(active - 1)}><ChevronLeft size={17} /></button><span>逐帧浏览</span><button className="icon-button" aria-label="下一检测帧" disabled={active >= frames.length - 1} onClick={() => select(active + 1)}><ChevronRight size={17} /></button></div>}
          </div>
        </div>
        {attention?.status !== "unavailable" && <div className="frame-attention-controls">
          <div className="frame-attention-toolbar">
            <label className="frame-attention-toggle"><input type="checkbox" checked={heatmap} onChange={event => setHeatmap(event.target.checked)} />显示注意力图</label>
            {heatmap && <label className="frame-opacity">叠加强度<input aria-label="注意力图叠加强度" type="range" min="0" max="100" value={opacity} onChange={event => setOpacity(Number(event.target.value))} /><output>{opacity}%</output></label>}
            {attention?.status === "pending" || submittingAttention ? <span className="frame-attention-progress" role="status"><LoaderCircle size={14} className="animate-spin" />生成中 · {attention?.completed_frames ?? 0} / {attention?.total_frames ?? valid}</span>
              : attention?.can_generate && <button className="secondary-button" onClick={() => void generateAttention()}>{attention.status === "failed" ? "重试生成注意力图" : "生成注意力图"}</button>}
          </div>
          {heatmap && <div className="frame-attention-explanation"><span className="frame-attention-legend"><span>关注低</span><i /><span>关注高</span></span><span>颜色表示当前帧内的相对关注程度，不代表伪造位置或伪造概率。{frame.attention_origin === "supplementary_replay" ? "此图为补充生成，保留原检测分数。" : frame.heatmap_url ? "使用本次检测的模型注意力。" : "结合模型空间注意力与层路由权重生成。"}</span></div>}
          {(attentionError || attention?.status === "failed") && <p className="frame-attention-error" role="alert">{attentionError || attention?.detail || "注意力图暂时不可用，原检测分数已保留。"}</p>}
        </div>}
        {video && <>
          <div className="frame-deck-label"><span>抽样帧卡片</span><span>悬停查看 · 横向滑动</span></div>
          <div className="frame-deck-scroll" ref={deck}>
            <div className="frame-deck" role="group" aria-label="检测帧卡片，可用左右方向键切换" style={{ width: Math.max(width, 138 + (frames.length - 1) * step) }}>
              {frames.map((item, index) => <button key={item.frame_index} ref={node => { cards.current[index] = node; }} type="button" className={`frame-card ${item.status === "no_face" ? "is-skipped" : ""} ${item.image_kind === "source_frame" ? "is-source" : ""}`} aria-label={`第 ${item.frame_index + 1} 帧，${item.score == null ? frameLabel(item, stopped, video) : `伪造倾向 ${(item.score * 100).toFixed(1)}%`}`} aria-pressed={index === active} tabIndex={index === active ? 0 : -1}
                onPointerEnter={event => { if (event.pointerType === "mouse") select(index); }} onFocus={() => select(index)} onClick={() => select(index)}
                onKeyDown={event => { const next = event.key === "ArrowRight" ? index + 1 : event.key === "ArrowLeft" ? index - 1 : event.key === "Home" ? 0 : event.key === "End" ? frames.length - 1 : null; if (next !== null) { event.preventDefault(); select(next, true); } }}
                style={{ "--card-hit-width": `${index === frames.length - 1 ? 106 : step}px`, left: 16 + index * step, zIndex: index === active ? 60 : index + 1, "--card-angle": `${Math.max(-5, Math.min(5, (index - (frames.length - 1) / 2) * 0.6))}deg`, "--frame-tone": `var(--${scoreTone(item.score, low, high, strict)})` } as CSSProperties}>
                <span className="frame-card-visual"><span className="frame-card-index">{(index + 1).toString().padStart(2, "0")}</span>
                  {item.image_url ? <>
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img src={item.image_url} alt="" loading="lazy" draggable={false} />
                  </> : <span className="frame-card-placeholder"><ScanFace size={23} /><small>{item.status === "pending" ? "待抽帧" : "等待画面"}</small></span>}
                  <span className="frame-card-footer"><span>{item.timestamp_seconds.toFixed(1)}s</span><strong>{item.score == null ? item.status === "no_face" ? "无人脸" : stopped ? "未完成" : item.status === "sampled" ? "待对齐" : item.status === "pending" ? "待抽帧" : "待检测" : `${(item.score * 100).toFixed(0)}%`}</strong></span>
                </span>
              </button>)}
            </div>
          </div>
          <div className="frame-score-strip" aria-label="各检测帧分数概览">{frames.map((item, index) => <span key={item.frame_index} className={`${index === active ? "is-active" : ""} ${item.score == null ? "is-pending" : ""}`} title={`第 ${item.frame_index + 1} 帧：${item.score == null ? frameLabel(item, stopped, video) : `${(item.score * 100).toFixed(1)}%`}`} style={{ "--bar-height": `${item.score == null ? 4 : Math.max(4, item.score * 100)}%`, "--frame-tone": `var(--${scoreTone(item.score, low, high, strict)})` } as CSSProperties}><i /></span>)}</div>
          <div className="frame-gallery-footnote"><span>{total} 帧抽样 · {valid} 帧有效 · {scored} 帧已检测{processed - valid > 0 ? ` · ${processed - valid} 帧无人脸` : ""}</span><span>{stage === "completed" ? "总体分数为有效帧分数的平均值" : "待检测与无人脸帧均不计作 0 分"}</span></div>
        </>}
      </>}
  </section>;
}
