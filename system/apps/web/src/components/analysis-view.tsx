"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  Ban,
  Clock3,
  FileWarning,
  Image as ImageIcon,
  LoaderCircle,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Trash2,
  Video,
} from "lucide-react";

import { apiFetch } from "@/lib/api";
import { VideoPreview } from "@/components/video-preview";
import { FrameGallery } from "@/components/frame-gallery";
import { RoutingWeights } from "@/components/routing-weights";
import { ReportExport } from "@/components/report-export";
import {
  detectorLabel,
  detectorRunStatusLabels,
  formatBytes,
  formatThreshold,
  formatDate,
  mediaLabels,
  shortHash,
  statusLabels,
  verdictLabels,
} from "@/lib/format";
import type { Analysis, DetectorRun, MediaType, Verdict } from "@/lib/types";

const terminalStatuses = new Set(["completed", "partially_completed", "failed", "cancelled"]);
const eventNames = [
  "analysis.queued",
  "analysis.preprocessing",
  "analysis.device_selected",
  "analysis.visualization",
  "detector.started",
  "detector.progress",
  "analysis.aggregating",
  "analysis.completed",
  "analysis.failed",
  "analysis.cancelled",
];

function MediaIcon({ type, size = 18 }: { type: MediaType; size?: number }) {
  if (type === "image") return <ImageIcon size={size} />;
  if (type === "video") return <Video size={size} />;
  return <FileWarning size={size} />;
}

function VerdictIcon({ verdict }: { verdict: Verdict | null }) {
  if (verdict === "likely_manipulated") return <ShieldAlert size={26} />;
  if (verdict === "likely_authentic") return <ShieldCheck size={26} />;
  return <FileWarning size={26} />;
}

function VerdictBadge({ verdict }: { verdict: Verdict | null }) {
  if (!verdict) return <span className="text-[var(--muted)]">—</span>;
  return <span className={`inline-block rounded-[4px] border px-2 py-0.5 text-xs font-medium ${verdictStyle(verdict)}`}>{verdictLabels[verdict]}</span>;
}

function verdictStyle(verdict: Verdict | null): string {
  if (verdict === "likely_manipulated") return "border-[var(--danger-border)] bg-[var(--danger-soft)] text-[var(--danger)]";
  if (verdict === "likely_authentic") return "border-[var(--success-border)] bg-[var(--success-soft)] text-[var(--success)]";
  return "border-[var(--warning-border)] bg-[var(--warning-soft)] text-[var(--warning)]";
}

function MediaPreview({ type, url, name, uploadId }: { type: MediaType; url: string; name: string; uploadId: string }) {
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  if (type === "video") return <VideoPreview key={uploadId} uploadId={uploadId} originalUrl={url} name={name} />;
  if (failedUrl === url) return <div className="py-8 text-center text-sm text-[var(--muted)]"><p>图片暂时无法预览</p><a className="mt-2 inline-block font-medium text-[var(--accent)]" href={url} target="_blank" rel="noreferrer">打开或下载原文件</a></div>;
  if (type === "image") {
    // User media is loaded from a short-lived signed object-storage URL.
    // eslint-disable-next-line @next/next/no-img-element
    return <img src={url} alt={name} onError={() => setFailedUrl(url)} className="block max-h-[420px] max-w-full rounded-xl border border-[var(--border)] object-contain" />;
  }
  return <p className="text-sm text-[var(--muted)]">此历史文件类型已停用。</p>;
}

export function AnalysisView({ analysisId }: { analysisId: string }) {
  const router = useRouter();
  const [analysis, setAnalysis] = useState<Analysis | null>(null);
  const [downloadUrl, setDownloadUrl] = useState<string | null>(null);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const next = await apiFetch<Analysis>(`/analyses/${analysisId}`);
      setAnalysis(next);
      setSelectedRunId((current) =>
        current && next.detector_runs.some((run) => run.id === current)
          ? current
          : (next.detector_runs[0]?.id ?? null),
      );
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "无法加载分析报告");
    } finally {
      setLoading(false);
    }
  }, [analysisId]);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const terminal = analysis ? terminalStatuses.has(analysis.status) : false;
  const streaming = analysis !== null && !terminal;
  const uploadId = analysis?.upload.id;

  useEffect(() => {
    if (!uploadId || analysis?.upload.status === "deleted") return;
    const controller = new AbortController();
    const refreshUrl = async () => {
      try {
        const source = await apiFetch<{ url: string }>(`/uploads/${uploadId}/download-url`, { signal: controller.signal });
        if (!controller.signal.aborted) setDownloadUrl(source.url);
      } catch (requestError) {
        if (!controller.signal.aborted) setError(requestError instanceof Error ? requestError.message : "无法加载媒体预览");
      }
    };
    void refreshUrl();
    // Signed URLs expire after five minutes; refresh without reconnecting SSE.
    const timer = window.setInterval(() => void refreshUrl(), 240_000);
    return () => { controller.abort(); window.clearInterval(timer); };
  }, [uploadId, analysis?.upload.status]);

  useEffect(() => {
    if (!streaming) return;
    const source = new EventSource(`/api/v1/analyses/${analysisId}/events`, { withCredentials: true });
    let refreshTimer: number | undefined;
    const refresh = () => {
      window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => void load(), 50);
    };
    eventNames.forEach((name) => source.addEventListener(name, refresh));
    // Keep the same EventSource: native reconnects carry Last-Event-ID.
    return () => {
      window.clearTimeout(refreshTimer);
      eventNames.forEach((name) => source.removeEventListener(name, refresh));
      source.close();
    };
  }, [analysisId, load, streaming]);

  useEffect(() => {
    if (!streaming) return;
    const timer = window.setInterval(() => void load(), 3000);
    return () => window.clearInterval(timer);
  }, [load, streaming]);

  const selectedRun =
    analysis?.detector_runs.find((run) => run.id === selectedRunId) ?? analysis?.detector_runs[0] ?? null;
  const verdictCounts = useMemo(() => {
    if (!analysis) return null;
    const withVerdict = analysis.detector_runs.filter((run) => run.verdict);
    return {
      total: withVerdict.length,
      manipulated: withVerdict.filter((run) => run.verdict === "likely_manipulated").length,
      authentic: withVerdict.filter((run) => run.verdict === "likely_authentic").length,
      uncertain: withVerdict.filter((run) => run.verdict === "uncertain").length,
    };
  }, [analysis]);
  const runningLabel = useMemo(() => {
    if (!analysis) return "正在读取任务";
    if (analysis.status === "queued") return "等待检测";
    const live = analysis.detector_runs[0]?.evidence?.visualization as { stage?: string; scored_frames?: number; valid_frames?: number } | undefined;
    if (live?.stage === "model_loading") return "人脸已准备好，正在加载模型";
    if (live?.stage === "detecting") return analysis.upload.media_type === "video" ? `逐帧检测中 · ${live.scored_frames ?? 0} / ${live.valid_frames ?? 0} 帧` : "正在检测图片人脸";
    if (live?.stage === "preprocessing") return analysis.upload.media_type === "video" ? "正在抽帧并对齐人脸" : "正在提取并对齐图片人脸";
    if (analysis.status === "preprocessing") return "正在读取原始素材与准备人脸";
    if (analysis.status === "running") return "模型检测中";
    if (analysis.status === "aggregating") return "正在汇总结果";
    return statusLabels[analysis.status];
  }, [analysis]);

  const runLabelsById = useMemo(() => {
    const labels = new Map<string, string>();
    for (const run of analysis?.detector_runs ?? []) {
      labels.set(run.id, detectorLabel(run.detector_id));
    }
    return labels;
  }, [analysis]);
  const runLabel = (run: DetectorRun): string =>
    runLabelsById.get(run.id) ?? detectorLabel(run.detector_id);

  const cancel = async () => {
    setActionLoading(true);
    try { await apiFetch(`/analyses/${analysisId}/cancel`, { method: "POST" }); await load(); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "取消失败"); }
    finally { setActionLoading(false); }
  };

  const remove = async () => {
    if (!analysis || !window.confirm(`删除分析记录后，${terminal ? "" : "未完成的检测将停止，"}报告和证据将无法查看，源文件会保留。确认继续？`)) return;
    setActionLoading(true);
    try { await apiFetch(`/analyses/${analysisId}`, { method: "DELETE" }); router.push("/history"); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "删除失败"); setActionLoading(false); }
  };

  if (loading && !analysis) {
    return <div className="grid gap-4"><div className="skeleton h-24 rounded-[6px]" /><div className="skeleton h-96 rounded-[6px]" /></div>;
  }
  if (!analysis) {
    return <div className="surface flex min-h-72 flex-col items-center justify-center p-6 text-center"><AlertCircle size={34} className="text-[var(--danger)]" /><h1 className="mt-4 text-lg font-semibold">无法打开分析报告</h1><p className="mt-2 text-sm text-[var(--muted)]">{error}</p><button className="primary-button mt-5" onClick={() => void load()}>重试</button></div>;
  }

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <h1 className="text-2xl font-semibold sm:text-[2rem]">分析报告</h1>
          <p className="mt-3 max-w-3xl break-all text-sm" title={analysis.upload.original_name}>{analysis.upload.original_name}</p>
          <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[var(--muted)]"><span>{mediaLabels[analysis.upload.media_type]}</span><span>{formatBytes(analysis.upload.size_bytes)}</span><span>{formatDate(analysis.created_at)}</span>{analysis.requested_device && <span>所选设备：{analysis.requested_device === "cuda" ? "GPU" : "CPU"}</span>}{analysis.executed_device && <span>实际设备：{analysis.executed_device === "cuda" ? "GPU" : "CPU"}</span>}</div>
        </div>
        <div className="flex flex-wrap items-start gap-2">
          <ReportExport analysisId={analysisId} enabled={terminal} />
          <button className="secondary-button" onClick={() => void load()} title="刷新报告"><RefreshCw size={16} /><span className="hidden sm:inline">刷新</span></button>
          {!terminal && <button className="secondary-button" disabled={actionLoading || analysis.cancel_requested} onClick={() => void cancel()}><Ban size={16} />{analysis.cancel_requested ? "取消中" : "取消任务"}</button>}
          <button className="danger-button" disabled={actionLoading} onClick={() => void remove()}><Trash2 size={16} />删除记录</button>
        </div>
      </div>

      {error && <div className="mb-4 flex items-center gap-2 rounded-[5px] border border-[var(--danger-border)] bg-[var(--danger-soft)] px-4 py-3 text-[var(--danger)]"><AlertCircle size={17} />{error}</div>}

      {!terminal ? (
        <section className="surface mb-5 p-5 sm:p-6" aria-live="polite">
          <div className="flex items-center justify-between gap-4"><div className="flex items-center gap-3"><span className="flex h-10 w-10 items-center justify-center rounded-[5px] bg-[var(--primary-soft)] text-[var(--accent)]"><LoaderCircle size={20} className="animate-spin" /></span><div><h2 className="font-semibold">{runningLabel}</h2><p className="mt-1 text-xs text-[var(--muted)]">{statusLabels[analysis.status]}</p></div></div><strong className="text-xl tabular-nums text-[var(--accent)]">{analysis.progress}%</strong></div>
          <div className="mt-5 h-2 overflow-hidden rounded-[3px] bg-[var(--surface-muted)]"><div className="h-full bg-[var(--accent)] transition-[width] duration-500" style={{ width: `${analysis.progress}%` }} /></div>
        </section>
      ) : analysis.verdict ? (
        <section className="surface mb-5 flex flex-col gap-4 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
          <div className="flex items-start gap-4"><span className={`shrink-0 rounded-lg p-2 ${verdictStyle(analysis.verdict)}`}><VerdictIcon verdict={analysis.verdict} /></span><div><p className="text-xs font-semibold uppercase">总体结论</p><h2 className="mt-1 text-xl font-bold">{verdictLabels[analysis.verdict]}</h2><p className="mt-2 max-w-3xl text-sm leading-6 text-[var(--muted)]">{analysis.summary}</p>{analysis.detector_runs.length > 1 && verdictCounts && verdictCounts.total > 0 && <p className="mt-2 text-xs text-[var(--muted)]">模型结论：{verdictCounts.manipulated} 个判定疑似伪造 · {verdictCounts.authentic} 个判定疑似真实 · {verdictCounts.uncertain} 个不确定</p>}</div></div>
          <div className="shrink-0 border-t border-[var(--border)] pt-3 text-left sm:border-l sm:border-t-0 sm:pl-6 sm:pt-0"><div className="text-xs font-semibold opacity-75">检测分数</div><div className="mt-1 text-3xl font-bold tabular-nums">{Math.round((analysis.score ?? 0) * 100)}<span className="text-base">%</span></div></div>
        </section>
      ) : (
        <section className="surface mb-5 flex items-center gap-3 p-5"><Ban size={22} className="text-[var(--muted)]" /><div><h2 className="font-semibold">{statusLabels[analysis.status]}</h2><p className="mt-1 text-sm text-[var(--muted)]">{analysis.summary || "任务没有生成检测结论。"}</p></div></section>
      )}

      {analysis.threshold_policy && <section className="surface mb-5 px-5 py-4 text-sm" aria-label="本次判断模式">
        <p className="font-semibold">判断模式：{analysis.threshold_policy.label}<span className="ml-3 font-normal text-[var(--accent)]">{analysis.threshold_policy.score_unit === "video_mean" ? "视频均值" : "图片分数"} &gt; {formatThreshold(analysis.threshold_policy.threshold)}</span></p>
        <p className="mt-2 text-xs leading-5 text-[var(--muted)]">分数严格大于阈值判为疑似伪造，否则为疑似真实。{analysis.threshold_policy.score_unit === "video_mean" && `逐帧卡片使用单帧阈值 ${formatThreshold(analysis.threshold_policy.frame_threshold)}，总体结论使用有效帧均值。`}本报告保留提交时的模式与阈值；分数不是已校准的伪造概率。</p>
      </section>}

      <div className="grid grid-cols-1 gap-5 xl:grid-cols-[minmax(0,1.45fr)_minmax(320px,0.55fr)]">
        <div className="min-w-0 space-y-5">
          <section className="surface overflow-hidden">
            <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-4">
              <h2 className="flex items-center gap-2 font-semibold"><ShieldCheck size={17} />模型结果</h2>
              <span className="text-xs text-[var(--muted)]">{analysis.detector_runs.length} 个模型</span>
            </div>
            {analysis.detector_runs.length === 0 ? (
              <p className="p-5 text-sm text-[var(--muted)]">暂无模型记录</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full min-w-[340px] text-left text-sm">
                  <thead className="bg-[var(--surface-muted)] text-xs font-semibold text-[var(--muted)]">
                    <tr>
                      <th className="border-b border-[var(--border)] px-5 py-2.5">模型</th>
                      <th className="border-b border-[var(--border)] px-3 py-2.5">结论</th>
                      <th className="border-b border-[var(--border)] px-3 py-2.5">分数</th>
                      <th className="border-b border-[var(--border)] px-3 py-2.5">状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analysis.detector_runs.map((run) => {
                      const active = run.id === selectedRun?.id;
                      const pending = run.status === "queued" || run.status === "running";
                      return (
                        <tr
                          key={run.id}
                          tabIndex={0}
                          onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); setSelectedRunId(run.id); } }}
                          aria-selected={active}
                          className={`cursor-pointer border-b border-[var(--border)] last:border-0 ${active ? "bg-[var(--primary-soft)]" : "hover:bg-[var(--surface-muted)]"}`}
                          title="查看模型证据"
                          onClick={() => setSelectedRunId(run.id)}
                        >
                          <td className="px-5 py-3 font-medium text-[var(--foreground)]">{runLabel(run)}</td>
                          <td className="px-3 py-3">{run.verdict ? <VerdictBadge verdict={run.verdict} /> : pending ? <span className="flex items-center gap-1.5 text-xs text-[var(--muted)]"><LoaderCircle size={14} className="animate-spin" aria-label="检测中" role="img" />{detectorRunStatusLabels[run.status]}</span> : "—"}</td>
                          <td className="px-3 py-3 font-mono tabular-nums">{run.score != null ? `${Math.round(run.score * 100)}%` : pending ? <LoaderCircle size={14} className="animate-spin text-[var(--muted)]" aria-label="检测中" role="img" /> : "—"}</td>
                          <td className="px-3 py-3 text-xs">{pending ? <span className="flex items-center gap-1.5 text-[var(--muted)]"><LoaderCircle size={13} className="animate-spin" aria-label="检测中" role="img" />{detectorRunStatusLabels[run.status]}</span> : detectorRunStatusLabels[run.status] ?? run.status}{run.error ? <span className="ml-1 cursor-help text-[var(--danger)]" title={run.error}>⚠</span> : null}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </section>
          {selectedRun && <details key={selectedRun.id} className="surface technical-details">
            <summary>技术详情<span>模型版本与内部参数</span></summary>
            <div className="technical-details-body">
              <dl className="technical-versions"><div><dt>模型版本</dt><dd>{selectedRun.model_version || "未记录"}</dd></div><div><dt>预处理版本</dt><dd>{selectedRun.preprocessing_version || "未记录"}</dd></div></dl>
              {(analysis.upload.media_type === "video" || analysis.upload.status === "deleted") && <><RoutingWeights title={analysis.upload.media_type === "video" ? "有效人脸帧平均路由权重" : "图片路由权重"} weights={selectedRun.evidence?.routing_weights} layers={selectedRun.evidence?.routing_layers} /><p className="routing-note">L6、L12、L18、L24 对应模型取用的四层特征；视频平均权重是有效人脸帧权重的算术平均。</p></>}
            </div>
          </details>}
          {selectedRun && analysis.upload.status !== "deleted" && <FrameGallery key={selectedRun.id} analysisId={analysisId} run={selectedRun} mediaType={analysis.upload.media_type} />}
        </div>

        <div className="min-w-0 space-y-5">
          <section className="surface overflow-hidden">
            <div className="flex items-center justify-between border-b border-[var(--border)] px-5 py-4"><h2 className="flex items-center gap-2 font-semibold"><MediaIcon type={analysis.upload.media_type} />媒体预览</h2><span className="text-xs text-[var(--muted)]">{analysis.upload.mime_type}</span></div>
            <div className="flex min-h-40 items-center justify-center bg-[var(--surface-muted)] p-4">{downloadUrl ? <MediaPreview key={analysis.upload.id} uploadId={analysis.upload.id} type={analysis.upload.media_type} url={downloadUrl} name={analysis.upload.original_name} /> : <LoaderCircle className="animate-spin text-[var(--muted)]" />}</div>
          </section>

          <section className="surface p-5">
            <h2 className="mb-4 font-semibold">文件信息</h2>
            <dl className="space-y-4 text-sm"><div><dt className="text-xs text-[var(--muted)]">文件名</dt><dd className="mt-1 break-all">{analysis.upload.original_name}</dd></div><div><dt className="text-xs text-[var(--muted)]">SHA-256</dt><dd className="mt-1 font-mono text-xs" title={analysis.upload.sha256}>{shortHash(analysis.upload.sha256)}</dd></div><div className="grid grid-cols-2 gap-4"><div><dt className="text-xs text-[var(--muted)]">大小</dt><dd className="mt-1">{formatBytes(analysis.upload.size_bytes)}</dd></div><div><dt className="text-xs text-[var(--muted)]">类型</dt><dd className="mt-1">{mediaLabels[analysis.upload.media_type]}</dd></div></div></dl>
          </section>

          <section className="surface flex items-start gap-3 p-4 text-xs leading-5 text-[var(--muted)]"><Clock3 size={16} className="mt-0.5 shrink-0" /><p>检测结果属于模型判断而非事实证明；小样本基准整体准确率有限，报告保留版本信息与不确定结论，不能用于高风险自动决策。</p></section>
        </div>
      </div>
    </div>
  );
}
