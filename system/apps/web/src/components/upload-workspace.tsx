"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import {
  AlertCircle,
  CheckCircle2,
  FileUp,
  Image as ImageIcon,
  LoaderCircle,
  ArrowRight,
  Cpu,
  FileCheck2,
  RefreshCw,
  Upload,
  Video,
  X,
} from "lucide-react";

import { AmbientBackdrop } from "@/components/ambient-backdrop";
import { MediaGraphic } from "@/components/media-graphic";
import { apiFetch } from "@/lib/api";
import { formatBytes, formatThreshold } from "@/lib/format";
import type { Analysis, DetectionDevice, DetectionDevices, ThresholdMode, ThresholdModeOption, Upload as UploadRecord } from "@/lib/types";

const MAX_BYTES = 256 * 1024 * 1024;

const mediaCards = [
  { icon: ImageIcon, title: "图片", formats: "JPG · PNG · WEBP · GIF · TIFF · HEIC", output: "PNG", note: "原图解码 · RGB 人脸" },
  { icon: Video, title: "视频", formats: "MP4 · MOV · WEBM · MKV · AVI", output: "MP4", note: "原视频解码 · 均匀抽帧" },
];

function fileKind(file: File) {
  const suffix = file.name.split(".").pop()?.toLowerCase() ?? "";
  if (file.type.startsWith("image/") || /^(jpg|jpeg|png|webp|gif|bmp|tif|tiff|avif|heic|heif)$/.test(suffix)) return "image";
  if (file.type.startsWith("video/") || /^(mp4|mov|m4v|webm|mkv|avi|wmv|mpg|mpeg|ogv)$/.test(suffix)) return "video";
  return null;
}

function FileKindIcon({ type }: { type: string }) {
  if (type === "image") return <ImageIcon size={22} />;
  if (type === "video") return <Video size={22} />;
  return <Video size={22} />;
}

export function UploadWorkspace() {
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragging, setDragging] = useState(false);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<"idle" | "uploading" | "creating">("idle");
  const [error, setError] = useState<string | null>(null);
  const [devices, setDevices] = useState<DetectionDevices | null>(null);
  const [device, setDevice] = useState<DetectionDevice>("cpu");
  const [deviceError, setDeviceError] = useState<string | null>(null);
  const deviceChosen = useRef(false);
  const [thresholdMode, setThresholdMode] = useState<ThresholdMode>("balanced");
  const [thresholdModes, setThresholdModes] = useState<ThresholdModeOption[]>([]);
  const [thresholdError, setThresholdError] = useState<string | null>(null);
  const [thresholdAttempt, setThresholdAttempt] = useState(0);
  const [customVideo, setCustomVideo] = useState("0.67");
  const [customFrame, setCustomFrame] = useState("0.85");
  const validThreshold = (value: string) => value.trim() !== "" && Number.isFinite(Number(value)) && Number(value) >= 0 && Number(value) <= 1 && Number(value) === Number(Number(value).toFixed(6));
  const customValid = thresholdMode !== "custom" || (validThreshold(customVideo) && validThreshold(customFrame));
  const selectedMode = thresholdModes.find(option => option.mode === thresholdMode);

  useEffect(() => {
    const controller = new AbortController();
    void apiFetch<ThresholdModeOption[]>("/analyses/threshold-modes", { signal: controller.signal })
      .then(options => { if (!controller.signal.aborted) { setThresholdModes(options); setThresholdError(null); } })
      .catch(error => { if (!controller.signal.aborted) setThresholdError(error instanceof Error ? error.message : "判断模式加载失败"); });
    return () => controller.abort();
  }, [thresholdAttempt]);

  useEffect(() => {
    const controller = new AbortController();
    const refreshDevices = async () => {
      try {
        const next = await apiFetch<DetectionDevices>("/analyses/devices", { signal: controller.signal });
        if (controller.signal.aborted) return;
        setDevices(next);
        setDeviceError(null);
        if (!deviceChosen.current && next.worker_ready) {
          setDevice(next.gpu ? "cuda" : "cpu");
          deviceChosen.current = true;
        }
      } catch (requestError) {
        if (!controller.signal.aborted) {
          setDeviceError(requestError instanceof Error ? requestError.message : "无法读取设备状态");
        }
      }
    };
    void refreshDevices();
    const interval = window.setInterval(() => void refreshDevices(), 10_000);
    return () => { controller.abort(); window.clearInterval(interval); };
  }, []);

  const deviceAvailable = !deviceError && !!devices?.worker_ready && (device === "cuda" ? devices.gpu : devices.cpu);

  const selectFile = (candidate?: File) => {
    setError(null);
    if (!candidate) return;
    if (!fileKind(candidate)) {
      setError("目前仅支持图片或视频文件");
      return;
    }
    if (candidate.size > MAX_BYTES) {
      setError("文件不能超过 256 MB");
      return;
    }
    setFile(candidate);
    setProgress(0);
  };

  const onInput = (event: ChangeEvent<HTMLInputElement>) => {
    selectFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    selectFile(event.dataTransfer.files?.[0]);
  };

  const upload = async () => {
    if (!file || !deviceAvailable || thresholdModes.length === 0 || !customValid) return;
    setPhase("uploading");
    setProgress(0);
    setError(null);
    try {
      const uploadRecord = await new Promise<UploadRecord>((resolve, reject) => {
        const data = new FormData();
        data.append("file", file);
        const request = new XMLHttpRequest();
        request.open("POST", "/api/v1/uploads");
        request.withCredentials = true;
        request.upload.onprogress = (event) => {
          if (event.lengthComputable) setProgress(Math.round((event.loaded / event.total) * 100));
        };
        request.onerror = () => reject(new Error("上传连接中断"));
        request.onload = () => {
          let body: { detail?: string } | UploadRecord = {};
          try { body = JSON.parse(request.responseText) as typeof body; } catch { /* ignore invalid error body */ }
          if (request.status >= 200 && request.status < 300) resolve(body as UploadRecord);
          else reject(new Error((body as { detail?: string }).detail ?? "上传失败"));
        };
        request.send(data);
      });

      setPhase("creating");
      const analysis = await apiFetch<Analysis>("/analyses", {
        method: "POST",
        body: JSON.stringify({ upload_id: uploadRecord.id, device, threshold_mode: thresholdMode, ...(thresholdMode === "custom" ? { custom_thresholds: { video_threshold: Number(customVideo), frame_threshold: Number(customFrame) } } : {}) }),
      });
      router.push(`/analysis/${analysis.id}`);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "提交失败");
      setPhase("idle");
    }
  };

  return (
    <div className="upload-workspace">
      <div className="workspace-heading"><AmbientBackdrop /><div className="heading-copy"><h1>智鉴</h1><p>伪造人脸检测系统</p><div className="heading-modalities"><span><ImageIcon size={13} />图片</span><span><Video size={13} />视频</span></div></div><MediaGraphic /></div>
      <section className="surface upload-panel">
        <div className="panel-heading"><h2><Upload size={16} />提交检测</h2><span>单文件 ≤ 256 MB</span></div>
        <div className="upload-body">
          <p className="mb-3 text-sm text-[var(--muted)]">检测模型：<strong className="font-semibold text-[var(--foreground)]">FADR-CLIP</strong><span className="ml-2 text-xs">最佳主模型权重 · 单模型检测</span></p>
          <fieldset className="submission-fieldset" disabled={phase !== "idle"}>
            <legend>计算设备</legend>
            <div className="device-options">
              {([
                { value: "cpu", label: "CPU", note: "处理器", available: devices?.cpu },
                { value: "cuda", label: "GPU", note: "NVIDIA 加速", available: devices?.gpu },
              ] as const).map(option => (
                <label key={option.value} className={`device-option ${device === option.value ? "is-selected" : ""} ${option.available ? "" : "is-unavailable"}`}>
                  <input type="radio" name="detection-device" value={option.value} aria-label={option.label} checked={device === option.value} disabled={!option.available} onChange={() => { deviceChosen.current = true; setDevice(option.value); }} />
                  <strong>{option.label}</strong><span>{devices && !option.available ? "暂不可用" : option.note}</span>
                </label>
              ))}
              <span className="device-status" role="status">{deviceError ? `设备状态读取失败：${deviceError}` : !devices ? "正在检查设备…" : !devices.worker_ready ? "检测服务准备中…" : !deviceAvailable ? "请选择可用设备" : device === "cuda" ? devices.gpu_name : "本次使用 CPU"}</span>
            </div>
          </fieldset>
          <fieldset className="submission-fieldset" disabled={phase !== "idle"}>
            <legend>判断模式</legend>
            <div className="mode-options">
              {thresholdModes.map(option => (
                <label key={option.mode} className={`mode-option ${thresholdMode === option.mode ? "is-selected" : ""}`}>
                  <input type="radio" name="threshold-mode" value={option.mode} aria-label={option.label} checked={thresholdMode === option.mode} onChange={() => setThresholdMode(option.mode)} />
                  <span><strong>{option.label}</strong>{option.mode === "balanced" && <small className="mode-default">默认</small>}
                    <span className="mode-values">{option.mode === "custom" ? "自行设置判断阈值" : <>视频 {formatThreshold(option.video_threshold)} · 图片 {formatThreshold(option.frame_threshold)}</>}</span>
                  </span>
                </label>
              ))}
            </div>
            {thresholdMode === "custom" && <div className="custom-thresholds">
              <label htmlFor="custom-video">视频均值阈值<input id="custom-video" type="number" min="0" max="1" step="0.000001" inputMode="decimal" value={customVideo} onChange={event => setCustomVideo(event.target.value)} aria-invalid={!validThreshold(customVideo)} aria-describedby="custom-threshold-help" /></label>
              <label htmlFor="custom-frame">图片／单帧阈值<input id="custom-frame" type="number" min="0" max="1" step="0.000001" inputMode="decimal" value={customFrame} onChange={event => setCustomFrame(event.target.value)} aria-invalid={!validThreshold(customFrame)} aria-describedby="custom-threshold-help" /></label>
              <span id="custom-threshold-help" role={customValid ? undefined : "alert"} className={customValid ? "" : "custom-error"}>{customValid ? "取值 0～1，最多 6 位小数" : "请输入 0～1 的数值，最多 6 位小数"}</span>
            </div>}
            {thresholdError ? <p className="submission-note custom-error" role="alert">判断模式加载失败：{thresholdError}<button type="button" className="ml-2 underline" onClick={() => setThresholdAttempt(value => value + 1)}>重试</button></p>
              : thresholdModes.length === 0 ? <p className="submission-note" role="status">正在加载判断模式…</p>
                : <p className="submission-note">{selectedMode?.description}。分数超过阈值判为疑似伪造，模式不改变模型分数。</p>}
          </fieldset>
          <p className="submission-note preprocessing-note">图片直接读取并对齐人脸；视频最多均匀抽取 32 帧。阈值为判断标准，分数不是已校准的伪造概率。</p>
          {!file ? (
            <div className={`upload-dropzone ${dragging ? "is-dragging" : ""}`} role="button" tabIndex={0} aria-label="选择媒体文件"
              onClick={() => inputRef.current?.click()}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); inputRef.current?.click(); } }}
              onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
              onDragOver={(event) => event.preventDefault()} onDragLeave={() => setDragging(false)} onDrop={onDrop}>
              <div className="upload-file-stack" aria-hidden="true"><span /><span /><span><FileUp size={27} strokeWidth={1.4} /></span></div>
              <h3 className="mt-3 text-base font-medium">拖放文件到此处</h3>
              <p className="mt-2 text-xs text-[var(--muted)]">图片或视频</p>
              <span className="secondary-button mt-3 text-sm">选择文件<ArrowRight size={14} /></span>
            </div>
          ) : (
            <div className="selected-file">
              <div className="flex w-full items-center gap-4"><span className="media-icon"><FileKindIcon type={fileKind(file) ?? "video"} /></span><div className="min-w-0 flex-1"><h3 className="truncate font-medium" title={file.name}>{file.name}</h3><p className="mt-1 text-xs text-[var(--muted)]">{file.type || "待识别格式"} · {formatBytes(file.size)}</p></div>{phase === "idle" && <button className="icon-button" title="移除文件" onClick={() => setFile(null)}><X size={18} /></button>}</div>
              {phase !== "idle" && <div className="mt-4 w-full" role="status"><div className="mb-3 flex justify-between text-xs"><span className="flex items-center gap-2">{phase === "uploading" ? <LoaderCircle className="animate-spin" size={14} /> : <CheckCircle2 size={14} />}{phase === "uploading" ? "正在上传" : "正在创建检测任务"}</span><span>{progress}%</span></div><div className="h-1 overflow-hidden rounded-full bg-[var(--surface-muted)]"><div className="h-full bg-[var(--primary)] transition-[width]" style={{ width: `${phase === "uploading" ? progress : 100}%` }} /></div></div>}
              <div className="mt-4 flex gap-3"><button className="secondary-button" disabled={phase !== "idle"} onClick={() => inputRef.current?.click()}>重新选择</button><button className="primary-button" disabled={phase !== "idle" || !deviceAvailable || thresholdModes.length === 0 || !customValid} onClick={() => void upload()}>开始检测<ArrowRight size={15} /></button></div>
            </div>
          )}
          {error && <div className="error-notice mt-4" role="alert"><AlertCircle size={16} />{error}</div>}
        </div>
        <div className="processing-flow" aria-label="检测流程">{[{ icon: Upload, label: "上传" }, { icon: RefreshCw, label: "人脸预处理" }, { icon: Cpu, label: "模型检测" }, { icon: FileCheck2, label: "分析报告" }].map(({ icon: Icon, label }, index) => <div key={label}><span className="flow-step"><Icon size={15} />{label}</span>{index < 3 && <ArrowRight className="flow-arrow" size={13} />}</div>)}</div>
      </section>
      <section className="formats-section" aria-labelledby="formats-title"><div className="mb-4 flex items-center justify-between"><h2 id="formats-title" className="text-sm font-medium">支持格式</h2><span className="text-xs text-[var(--muted)]">按论文准备人脸输入</span></div><div className="format-grid">{mediaCards.map(({ icon: Icon, title, formats, output, note }) => <article className="format-card" key={title}><div className={`format-art format-art-${output.toLowerCase()}`} aria-hidden="true">{Array.from({ length: 16 }, (_, i) => <i key={i} />)}</div><h3><Icon size={17} />{title}</h3><p className="format-list">{formats}</p><div className="format-output"><span className="font-mono">{output}</span><span>{note}</span></div></article>)}</div><p className="mt-3 text-[0.8125rem] leading-5 text-[var(--muted)]">图片与视频需包含清晰人脸。部分格式的可用性取决于文件编码。</p></section>
      <input ref={inputRef} className="hidden" type="file" accept="image/*,video/*,.mkv,.avi,.mov,.m4v,.heic,.heif,.tif,.tiff" onChange={onInput} />
    </div>
  );
}
