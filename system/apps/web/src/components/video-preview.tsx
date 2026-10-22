"use client";

import { useEffect, useState } from "react";
import { LoaderCircle, RefreshCw } from "lucide-react";
import { apiFetch } from "@/lib/api";

type Preview = { status: "pending" | "ready" | "failed"; url?: string; detail?: string };

export function VideoPreview({ uploadId, originalUrl, name }: { uploadId: string; originalUrl: string; name: string }) {
  const [preview, setPreview] = useState<Preview>({ status: "pending" });
  const [attempt, setAttempt] = useState(0);
  const [failedUrl, setFailedUrl] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timer: number | undefined;
    let firstRequest = true;
    const refresh = async () => {
      try {
        const next = await apiFetch<Preview>(`/uploads/${uploadId}/preview-url`, {
          signal: controller.signal,
          method: firstRequest && attempt > 0 ? "POST" : "GET",
        });
        firstRequest = false;
        if (controller.signal.aborted) return;
        setPreview(next);
        if (next.status !== "failed") timer = window.setTimeout(() => void refresh(), next.status === "pending" ? 2500 : 240_000);
      } catch (error) {
        if (!controller.signal.aborted) setPreview({ status: "failed", detail: error instanceof Error ? error.message : "视频预览暂时无法加载" });
      }
    };
    void refresh();
    return () => { controller.abort(); window.clearTimeout(timer); };
  }, [uploadId, attempt]);

  if (preview.status === "pending") return <div className="flex min-h-40 flex-col items-center justify-center gap-3 px-4 py-8 text-sm text-[var(--muted)]" role="status"><LoaderCircle size={22} className="animate-spin" /><p>正在准备视频预览…</p><p className="text-xs">可先查看检测结果，预览会自动显示。</p></div>;

  if (preview.status === "failed" || !preview.url || failedUrl === preview.url) {
    return <div className="py-8 text-center text-sm text-[var(--muted)]"><p>{preview.detail || "视频加载失败，请重试。"}</p><div className="mt-4 flex flex-wrap items-center justify-center gap-4"><button className="secondary-button" onClick={() => { setFailedUrl(null); setPreview({ status: "pending" }); setAttempt((value) => value + 1); }}><RefreshCw size={15} />重试预览</button><a className="font-medium text-[var(--accent)]" href={originalUrl} target="_blank" rel="noreferrer">下载原文件</a></div></div>;
  }

  return <video key={preview.url} src={preview.url} controls playsInline preload="metadata" aria-label={`${name} 视频预览`} onError={() => setFailedUrl(preview.url ?? null)} className="block max-h-[420px] w-full rounded-xl border border-[var(--border)] bg-black" />;
}
