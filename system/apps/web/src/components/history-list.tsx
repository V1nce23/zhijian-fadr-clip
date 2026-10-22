"use client";

import Link from "next/link";
import { AlertCircle, CheckCircle2, ChevronRight, FileClock, LoaderCircle, RefreshCw, Trash2 } from "lucide-react";
import { useCallback, useEffect, useState } from "react";

import { apiFetch } from "@/lib/api";
import { formatBytes, formatDate, formatThreshold, mediaLabels, statusLabels, verdictLabels } from "@/lib/format";
import type { AnalysisListItem } from "@/lib/types";

const terminalStatuses = new Set(["completed", "partially_completed", "failed", "cancelled"]);

function HistoryActions({ item, deleting, disabled, onDelete }: {
  item: AnalysisListItem; deleting: boolean; disabled: boolean; onDelete: () => void;
}) {
  return <div className="flex items-center justify-start gap-2 sm:justify-end">
    <Link href={`/analysis/${item.id}`} className="history-details-button" aria-label={`查看详情：${item.upload.original_name}`}>查看详情<ChevronRight size={16} aria-hidden="true" /></Link>
    <button className="icon-button text-[var(--danger)] hover:bg-[var(--danger-soft)]" title="删除记录"
      aria-label={`删除记录：${item.upload.original_name}`} disabled={disabled} onClick={onDelete}>
      {deleting ? <LoaderCircle size={18} className="animate-spin" /> : <Trash2 size={18} />}
    </button>
  </div>;
}

export function HistoryList() {
  const [items, setItems] = useState<AnalysisListItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [batchDeleting, setBatchDeleting] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const busy = deletingId !== null || batchDeleting;

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const next = await apiFetch<AnalysisListItem[]>("/analyses");
      setItems(next);
      setSelectedIds(current => new Set([...current].filter(id => next.some(item => item.id === id))));
      setError(null);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const removeItem = async (item: AnalysisListItem) => {
    const active = !terminalStatuses.has(item.status);
    if (!window.confirm(`删除分析记录「${item.upload.original_name}」？${active ? "未完成的检测将停止，" : ""}报告和检测证据将被删除，源文件会保留。`)) return;
    setDeletingId(item.id);
    setError(null);
    setNotice(null);
    try {
      const result = await apiFetch<{ message: string }>(`/analyses/${item.id}`, { method: "DELETE" });
      setItems((current) => current.filter((entry) => entry.id !== item.id));
      setSelectedIds(current => new Set([...current].filter(id => id !== item.id)));
      setNotice(result.message);
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "删除失败");
    } finally {
      setDeletingId(null);
    }
  };

  const toggleSelected = (id: string) => {
    setSelectedIds((current) => {
      const next = new Set(current);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const toggleSelectAll = (checked: boolean) => {
    setSelectedIds(checked ? new Set(items.map((item) => item.id)) : new Set());
  };

  const batchDelete = async () => {
    const ids = [...selectedIds];
    if (ids.length === 0) return;
    if (!window.confirm(`确定删除选中的 ${ids.length} 条分析记录？未完成的检测将停止，报告和检测证据将被删除，源文件会保留。`)) return;
    setBatchDeleting(true);
    setError(null);
    setNotice(null);
    try {
      const result = await apiFetch<{ message: string; deleted: number; skipped: number }>("/analyses/batch-delete", {
        method: "POST",
        body: JSON.stringify({ analysis_ids: ids }),
      });
      setSelectedIds(new Set());
      setNotice(result.message);
      setError(null);
      if (result.skipped > 0) {
        // Some records were missing or inaccessible; refresh the current list.
        await load();
      } else {
        setItems((current) => current.filter((entry) => !ids.includes(entry.id)));
      }
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "批量删除失败");
    } finally {
      setBatchDeleting(false);
    }
  };

  return (
    <div>
      <div className="mb-6 flex items-end justify-between gap-3">
        <div>

          <h1 className="text-2xl font-semibold sm:text-[2rem]">分析记录</h1>
        </div>
        <button className="secondary-button" onClick={() => void load()} disabled={loading || busy} title="刷新记录">
          <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
          <span className="hidden sm:inline">刷新</span>
        </button>
      </div>

      {error && <div className="mb-4 flex items-center gap-2 rounded-[5px] border border-[var(--danger-border)] bg-[var(--danger-soft)] px-4 py-3 text-[var(--danger)]"><AlertCircle size={17} />{error}</div>}
      {notice && <div className="mb-4 flex items-center gap-2 rounded-[5px] border border-[var(--success-border)] bg-[var(--success-soft)] px-4 py-3 text-[var(--success)]"><CheckCircle2 size={17} />{notice}</div>}

      {selectedIds.size > 0 && (
        <div className="mb-4 flex flex-wrap items-center justify-between gap-3 rounded-[5px] border border-[var(--border)] bg-[var(--primary-soft)] px-4 py-3">
          <span className="text-sm font-medium text-[var(--foreground)]">已选 {selectedIds.size} 项</span>
          <div className="flex items-center gap-2">
            <button className="secondary-button" disabled={busy || loading} onClick={() => setSelectedIds(new Set())}>取消选择</button>
            <button className="danger-button" disabled={busy || loading} onClick={() => void batchDelete()}>
              {batchDeleting ? <LoaderCircle size={16} className="animate-spin" /> : <Trash2 size={16} />}删除所选
            </button>
          </div>
        </div>
      )}

      <div className="surface overflow-hidden">
        {loading && items.length === 0 ? (
          <div className="space-y-3 p-5" role="status">
            {[1, 2, 3].map((item) => <div key={item} className="skeleton h-16 rounded-[4px]" />)}
          </div>
        ) : items.length === 0 ? (
          <div className="flex min-h-72 flex-col items-center justify-center px-5 text-center">
            <FileClock size={36} className="text-[var(--muted)]" strokeWidth={1.6} />
            <h2 className="mt-4 font-semibold">暂无分析记录</h2>
            <Link href="/" className="mt-4 text-sm font-semibold text-[var(--accent)]">提交第一个文件</Link>
          </div>
        ) : (
          <div className="relative overflow-x-auto">
            <table className="w-full min-w-[860px] border-collapse text-left">
              <thead className="bg-[var(--surface-muted)] text-xs font-semibold text-[var(--muted)]">
                <tr>
                  <th className="w-12 border-b border-[var(--border)] px-4 py-3">
                    <input
                      type="checkbox"
                      aria-label="选择全部"
                      disabled={busy || loading}
                      className="h-4 w-4 accent-[var(--accent)]"
                      checked={items.length > 0 && selectedIds.size === items.length}
                      ref={(element) => {
                        if (element) element.indeterminate = selectedIds.size > 0 && selectedIds.size < items.length;
                      }}
                      onChange={(event) => toggleSelectAll(event.target.checked)}
                    />
                  </th>
                  <th className="border-b border-[var(--border)] px-5 py-3">文件</th>
                  <th className="whitespace-nowrap border-b border-[var(--border)] px-4 py-3">类型</th>
                  <th className="border-b border-[var(--border)] px-4 py-3">设备</th>
                  <th className="border-b border-[var(--border)] px-4 py-3">状态</th>
                  <th className="border-b border-[var(--border)] px-4 py-3">结论</th>
                  <th className="border-b border-[var(--border)] px-4 py-3">提交时间</th>
                  <th className="hidden w-44 border-b border-[var(--border)] px-4 py-3 text-right sm:table-cell">操作</th>
                </tr>
              </thead>
              <tbody>
                {items.map((item) => (
                  <tr key={item.id} className="border-b border-[var(--border)] last:border-0 hover:bg-[var(--surface-muted)]">
                    <td className="px-4 py-4">
                      <input
                        type="checkbox"
                        aria-label={`选择 ${item.upload.original_name}`}
                        disabled={busy || loading}
                        className="h-4 w-4 accent-[var(--accent)]"
                        checked={selectedIds.has(item.id)}
                        onChange={() => toggleSelected(item.id)}
                      />
                    </td>
                    <td className="max-w-[320px] px-5 py-4">
                      <Link href={`/analysis/${item.id}`} title={item.upload.original_name} className="block max-w-[200px] truncate font-semibold text-[var(--foreground)] no-underline hover:text-[var(--accent)] sm:max-w-[220px] 2xl:max-w-[320px]">{item.upload.original_name}</Link>
                      <span className="mt-1 block text-xs text-[var(--muted)]">{formatBytes(item.upload.size_bytes)}</span>
                      <div className="mt-2 sm:hidden"><HistoryActions item={item} deleting={deletingId === item.id} disabled={busy || loading} onDelete={() => void removeItem(item)} /></div>
                    </td>
                    <td className="whitespace-nowrap px-4 py-4 text-sm">{mediaLabels[item.upload.media_type]}</td>
                    <td className="px-4 py-4 text-sm" title={item.executed_device ? "实际执行设备" : "所选设备"}>{(item.executed_device ?? item.requested_device) === "cuda" ? "GPU" : (item.executed_device ?? item.requested_device) === "cpu" ? "CPU" : "未记录"}</td>
                    <td className="px-4 py-4"><span className="whitespace-nowrap rounded-[4px] bg-[var(--surface-muted)] px-2 py-1 text-xs font-medium">{statusLabels[item.status]}{item.status === "running" ? ` ${item.progress}%` : ""}</span></td>
                    <td className="px-4 py-4 text-sm font-medium">{item.verdict ? verdictLabels[item.verdict] : "—"}<span className="mt-1 block whitespace-nowrap text-xs font-normal text-[var(--muted)]">{item.threshold_policy ? `${item.threshold_policy.label} · 阈值 ${formatThreshold(item.threshold_policy.threshold)}` : "历史判断规则"}</span></td>
                    <td className="px-4 py-4 text-sm text-[var(--muted)]">{formatDate(item.created_at)}</td>
                    <td className="hidden px-3 py-4 sm:table-cell">
                      <HistoryActions item={item} deleting={deletingId === item.id} disabled={busy || loading} onDelete={() => void removeItem(item)} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
