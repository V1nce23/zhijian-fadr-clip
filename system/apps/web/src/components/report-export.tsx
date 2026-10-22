"use client";

import { useState } from "react";
import { Download, LoaderCircle } from "lucide-react";

export function ReportExport({ analysisId, enabled }: { analysisId: string; enabled: boolean }) {
  const [format, setFormat] = useState("pdf");
  const [includeTechnical, setIncludeTechnical] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const download = async () => {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`/api/v1/analyses/${analysisId}/export?format=${format}&include_technical=${includeTechnical}`, { credentials: "include", cache: "no-store" });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "导出失败，请重试");
      }
      const url = URL.createObjectURL(await response.blob());
      const link = document.createElement("a");
      link.href = url;
      link.download = `zhijian-report-${analysisId}.${format}`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      // Keep the Blob alive long enough for browsers to start the download.
      window.setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (error) { setError(error instanceof Error ? error.message : "导出失败，请重试"); }
    finally { setBusy(false); }
  };
  return <div className="report-export">
    <div className="report-export-controls"><select aria-label="报告导出格式" value={format} onChange={event => setFormat(event.target.value)} disabled={busy || !enabled}>
      <option value="pdf">检测报告 PDF</option><option value="docx">可编辑报告 Word</option><option value="html">检测报告 HTML</option><option value="csv">逐帧数据 CSV</option><option value="json">报告数据 JSON</option>
    </select><button className="secondary-button" onClick={() => void download()} disabled={busy || !enabled} title={enabled ? "下载已有记录，不重新检测" : "任务结束后可导出报告"}>{busy ? <LoaderCircle size={16} className="animate-spin" /> : <Download size={16} />}{busy ? "导出中" : "导出报告"}</button></div>
    <label className="report-export-technical"><input type="checkbox" checked={includeTechnical} onChange={event => setIncludeTechnical(event.target.checked)} disabled={busy || !enabled} />包含技术详情<span>版本与层权重</span></label>
    {error && <p className="report-export-error" role="alert">{error}</p>}
  </div>;
}
