import type { AnalysisStatus, MediaType, Verdict } from "@/lib/types";

export const statusLabels: Record<AnalysisStatus, string> = {
  created: "已创建",
  queued: "排队中",
  preprocessing: "预处理中",
  running: "检测中",
  aggregating: "汇总中",
  completed: "已完成",
  partially_completed: "部分完成",
  failed: "失败",
  cancelled: "已取消",
};

export const verdictLabels: Record<Verdict, string> = {
  likely_manipulated: "疑似伪造",
  uncertain: "结论不确定",
  likely_authentic: "疑似真实",
};

export const mediaLabels: Record<MediaType, string> = {
  image: "图片",
  video: "视频",
  audio: "已停用类型",
};

const detectorLabels: Record<string, string> = {
  "fadr-clip": "FADR-CLIP",
  "dfbench-xception": "Xception",
  "dfbench-efficientnetb4": "EfficientNet-B4",
  "dfbench-resnet34": "ResNet34 (CNN-Aug)",
  "dfbench-meso4": "Meso4",
  "dfbench-meso4Inception": "MesoInception4",
  "dfbench-capsule_net": "Capsule",
  "dfbench-core": "CORE",
  "dfbench-f3net": "F3Net",
  "dfbench-ffd": "FFD",
  "dfbench-recce": "RECCE",
  "dfbench-spsl": "SPSL",
  "dfbench-srm": "SRM",
  "dfbench-ucf": "UCF",
};

export function detectorLabel(detectorId: string): string {
  return detectorLabels[detectorId] ?? detectorId;
}

export const detectorRunStatusLabels: Record<string, string> = {
  queued: "排队中",
  running: "检测中",
  succeeded: "已完成",
  failed: "失败",
  skipped: "已跳过",
  timed_out: "超时",
  cancelled: "已取消",
};

export function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB"];
  let size = value / 1024;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(size >= 10 ? 1 : 2)} ${units[index]}`;
}

export function formatDate(value: string): string {
  return new Intl.DateTimeFormat("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(new Date(value));
}

export function shortHash(value: string): string {
  return `${value.slice(0, 10)}...${value.slice(-8)}`;
}

export function formatThreshold(value: number): string {
  return value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 6, useGrouping: false });
}
