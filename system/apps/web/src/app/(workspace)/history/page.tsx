import type { Metadata } from "next";
import { HistoryList } from "@/components/history-list";

export const metadata: Metadata = { title: "分析记录" };

export default function HistoryPage() {
  return <HistoryList />;
}
