import type { Metadata } from "next";
import { AnalysisView } from "@/components/analysis-view";

export const metadata: Metadata = { title: "分析报告" };

export default async function AnalysisPage(props: PageProps<"/analysis/[id]">) {
  const { id } = await props.params;
  return <AnalysisView analysisId={id} />;
}
