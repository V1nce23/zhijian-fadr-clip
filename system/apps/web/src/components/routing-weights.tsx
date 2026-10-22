type RoutingEvidence = { layers: number[]; weights: number[]; origin?: string };

function validRouting(weights: unknown, layers: unknown): weights is number[] {
  return Array.isArray(weights) && weights.length === 4 && weights.every(value => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1)
    && Array.isArray(layers) && layers.join(",") === "6,12,18,24" && Math.abs(weights.reduce((sum, value) => sum + value, 0) - 1) < 0.0001;
}

export function RoutingWeights({ weights, layers, title, origin, compact = false, emptyText = "此记录未保存可用的路由权重。" }: {
  weights: unknown; layers: unknown; title: string; origin?: string; compact?: boolean; emptyText?: string;
}) {
  const valid = validRouting(weights, layers);
  const values = valid ? weights : [];
  const highest = values.length ? values.indexOf(Math.max(...values)) : -1;
  return <div className={`routing-weights ${compact ? "is-compact" : ""}`} aria-label={title}>
    <div className="routing-heading"><h3>{title}</h3>{valid && <span>最高 L{[6, 12, 18, 24][highest]}</span>}</div>
    {valid ? <>
      <div className="routing-bars" role="img" aria-label={values.map((weight, index) => `L${[6, 12, 18, 24][index]} ${(weight * 100).toFixed(2)}%`).join("，")}>
        {values.map((weight, index) => <div key={index} className={`routing-row ${highest === index ? "is-highest" : ""}`}><span>L{[6, 12, 18, 24][index]}</span><span className="routing-track"><i style={{ width: `${weight * 100}%` }} /></span><strong>{(weight * 100).toFixed(2)}%</strong></div>)}
      </div>
      <p className="routing-note">{origin === "supplementary_replay" ? "补充生成的路由数据，原检测结果保持不变。" : "来自原检测记录。"}权重表示层特征融合比例，不是伪造概率或准确率。</p>
    </> : <p className="routing-note">{emptyText}</p>}
  </div>;
}

export type { RoutingEvidence };
