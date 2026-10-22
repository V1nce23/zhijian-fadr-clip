/** Subtle contour lines for the workspace banner. */
export function AmbientBackdrop() {
  return (
    <svg className="ambient-backdrop" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMid slice" fill="none" aria-hidden="true">
      <path className="ambient-ribbon" d="M-200 850C130 240 420 1050 880 575S1370 250 1800-100" strokeWidth="155" />
      <path className="ambient-ribbon" d="M-200 850C130 240 420 1050 880 575S1370 250 1800-100" strokeWidth="1" />
      <g className="ambient-lines">
        {Array.from({ length: 4 }, (_, i) => <path key={i} d={`M-100 ${570+i*12}C340 ${850+i*10} 580 ${610+i*10} 850 ${450+i*9}S1370 ${210+i*11} 1700 ${330+i*10}`} />)}
      </g>
      <ellipse className="ambient-orbit" cx="920" cy="700" rx="530" ry="155" transform="rotate(-20 920 700)" />
    </svg>
  );
}
