"use client";

import { useId } from "react";

/** Original vector illustration; its geometry and colors follow the active theme. */
export function MediaGraphic() {
  const id = useId().replaceAll(":", "");
  const fill = (name: string) => `url(#${id}-${name})`;
  return (
    <svg className="media-graphic" viewBox="0 0 600 380" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id={`${id}-base`} x1="140" y1="220" x2="450" y2="340" gradientUnits="userSpaceOnUse"><stop stopColor="var(--scene-white)" /><stop offset="1" stopColor="var(--scene-mid)" /></linearGradient>
        <linearGradient id={`${id}-edge`} x1="100" y1="275" x2="510" y2="310" gradientUnits="userSpaceOnUse"><stop stopColor="var(--scene-mid)" /><stop offset=".45" stopColor="var(--scene-white)" /><stop offset="1" stopColor="var(--scene-blue)" /></linearGradient>
        <linearGradient id={`${id}-glass`} x1="190" y1="80" x2="395" y2="265" gradientUnits="userSpaceOnUse"><stop stopColor="var(--scene-white)" stopOpacity=".92" /><stop offset=".55" stopColor="var(--scene-blue)" stopOpacity=".3" /><stop offset="1" stopColor="var(--scene-blue)" stopOpacity=".72" /></linearGradient>
        <linearGradient id={`${id}-core`} x1="248" y1="95" x2="380" y2="200" gradientUnits="userSpaceOnUse"><stop stopColor="var(--scene-core-top)" /><stop offset=".6" stopColor="var(--scene-core)" /><stop offset="1" stopColor="var(--scene-core-side)" /></linearGradient>
        <linearGradient id={`${id}-card`} x1="0" y1="0" x2="1" y2="1"><stop stopColor="var(--scene-white)" stopOpacity=".95" /><stop offset="1" stopColor="var(--scene-mid)" stopOpacity=".8" /></linearGradient>
        <radialGradient id={`${id}-shadow`}><stop stopColor="var(--scene-shadow)" stopOpacity=".18" /><stop offset="1" stopColor="var(--scene-shadow)" stopOpacity="0" /></radialGradient>
        <radialGradient id={`${id}-halo`}><stop stopColor="var(--scene-blue)" stopOpacity=".06" /><stop offset="1" stopColor="var(--scene-blue)" stopOpacity="0" /></radialGradient>
      </defs>
      <ellipse cx="307" cy="217" rx="230" ry="155" fill={fill("halo")} />
      <ellipse cx="304" cy="324" rx="246" ry="47" fill={fill("shadow")} />
      <g className="scene-traces" stroke="var(--scene-line)" strokeWidth="1.2">
        <path d="M10 252L162 335Q180 345 207 334L474 210L590 267" /><path d="M6 269L157 351Q180 364 209 350L478 226L597 283" opacity=".5" />
        <path d="M60 177L202 246M405 314L547 239M432 182L525 135" />
        <circle cx="60" cy="177" r="3" /><circle cx="525" cy="135" r="3" /><circle cx="547" cy="239" r="3" />
      </g>
      {/* The platform has separate top and side faces, rather than a flat panel. */}
      <path d="M102 261L300 171L501 261V281Q501 285 495 288L315 368Q302 374 289 368L108 287Q102 284 102 278Z" fill={fill("edge")} />
      <path d="M109 251L288 170Q300 164 313 170L494 251Q507 257 494 263L313 345Q300 351 288 345L109 264Q95 257 109 251Z" fill={fill("base")} stroke="var(--scene-rim)" />
      <path d="M118 276L288 353Q301 359 314 353L489 274" stroke="var(--scene-light)" strokeWidth="1" />
      <ellipse cx="302" cy="260" rx="127" ry="51" fill="var(--scene-mid)" fillOpacity=".5" />
      <path d="M188 246V258C188 285 239 306 302 306C365 306 416 285 416 258V246" fill={fill("edge")} />
      <ellipse cx="302" cy="246" rx="114" ry="47" fill={fill("base")} stroke="var(--scene-rim)" />
      <ellipse cx="302" cy="246" rx="95" ry="37" stroke="var(--scene-light)" strokeWidth="1.4" />
      <ellipse cx="302" cy="246" rx="73" ry="27" fill={fill("halo")} />
      {/* Floating glass frame and solid blue center. */}
      <g className="scene-center">
        <path d="M218 86L332 56L393 87V218L272 253L218 221Z" fill={fill("glass")} stroke="var(--scene-rim)" strokeWidth="1.2" />
        <path d="M218 86L272 113L393 87M272 113V253" stroke="var(--scene-white)" strokeOpacity=".5" />
        <path d="M242 96L329 74L371 96L283 120Z" fill="var(--scene-core-top)" />
        <path d="M283 120L371 96V202L283 229Z" fill={fill("core")} />
        <path d="M242 96L283 120V229L242 207Z" fill="var(--scene-core-side)" />
        <path d="M288 126L365 104V196L288 219Z" stroke="#bbd9ff" strokeOpacity=".5" />
        <g transform="matrix(.66 -.18 0 .68 298 135)" stroke="#e1f0ff" strokeWidth="1.4" strokeLinejoin="round">
          <path d="M13 15L32 2H56L76 15L83 49L76 81L58 103H31L13 81L6 49Z" />
          <path d="M13 15L44 27L76 15M6 49L27 43L44 27L62 43L83 49M13 81L31 69H59L76 81M31 103L44 88L58 103M27 43L31 69L44 88L59 69L62 43M44 27V59L31 69M44 59L59 69" />
          {[[13,15],[76,15],[44,27],[27,43],[62,43],[31,69],[59,69],[44,88]].map(([cx,cy]) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="2.8" fill="#eff8ff" stroke="none" />)}
        </g>
        <path d="M282 164L373 140" stroke="#ecfaff" strokeWidth="2" className="scene-scan" />
      </g>
      {/* Media cards use icons only: no artificial scores or status data. */}
      <g transform="matrix(.95 .28 0 1 110 105)">
        <rect x="5" y="5" width="78" height="79" rx="9" fill="var(--scene-blue)" fillOpacity=".3" />
        <rect width="78" height="79" rx="9" fill={fill("card")} stroke="var(--scene-rim)" />
        <rect x="17" y="18" width="44" height="36" rx="5" stroke="var(--scene-icon)" strokeWidth="2.5" />
        <path d="M18 46L31 32L41 42L48 35L61 48" stroke="var(--scene-icon)" strokeWidth="2.5" /><circle cx="48" cy="27" r="3" fill="var(--scene-icon)" />
        <path d="M20 65H57" stroke="var(--scene-line)" strokeWidth="3" strokeLinecap="round" />
      </g>
      <g transform="matrix(1 -.26 0 1 426 128)">
        <rect x="5" y="5" width="96" height="64" rx="8" fill="var(--scene-blue)" fillOpacity=".25" />
        <rect width="96" height="64" rx="8" fill={fill("card")} stroke="var(--scene-rim)" />
        <rect x="17" y="14" width="43" height="34" rx="5" fill="var(--scene-icon)" />
        <path d="M35 22L48 31L35 40Z" fill="var(--scene-white)" /><path d="M65 23L79 17V46L65 40Z" fill="var(--scene-icon)" />
      </g>
      <g fill="var(--scene-light)" opacity=".35"><circle cx="162" cy="80" r="4" /><circle cx="418" cy="78" r="3" /><circle cx="468" cy="259" r="3" /></g>
    </svg>
  );
}
