/** Resolution-independent curves; palette follows the page theme. */
export function SiteBackground() {
  return (
    <div className="site-background" aria-hidden="true">
      <svg className="site-background-art" viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice" fill="none">
        <defs>
          <linearGradient id="background-sheet" x1="100" y1="400" x2="850" y2="1100" gradientUnits="userSpaceOnUse">
            <stop stopColor="var(--backdrop-edge)" />
            <stop offset="1" stopColor="var(--background)" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="background-fold" x1="0" y1="760" x2="760" y2="1060" gradientUnits="userSpaceOnUse">
            <stop stopColor="var(--backdrop-edge)" />
            <stop offset="1" stopColor="var(--background)" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="background-right" x1="1600" y1="120" x2="1170" y2="680" gradientUnits="userSpaceOnUse">
            <stop stopColor="var(--backdrop-edge)" />
            <stop offset="1" stopColor="var(--background)" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="background-rim" x1="0" y1="500" x2="700" y2="1040" gradientUnits="userSpaceOnUse">
            <stop stopColor="var(--backdrop-rim)" />
            <stop offset="1" stopColor="var(--backdrop-rim)" stopOpacity="0" />
          </linearGradient>
        </defs>
        <path d="M-80 510C170 420 283 540 340 694S558 869 916 1070H-80Z" fill="url(#background-sheet)" />
        <path d="M-80 510C170 420 283 540 340 694S558 869 916 1070" stroke="url(#background-rim)" strokeWidth="1.2" />
        <path d="M-100 710C190 628 452 754 726 1080H-100Z" fill="url(#background-fold)" />
        <path d="M-100 710C190 628 452 754 726 1080" stroke="url(#background-rim)" strokeWidth="1" />
        <path d="M1450-80C1300 197 1410 359 1570 468S1690 833 1400 1070H1700V-80Z" fill="url(#background-right)" />
        <path d="M1450-80C1300 197 1410 359 1570 468S1690 833 1400 1070" stroke="var(--backdrop-rim)" strokeOpacity=".5" />
      </svg>
    </div>
  );
}
