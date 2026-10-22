import Image from "next/image";

export function BrandMark({ size = 40 }: { size?: number }) {
  return <Image src="/brand-mark.svg" alt="智鉴标识" width={size} height={size} className="brand-mark" priority />;
}
