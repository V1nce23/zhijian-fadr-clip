import type { Metadata } from "next";
import "./globals.css";
import { SiteBackground } from "@/components/site-background";
import { ThemeProvider } from "next-themes";

export const metadata: Metadata = {
  title: {
    default: "智鉴：伪造人脸检测系统",
    template: "%s | 智鉴",
  },
  description: "智鉴：伪造人脸检测系统，支持图片与视频智能检测。",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN" className="h-full" suppressHydrationWarning>
      <body className="min-h-full"><ThemeProvider attribute="data-theme" defaultTheme="system" storageKey="zhijian-theme" enableSystem disableTransitionOnChange><SiteBackground />{children}</ThemeProvider></body>
    </html>
  );
}
