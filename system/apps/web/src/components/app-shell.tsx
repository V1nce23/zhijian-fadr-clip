"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { ChevronRight, History, LogOut, Menu, Upload, Users, X } from "lucide-react";
import { useState } from "react";
import { apiFetch } from "@/lib/api";
import { useAuth } from "@/components/auth-context";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { BrandMark } from "@/components/brand-mark";

const navItems = [
  { href: "/", label: "提交检测", icon: Upload },
  { href: "/history", label: "分析记录", icon: History },
];

export function AppShell({ children }: { children: React.ReactNode }) {
  const { user } = useAuth();
  const pathname = usePathname();
  const router = useRouter();
  const [menuOpen, setMenuOpen] = useState(false);
  const [logoutError, setLogoutError] = useState<string | null>(null);
  const logout = async () => {
    try {
      await apiFetch("/auth/logout", { method: "POST" });
      router.replace("/login");
      router.refresh();
    } catch { setLogoutError("退出失败，请重试"); }
  };
  const items = user.role === "admin" ? [...navItems, { href: "/admin/users", label: "用户管理", icon: Users }] : navItems;
  const pageName = pathname.startsWith("/analysis/") ? "分析报告" : items.find((item) => item.href === pathname)?.label ?? "智鉴";

  return (
    <div className="app-frame">
      <header className="mobile-header">
        <Link href="/" className="flex items-center gap-3 no-underline"><BrandMark size={32} /><strong className="text-lg">智鉴</strong></Link>
        <div className="flex items-center gap-2"><ThemeSwitcher /><button className="icon-button" aria-label={menuOpen ? "关闭菜单" : "打开菜单"} aria-expanded={menuOpen} aria-controls="workspace-sidebar" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={20} /> : <Menu size={20} />}</button></div>
      </header>
      {menuOpen && <button className="sidebar-backdrop" aria-label="关闭导航" onClick={() => setMenuOpen(false)} />}
      <aside id="workspace-sidebar" className={`workspace-sidebar ${menuOpen ? "is-open" : ""}`}>
        <Link href="/" className="sidebar-brand" onClick={() => setMenuOpen(false)}><BrandMark size={40} /><div><strong className="text-xl tracking-wider">智鉴</strong><p className="mt-1 text-[0.75rem] text-[var(--muted)]">伪造人脸检测系统</p></div></Link>
        <nav className="mt-9 flex flex-col gap-1" aria-label="主导航">
          {items.map(({ href, label, icon: Icon }) => {
            const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
            return <Link key={href} href={href} aria-current={active ? "page" : undefined} className={`sidebar-link ${active ? "is-active" : ""}`} onClick={() => setMenuOpen(false)}><Icon size={17} strokeWidth={1.7} />{label}</Link>;
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="mb-5 border-b border-[var(--border)] pb-5 text-[0.8125rem] leading-6 text-[var(--muted)]">伪造人脸检测系统</div>
          <div className="flex items-center gap-2"><span className="user-avatar">{user.name.slice(0, 1)}</span><div className="min-w-0 flex-1"><p className="truncate text-xs font-medium">{user.name}</p><p className="mt-1 text-[0.75rem] text-[var(--muted)]">{user.role === "admin" ? "管理员" : "分析员"}</p></div><button className="icon-button" title="退出登录" onClick={() => void logout()}><LogOut size={16} /></button></div>
          {logoutError && <p className="mt-2 text-xs text-[var(--danger)]" role="alert">{logoutError}</p>}
        </div>
      </aside>
      <div className="workspace-content">
        <div className="workspace-toolbar"><span>智鉴</span><ChevronRight size={13} /><span className="text-[var(--foreground)]">{pageName}</span><div className="ml-auto"><ThemeSwitcher /></div></div>
        <main className="workspace-main">{children}</main>
        <footer className="workspace-footer"><span>智鉴</span><span>检测结果仅供辅助研判</span></footer>
      </div>
    </div>
  );
}
