"use client";

import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { ArrowRight } from "lucide-react";

import { apiFetch } from "@/lib/api";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { MediaGraphic } from "@/components/media-graphic";
import { BrandMark } from "@/components/brand-mark";
import type { User } from "@/lib/types";

export default function LoginPage() {
  const router = useRouter();
  const [email, setEmail] = useState("admin@deeptrace.example.com");
  const [password, setPassword] = useState("DeepTrace123!");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await apiFetch<User>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      const next = new URLSearchParams(window.location.search).get("next");
      router.replace(next?.startsWith("/") ? next : "/");
      router.refresh();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "登录失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="login-page">
      <div className="login-theme"><ThemeSwitcher /></div>
      <section className="login-identity">
        <div className="flex items-center gap-3"><BrandMark size={46} /><span className="text-sm tracking-widest">伪造人脸检测系统</span></div>
        <div className="login-title"><h1>智鉴</h1><p>伪造人脸检测系统</p>
          <div className="login-graphic"><MediaGraphic /></div>
        </div>
      </section>
      <section className="login-access">
        <div className="login-card">
          <div className="mb-10 flex items-center gap-3 lg:hidden"><BrandMark size={40} /><div><strong className="text-xl">智鉴</strong><p className="mt-1 text-xs text-[var(--muted)]">伪造人脸检测系统</p></div></div>
          <h2 className="mb-8 text-2xl font-semibold">登录</h2>
          <form onSubmit={submit}>
            <label className="block text-xs font-medium" htmlFor="email">邮箱</label>
            <input id="email" className="field mt-2.5" type="email" autoComplete="username" value={email} onChange={(event) => setEmail(event.target.value)} required />
            <label className="mt-5 block text-xs font-medium" htmlFor="password">密码</label>
            <input id="password" className="field mt-2.5" type="password" autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} required />
            {error && <div className="error-notice mt-4" role="alert">{error}</div>}
            <button className="primary-button mt-7 w-full" type="submit" disabled={loading}>{loading ? "正在登录" : "登录"}{!loading && <ArrowRight size={16} />}</button>
          </form>
          <p className="mt-6 text-center text-xs text-[var(--muted)]">账号由管理员创建</p>
        </div>
      </section>
    </main>
  );
}
