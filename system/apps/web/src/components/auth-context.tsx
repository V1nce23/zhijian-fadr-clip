"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { usePathname, useRouter } from "next/navigation";

import { ApiError, apiFetch } from "@/lib/api";
import type { User } from "@/lib/types";

interface AuthContextValue {
  user: User;
  refreshUser: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const router = useRouter();
  const pathname = usePathname();

  const refreshUser = useCallback(async () => {
    try {
      const current = await apiFetch<User>("/auth/me");
      setUser(current);
      setError(null);
    } catch (requestError) {
      if (requestError instanceof ApiError && requestError.status === 401) {
        router.replace(`/login?next=${encodeURIComponent(pathname)}`);
        return;
      }
      setError(requestError instanceof Error ? requestError.message : "无法读取账号信息");
    } finally {
      setLoading(false);
    }
  }, [pathname, router]);

  useEffect(() => {
    const timer = window.setTimeout(() => void refreshUser(), 0);
    return () => window.clearTimeout(timer);
  }, [refreshUser]);

  const value = useMemo(
    () => (user ? { user, refreshUser } : null),
    [refreshUser, user],
  );

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-[var(--background)]">
        <div className="flex items-center gap-3 text-[var(--muted)]" role="status">
          <span className="h-4 w-4 animate-spin rounded-full border-2 border-[var(--border)] border-t-[var(--accent)]" />
          正在验证会话
        </div>
      </div>
    );
  }

  if (error || !value) {
    return (
      <div className="flex min-h-screen items-center justify-center px-6">
        <div className="surface max-w-md p-6 text-center">
          <p className="font-semibold text-[var(--danger)]">{error ?? "登录已失效"}</p>
          <button className="primary-button mt-5" onClick={() => void refreshUser()}>
            重试
          </button>
        </div>
      </div>
    );
  }

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth 必须在 AuthProvider 内使用");
  return context;
}
