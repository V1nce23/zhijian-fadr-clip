"use client";

import { FormEvent, useCallback, useEffect, useState } from "react";
import { AlertCircle, Plus, Shield, UserRound } from "lucide-react";

import { apiFetch } from "@/lib/api";
import { formatDate } from "@/lib/format";
import type { User, UserRole } from "@/lib/types";

export function AdminUsers() {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ name: "", email: "", password: "", role: "analyst" as UserRole });

  const load = useCallback(async () => {
    try { setUsers(await apiFetch<User[]>("/users")); setError(null); }
    catch (requestError) { setError(requestError instanceof Error ? requestError.message : "加载失败"); }
    finally { setLoading(false); }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError(null);
    try {
      await apiFetch<User>("/users", { method: "POST", body: JSON.stringify(form) });
      setForm({ name: "", email: "", password: "", role: "analyst" });
      await load();
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : "创建失败");
    } finally { setSaving(false); }
  };

  return (
    <div>
      <div className="mb-6">

        <h1 className="text-2xl font-semibold sm:text-[2rem]">用户管理</h1>
      </div>
      {error && <div className="mb-4 flex items-center gap-2 rounded-[5px] border border-[var(--danger-border)] bg-[var(--danger-soft)] px-4 py-3 text-[var(--danger)]"><AlertCircle size={17} />{error}</div>}
      <div className="grid grid-cols-1 gap-5 lg:grid-cols-[360px_minmax(0,1fr)]">
        <form className="surface self-start p-5" onSubmit={submit}>
          <h2 className="flex items-center gap-2 font-semibold"><Plus size={17} className="text-[var(--accent)]" />创建账号</h2>
          <label className="mt-5 block text-sm font-medium">姓名<input className="field mt-2" value={form.name} onChange={(event) => setForm({ ...form, name: event.target.value })} required /></label>
          <label className="mt-4 block text-sm font-medium">邮箱<input className="field mt-2" type="email" value={form.email} onChange={(event) => setForm({ ...form, email: event.target.value })} required /></label>
          <label className="mt-4 block text-sm font-medium">初始密码<input className="field mt-2" type="password" minLength={10} value={form.password} onChange={(event) => setForm({ ...form, password: event.target.value })} required /></label>
          <label className="mt-4 block text-sm font-medium">角色<select className="field mt-2" value={form.role} onChange={(event) => setForm({ ...form, role: event.target.value as UserRole })}><option value="analyst">分析员</option><option value="admin">管理员</option></select></label>
          <button className="primary-button mt-6 w-full" disabled={saving}>{saving ? "正在创建" : "创建账号"}</button>
        </form>

        <section className="surface overflow-hidden">
          <div className="border-b border-[var(--border)] px-5 py-4"><h2 className="font-semibold">用户 <span className="ml-1 text-sm font-normal text-[var(--muted)]">{users.length}</span></h2></div>
          {loading ? <div className="p-5"><div className="skeleton h-48 rounded-[4px]" /></div> : (
            <div className="divide-y divide-[var(--border)]">
              {users.map((user) => (
                <div key={user.id} className="flex items-center gap-4 px-5 py-4">
                  <span className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-[5px] ${user.role === "admin" ? "bg-[var(--primary-soft)] text-[var(--accent)]" : "bg-[var(--surface-muted)] text-[var(--muted)]"}`}>{user.role === "admin" ? <Shield size={19} /> : <UserRound size={19} />}</span>
                  <div className="min-w-0 flex-1"><div className="truncate font-semibold">{user.name}</div><div className="mt-1 truncate text-xs text-[var(--muted)]">{user.email}</div></div>
                  <div className="text-right"><span className="rounded-[4px] bg-[var(--surface-muted)] px-2 py-1 text-xs">{user.role === "admin" ? "管理员" : "分析员"}</span><div className="mt-2 text-xs text-[var(--muted)]">{formatDate(user.created_at)}</div></div>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
