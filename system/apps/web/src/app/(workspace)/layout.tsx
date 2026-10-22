import { AppShell } from "@/components/app-shell";
import { AuthProvider } from "@/components/auth-context";

export default function WorkspaceLayout({ children }: { children: React.ReactNode }) {
  return (
    <AuthProvider>
      <AppShell>{children}</AppShell>
    </AuthProvider>
  );
}
