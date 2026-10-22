import type { Metadata } from "next";
import { AdminUsers } from "@/components/admin-users";

export const metadata: Metadata = { title: "用户管理" };

export default function UsersPage() {
  return <AdminUsers />;
}
