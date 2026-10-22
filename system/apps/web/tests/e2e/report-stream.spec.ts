import { expect, test } from "@playwright/test";

const analysisId = "11111111-1111-1111-1111-111111111111";
type StreamProbe = EventTarget & { closed: boolean };
declare global {
  interface Window { reportStreams: StreamProbe[] }
}

test("报告刷新复用 SSE，断线保留重连能力，结束后关闭连接", async ({ page }) => {
  let completed = false;
  let reportRequests = 0;
  await page.addInitScript(() => {
    window.reportStreams = [];
    class ProbeEventSource extends EventTarget {
      closed = false;
      onerror: ((event: Event) => void) | null = null;
      constructor() { super(); window.reportStreams.push(this); }
      close() { this.closed = true; }
      dispatchEvent(event: Event) {
        if (event.type === "error") this.onerror?.(event);
        return super.dispatchEvent(event);
      }
    }
    Object.defineProperty(window, "EventSource", { value: ProbeEventSource });
  });
  await page.route("**/api/v1/auth/me", (route) => route.fulfill({ json: {
    id: "user", email: "test@example.com", name: "测试用户", role: "analyst", is_active: true,
  } }));
  await page.route("**/api/v1/uploads/*/download-url", (route) => route.fulfill({ json: { url: "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7" } }));
  await page.route(`**/api/v1/analyses/${analysisId}`, (route) => {
    reportRequests++;
    return route.fulfill({ json: {
      id: analysisId, status: completed ? "completed" : "running", progress: completed ? 100 : 20 + reportRequests,
      verdict: completed ? "uncertain" : null, score: null, summary: null, cancel_requested: false,
      created_at: "2026-10-13T00:00:00Z", detector_runs: [],
      upload: { id: "upload", original_name: "face.png", media_type: "image", mime_type: "image/png", size_bytes: 10, sha256: "abc123", status: "validated" },
    } });
  });
  await page.goto(`/analysis/${analysisId}`);
  await expect.poll(() => page.evaluate(() => window.reportStreams.length)).toBe(1);
  const initialRequests = reportRequests;
  for (let index = 0; index < 3; index++) {
    await page.evaluate(() => window.reportStreams[0].dispatchEvent(new Event("detector.progress")));
    await expect.poll(() => reportRequests).toBeGreaterThan(initialRequests + index);
  }
  expect(await page.evaluate(() => window.reportStreams.length)).toBe(1);
  // Native EventSource must be allowed to reconnect using its Last-Event-ID.
  await page.evaluate(() => window.reportStreams[0].dispatchEvent(new Event("error")));
  expect(await page.evaluate(() => window.reportStreams[0].closed)).toBe(false);
  completed = true;
  await page.evaluate(() => window.reportStreams[0].dispatchEvent(new Event("analysis.completed")));
  await expect(page.getByText("总体结论", { exact: true })).toBeVisible();
  expect(await page.evaluate(() => window.reportStreams[0].closed)).toBe(true);
  expect(await page.evaluate(() => window.reportStreams.length)).toBe(1);
});
