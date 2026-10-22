import { expect, test } from "@playwright/test";

const adminEmail = process.env.E2E_ADMIN_EMAIL ?? "admin@deeptrace.example.com";
const adminPassword = process.env.E2E_ADMIN_PASSWORD ?? "DeepTrace123!";
const pngFixture = Buffer.from(
  "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
  "base64",
);
const webmFixture = Buffer.from(
  "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQJChYECGFOAZwEAAAAAAAHs" +
    "EU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHYTbuMU6uEElTDZ1OsggEl" +
    "TbuMU6uEHFO7a1OsggHW7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" +
    "AAAAAAAAAAAAAAAAAAAAAAAVSalmsirXsYMPQkBNgI1MYXZmNjIuMTIuMTAxV0GN" +
    "TGF2ZjYyLjEyLjEwMUSJiEBpAAAAAAAAFlSua8iuAQAAAAAAAD/XgQFzxYgenBbv" +
    "0WygSpyBACK1nIN1bmSIgQCGhVZfVlA5g4EBI+ODhAvrwgDgkLCBILqBIJqBAlWw" +
    "hFW5gQESVMNnQIBzc6BjwIBnyJpFo4dFTkNPREVSRIeNTGF2ZjYyLjEyLjEwMXNz" +
    "2mPAi2PFiB6cFu/RbKBKZ8ilRaOHRU5DT0RFUkSHmExhdmM2Mi4yOC4xMDEgbGli" +
    "dnB4LXZwOWfIoUWjiERVUkFUSU9ORIeTMDA6MDA6MDAuMjAwMDAwMDAwAB9DtnWm" +
    "54EAo6GBAACAgkmDQgAB8AH2ADgkHBhKAAAwYAAAEN///VkVgAAcU7trkbuPs4EA" +
    "t4r3gQHxggGr8IED",
  "base64",
);
const mp4Fixture = Buffer.from(
  "AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAMVbW9vdgAAAGxtdmhkAAAAAAAAAAAAAAAAAAAD6AAAAMgAAQAAAQAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAgAAAj90cmFrAAAAXHRraGQAAAADAAAAAAAAAAAAAAABAAAAAAAAAMgAAAAAAAAAAAAAAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAABAAAAAABAAAAAQAAAAAAAkZWR0cwAAABxlbHN0AAAAAAAAAAEAAADIAAAAAAABAAAAAAG3bWRpYQAAACBtZGhkAAAAAAAAAAAAAAAAAAAoAAAACABVxAAAAAAALWhkbHIAAAAAAAAAAHZpZGUAAAAAAAAAAAAAAABWaWRlb0hhbmRsZXIAAAABYm1pbmYAAAAUdm1oZAAAAAEAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAASJzdGJsAAAAvnN0c2QAAAAAAAAAAQAAAK5hdmMxAAAAAAAAAAEAAAAAAAAAAAAAAAAAAAAAABAAEABIAAAASAAAAAAAAAABFUxhdmM2Mi4yOC4xMDEgbGlieDI2NAAAAAAAAAAAAAAAGP//AAAANGF2Y0MBZAAK/+EAF2dkAAqs2V7ARAAAAwAEAAADACg8SJZYAQAGaOvjyyLA/fj4AAAAABBwYXNwAAAAAQAAAAEAAAAUYnRydAAAAAAAAGxwAAAAAAAAABhzdHRzAAAAAAAAAAEAAAABAAAIAAAAABxzdHNjAAAAAAAAAAEAAAABAAAAAQAAAAEAAAAUc3RzegAAAAAAAAK2AAAAAQAAABRzdGNvAAAAAAAAAAEAAANFAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY2Mi4xMi4xMDEAAAAIZnJlZQAAAr5tZGF0AAACnwYF//+b3EXpvebZSLeWLNgg2SPu73gyNjQgLSBjb3JlIDE2NSAtIEguMjY0L01QRUctNCBBVkMgY29kZWMgLSBDb3B5bGVmdCAyMDAzLTIwMjUgLSBodHRwOi8vd3d3LnZpZGVvbGFuLm9yZy94MjY0Lmh0bWwgLSBvcHRpb25zOiBjYWJhYz0xIHJlZj0zIGRlYmxvY2s9MTowOjAgYW5hbHlzZT0weDM6MHgxMTMgbWU9aGV4IHN1Ym1lPTcgcHN5PTEgcHN5X3JkPTEuMDA6MC4wMCBtaXhlZF9yZWY9MSBtZV9yYW5nZT0xNiBjaHJvbWFfbWU9MSB0cmVsbGlzPTEgOHg4ZGN0PTEgY3FtPTAgZGVhZHpvbmU9MjEsMTEgZmFzdF9wc2tpcD0xIGNocm9tYV9xcF9vZmZzZXQ9LTIgdGhyZWFkcz0xIGxvb2thaGVhZF90aHJlYWRzPTEgc2xpY2VkX3RocmVhZHM9MCBucj0wIGRlY2ltYXRlPTEgaW50ZXJsYWNlZD0wIGJsdXJheV9jb21wYXQ9MCBjb25zdHJhaW5lZF9pbnRyYT0wIGJmcmFtZXM9MyBiX3B5cmFtaWQ9MiBiX2FkYXB0PTEgYl9iaWFzPTAgZGlyZWN0PTEgd2VpZ2h0Yj0xIG9wZW5fZ29wPTAgd2VpZ2h0cD0yIGtleWludD0yNTAga2V5aW50X21pbj01IHNjZW5lY3V0PTQwIGludHJhX3JlZnJlc2g9MCByY19sb29rYWhlYWQ9NDAgcmM9Y3JmIG1idHJlZT0xIGNyZj0yMy4wIHFjb21wPTAuNjAgcXBtaW49MCBxcG1heD02OSBxcHN0ZXA9NCBpcF9yYXRpbz0xLjQwIGFxPTE6MS4wMACAAAAAD2WIhAA///73aJ8Cm15huQ==",
  "base64",
);

function largeMp4Fixture(): Buffer {
  const targetSize = 22 * 1024 * 1024;
  const freeBox = Buffer.alloc(targetSize - mp4Fixture.length);
  freeBox.writeUInt32BE(freeBox.length, 0);
  freeBox.write("free", 4);
  return Buffer.concat([mp4Fixture, freeBox]);
}

function silentWavFixture(): Buffer {
  const sampleCount = 800;
  const dataSize = sampleCount * 2;
  const buffer = Buffer.alloc(44 + dataSize);
  buffer.write("RIFF", 0);
  buffer.writeUInt32LE(36 + dataSize, 4);
  buffer.write("WAVE", 8);
  buffer.write("fmt ", 12);
  buffer.writeUInt32LE(16, 16);
  buffer.writeUInt16LE(1, 20);
  buffer.writeUInt16LE(1, 22);
  buffer.writeUInt32LE(8_000, 24);
  buffer.writeUInt32LE(16_000, 28);
  buffer.writeUInt16LE(2, 32);
  buffer.writeUInt16LE(16, 34);
  buffer.write("data", 36);
  buffer.writeUInt32LE(dataSize, 40);
  return buffer;
}

const mediaFixtures = [
  {
    label: "图片",
    extension: "png",
    mimeType: "image/png",
    mediaType: "image",
    buffer: pngFixture,
    evidencePrefix: "区域 1",
    previewSelector: "img",
  },
  {
    label: "视频",
    extension: "webm",
    mimeType: "video/webm",
    mediaType: "video",
    buffer: webmFixture,
    evidencePrefix: "片段 1",
    previewSelector: "video",
  },
  {
    label: "音频",
    extension: "wav",
    mimeType: "audio/wav",
    mediaType: "audio",
    buffer: silentWavFixture(),
    evidencePrefix: "片段 1",
    previewSelector: "audio",
  },
] as const;

test("登录接口和登录页面 HTTP 可用", async ({ request }) => {
  const response = await request.post("/api/v1/auth/login", {
    data: { email: adminEmail, password: adminPassword },
  });
  expect(response.ok()).toBe(true);
  await expect(response.json()).resolves.toMatchObject({ email: adminEmail, role: "admin" });

  const pageResponse = await request.get("/login");
  expect(pageResponse.ok()).toBe(true);
  const pageHtml = await pageResponse.text();
  expect(pageHtml).toContain("智鉴");
  expect(pageHtml).toContain("伪造人脸检测系统");
  expect(pageHtml).not.toContain("内部多模态检测系统");
});

test("可通过同源代理上传超过 10 MB 的 MP4", async ({ request }, testInfo) => {
  test.skip(testInfo.project.name !== "desktop-chromium", "大文件代理边界只需验证一次");

  const loginResponse = await request.post("/api/v1/auth/login", {
    data: { email: adminEmail, password: adminPassword },
  });
  expect(loginResponse.ok()).toBe(true);

  const buffer = largeMp4Fixture();
  expect(buffer.length).toBeGreaterThan(10 * 1024 * 1024);

  const uploadResponse = await request.post("/api/v1/uploads", {
    multipart: {
      file: {
        name: "e2e-large-proxy.mp4",
        mimeType: "video/mp4",
        buffer,
      },
    },
  });
  expect(uploadResponse.ok()).toBe(true);
  const upload = await uploadResponse.json();
  expect(upload).toMatchObject({
    mime_type: "video/mp4",
    media_type: "video",
    size_bytes: buffer.length,
  });

  const deleteResponse = await request.delete(`/api/v1/uploads/${upload.id}`);
  expect(deleteResponse.ok()).toBe(true);
});

for (const fixture of mediaFixtures) {
  test(`登录后可完成${fixture.label}上传、异步检测和删除`, async ({ page }, testInfo) => {
    const fileName = `e2e-${fixture.mediaType}-${testInfo.project.name}.${fixture.extension}`;

    await page.goto("/login");
    await expect(page.getByRole("heading", { name: "登录", exact: true })).toBeVisible();
    await page.getByRole("button", { name: "登录" }).click();
    await expect(page).toHaveURL("/");
    const menuButton = page.getByRole("button", { name: "打开菜单" });
    if (testInfo.project.name === "desktop-chromium") {
      await expect(menuButton).toBeHidden();
    } else {
      await expect(menuButton).toBeVisible();
    }

    await page.locator('input[type="file"]').setInputFiles({
      name: fileName,
      mimeType: fixture.mimeType,
      buffer: fixture.buffer,
    });
    await expect(page.getByText(fileName)).toBeVisible();

    const uploadResponsePromise = page.waitForResponse(
      (response) =>
        response.url().endsWith("/api/v1/uploads") &&
        response.request().method() === "POST",
    );
    await page.getByRole("button", { name: "开始检测" }).click();
    const uploadResponse = await uploadResponsePromise;
    expect(uploadResponse.ok()).toBe(true);
    await expect(uploadResponse.json()).resolves.toMatchObject({
      media_type: fixture.mediaType,
    });

    await expect(page).toHaveURL(/\/analysis\/[0-9a-f-]+$/);
    await expect(page.getByText("总体结论")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByRole("heading", { name: "模型结果" })).toBeVisible();
    await expect(page.locator("tbody tr").first()).toBeVisible();
    const preview = page.locator("section").filter({ has: page.getByRole("heading", { name: "媒体预览" }) }).locator(fixture.previewSelector);
    await expect(preview).toBeVisible();
    await expect(preview).toHaveAttribute("src", /.+/);
    if (fixture.mediaType === "audio") {
      await expect(page.getByText(fixture.evidencePrefix, { exact: false })).toBeVisible();
    }
    await page.screenshot({
      path: testInfo.outputPath(`analysis-${fixture.mediaType}.png`),
      fullPage: true,
    });

    page.once("dialog", (dialog) => dialog.accept());
    await page.getByRole("button", { name: "删除记录" }).click();
    await expect(page).toHaveURL("/history");
  });
}

test("可批量删除选中的分析记录", async ({ page, request }, testInfo) => {
  const projectName = testInfo.project.name;
  const runId = Date.now();
  const fileNames = [`e2e-batch-${runId}-a-${projectName}.png`, `e2e-batch-${runId}-b-${projectName}.png`];

  const loginResponse = await request.post("/api/v1/auth/login", {
    data: { email: adminEmail, password: adminPassword },
  });
  expect(loginResponse.ok()).toBe(true);

  const analysisIds: string[] = [];
  for (const fileName of fileNames) {
    const uploadResponse = await request.post("/api/v1/uploads", {
      multipart: { file: { name: fileName, mimeType: "image/png", buffer: pngFixture } },
    });
    expect(uploadResponse.ok()).toBe(true);
    const upload = await uploadResponse.json();
    const analysisResponse = await request.post("/api/v1/analyses", {
      data: { upload_id: upload.id },
    });
    expect(analysisResponse.ok()).toBe(true);
    analysisIds.push((await analysisResponse.json()).id);
  }

  for (const analysisId of analysisIds) {
    await expect
      .poll(
        async () => {
          const reportResponse = await request.get(`/api/v1/analyses/${analysisId}`);
          return reportResponse.ok() ? (await reportResponse.json()).status : "missing";
        },
        { timeout: 30_000 },
      )
      .toMatch(/^(completed|partially_completed)$/);
  }

  await page.goto("/login");
  await page.getByRole("button", { name: "登录" }).click();
  await expect(page).toHaveURL("/");
  await page.goto("/history");
  await expect(page.getByRole("heading", { name: "分析记录" })).toBeVisible();
  for (const fileName of fileNames) {
    const row = page.locator("tr", { hasText: fileName }).first();
    await expect(row).toBeVisible();
    await row.getByRole("checkbox").check();
  }
  await expect(page.getByText("已选 2 项")).toBeVisible();

  page.once("dialog", (dialog) => dialog.accept());
  await page.getByRole("button", { name: "删除所选" }).click();
  await expect(page.getByText(/已删除 2 条分析记录/)).toBeVisible();

  for (const fileName of fileNames) {
    await expect(page.locator("tr", { hasText: fileName }).first()).toHaveCount(0);
  }
  for (const analysisId of analysisIds) {
    const reportResponse = await request.get(`/api/v1/analyses/${analysisId}`);
    expect(reportResponse.status()).toBe(404);
  }
});
