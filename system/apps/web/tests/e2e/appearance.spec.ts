import { expect, test } from "@playwright/test";

test("外观跟随系统，并记住手动选择", async ({ page, context }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto("/login");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await expect(page.getByRole("button", { name: "跟随系统" })).toHaveAttribute("aria-pressed", "true");
  await page.emulateMedia({ colorScheme: "dark" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  const dayButton = page.getByRole("button", { name: "白天模式" });
  await dayButton.focus();
  await page.keyboard.press("Enter");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.reload();
  await expect(dayButton).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  const secondPage = await context.newPage();
  await secondPage.goto("/login");
  await expect(secondPage.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("button", { name: "夜间模式" }).click();
  await expect(secondPage.locator("html")).toHaveAttribute("data-theme", "dark");
  await page.getByRole("button", { name: "跟随系统" }).click();
  await page.emulateMedia({ colorScheme: "light" });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  expect(errors).toEqual([]);
});

test("页面在两种外观下可操作且无横向溢出", async ({ page }, testInfo) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/login");
  await page.getByRole("button", { name: "夜间模式" }).click();
  await page.getByRole("button", { name: "登录", exact: true }).click();
  await expect(page).toHaveURL("/");
  for (const [mode, label] of [["light", "白天模式"], ["dark", "夜间模式"]]) {
    await page.getByRole("button", { name: label }).click();
    await expect(page.locator("html")).toHaveAttribute("data-theme", mode);
    await expect(page.getByRole("button", { name: "选择媒体文件" })).toBeVisible();
    await page.locator('input[type="file"]').setInputFiles({ name: "theme-check.mkv", mimeType: "application/octet-stream", buffer: Buffer.from("selection only") });
    await expect(page.getByRole("heading", { name: "theme-check.mkv" })).toBeVisible();
    await page.getByTitle("移除文件").click();
    await page.screenshot({ path: testInfo.outputPath(`home-${mode}.png`), fullPage: true });
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    if (testInfo.project.name === "mobile-chromium") await page.getByRole("button", { name: "打开菜单" }).click();
    await page.getByRole("navigation", { name: "主导航" }).getByRole("link", { name: "分析记录" }).click();
    await expect(page.getByRole("heading", { name: "分析记录" })).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("data-theme", mode);
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
    await page.goto("/");
    await expect(page.getByRole("button", { name: "选择媒体文件" })).toBeVisible();
  }
});
