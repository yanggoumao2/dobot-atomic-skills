"""Real Chrome browser verification for the calibration wizard.

Run with the isolated Windows calibration environment:
  .venv-calibration/Scripts/python.exe -X utf8 -m tests.browser_calibration_ui \\
      --base-url http://127.0.0.1:8876 --output C:/path/to/browser-evidence

Playwright is a test-only dependency and is not listed in the runtime calibration
requirements. The local Chrome binary is used; no hardware or camera is opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from playwright.sync_api import expect, sync_playwright

from task_skills.calibration import table_calibration as table_math


DEFAULT_CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8876")
    parser.add_argument("--output", required=True)
    parser.add_argument("--chrome", default=DEFAULT_CHROME)
    parser.add_argument("--headed", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    checks: dict[str, object] = {}
    console_errors: list[str] = []
    page_errors: list[str] = []

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            headless=not args.headed,
            executable_path=args.chrome,
            args=["--disable-gpu", "--no-first-run", "--no-default-browser-check"],
        )
        context = browser.new_context(viewport={"width": 1600, "height": 1100},
                                      device_scale_factor=1, accept_downloads=True,
                                      bypass_csp=False)
        page = context.new_page()
        page.on("console", lambda message: console_errors.append(message.text)
                if message.type == "error" else None)
        page.on("pageerror", lambda error: page_errors.append(str(error)))

        def shot(name: str, full_page: bool = False) -> str:
            path = output / name
            page.screenshot(path=str(path), full_page=full_page)
            return str(path)

        page.goto(args.base_url, wait_until="networkidle")
        page.locator("#stageTitle").wait_for()
        assert page.locator("#dependencyOverall").inner_text() == "依赖符合"
        checks["step1"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "dependencies": page.locator("#dependencyOverall").inner_text(),
            "screenshot": shot("step1-preparation.png", full_page=True),
        }

        form = page.locator("#createSessionForm")
        form.locator('input[name="display_name"]').fill("浏览器 session race target")
        form.locator('select[name="mode"]').select_option("real")
        page.get_by_role("button", name="创建任务").click()
        expect(page.locator("#sessionBadge")).to_have_text(
            "浏览器 session race target", timeout=20_000)
        race_target_session_id = page.evaluate(
            "() => new URL(location.href).searchParams.get('session')")

        form = page.locator("#createSessionForm")
        form.locator('input[name="display_name"]').fill("浏览器验收 synthetic 20261020")
        form.locator('select[name="mode"]').select_option("synthetic")
        page.get_by_role("button", name="创建任务").click()
        expect(page.locator("#sessionBadge")).to_have_text(
            "浏览器验收 synthetic 20261020", timeout=20_000)
        synthetic_session_id = page.evaluate(
            "() => new URL(location.href).searchParams.get('session')")
        page.locator("#syntheticBadge").wait_for(state="visible")
        assert "synthetic_only" in page.locator("#syntheticBadge").inner_text()

        # The demo is a real background job, not a pre-canned result. Wait for
        # both arm panels to become usable and for all generated gallery items.
        expect(page.locator(".fit-panel")).to_have_count(2, timeout=150_000)
        expect(page.locator(".fit-panel").first).to_contain_text("当前可用", timeout=150_000)
        expect(page.locator(".fit-panel").nth(1)).to_contain_text("当前可用", timeout=150_000)
        page.locator('.step-button[data-step="2"]').click()
        page.locator('#stageTitle:has-text("相机内参")').wait_for()
        page.locator("#intrinsicsGallery .image-card").first.wait_for(timeout=30_000)
        image_count = page.locator("#intrinsicsGallery .image-card").count()
        assert image_count == 24, image_count
        assert "18 train / 6 val" in page.locator("#intrinsicsCountBadge").inner_text()
        assert "px" in page.locator("#intrinsicsMetrics").inner_text()
        checks["step2"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "images": image_count,
            "counts": page.locator("#intrinsicsCountBadge").inner_text(),
            "metrics": page.locator("#intrinsicsMetrics").inner_text(),
            "screenshot": shot("step2-intrinsics.png", full_page=True),
            "gallery_screenshot": shot("step2-gallery.png"),
        }

        page.locator('.step-button[data-step="3"]').click()
        page.locator('#stageTitle:has-text("固定棋盘")').wait_for()
        page.locator("#boardNumbered").wait_for(state="visible")
        assert page.locator("#boardOriginal").is_visible()
        assert page.locator("#boardStateBadge").inner_text() == "已冻结"
        confirmation = page.locator("#boardConfirmation").inner_text()
        assert "编号已冻结" in confirmation and "immutable revision" in confirmation
        radios = page.locator('input[name="cornerOrder"]')
        assert radios.count() == 2
        assert all(radios.nth(index).is_disabled() for index in range(2))
        checks["step3"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "state": page.locator("#boardStateBadge").inner_text(),
            "confirmation": confirmation,
            "numbered_visible": page.locator("#boardNumbered").is_visible(),
            "screenshot": shot("step3-fixed-board.png", full_page=True),
        }

        page.locator('.step-button[data-step="4"]').click()
        page.locator('#stageTitle:has-text("TCP")').wait_for()
        page.locator('.arm-tab[data-arm="right"]').click()
        right_panel = page.locator('.teach-panel[data-arm="right"]')
        right_panel.wait_for()
        markers = right_panel.locator(".corner-marker")
        assert markers.count() == 88
        markers.nth(7).click()
        selected_id = right_panel.locator(".teach-row input").nth(0).input_value()
        assert selected_id.isdigit(), selected_id
        assert int(selected_id) == 7
        expect(right_panel.locator(".panel-heading").first).to_contain_text("11 train")
        expect(right_panel.locator(".panel-heading").first).to_contain_text("4 val")
        assert right_panel.locator("table.data tbody tr").count() == 15
        assert right_panel.locator(".getpose-box").is_visible()
        assert right_panel.get_by_role("button", name="只读 GetPose()").is_disabled()
        entry_x = right_panel.locator(".teach-row input").nth(1)
        selected_x = entry_x.input_value()
        entry_x.fill("")
        right_panel.locator(".teach-row button").click()
        expect(page.locator("#toastRegion .toast.error").last).to_contain_text("不能为空")
        entry_x.fill(selected_x)
        first_x = right_panel.locator("table.data tbody tr").first.locator("input").first
        original_x = first_x.input_value()
        first_x.fill("")
        right_panel.get_by_role("button", name="保存表格全部编辑").click()
        expect(page.locator("#toastRegion .toast.error").last).to_contain_text("不能为空")
        first_x.fill(original_x)
        right_panel.get_by_role("button", name="保存表格全部编辑").click()
        expect(page.locator("#toastRegion .toast.success").last).to_contain_text("示教表已保存")
        checks["step4"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "arm": "right",
            "markers": markers.count(),
            "selected_corner": int(selected_id),
            "teach_rows": right_panel.locator("table.data tbody tr").count(),
            "blank_entry_rejected": True,
            "blank_xy_rejected": True,
            "screenshot": shot("step4-teach-right.png", full_page=True),
        }

        page.locator('.step-button[data-step="5"]').click()
        page.locator('#stageTitle:has-text("拟合")').wait_for()
        fit_panels = page.locator(".fit-panel")
        assert fit_panels.count() == 2
        fit_text = fit_panels.first.inner_text()
        assert "exit 0" in fit_text and "validation_passed" in fit_text
        assert "T_base_camera" in fit_text and "PNP RMS" in fit_text
        assert fit_panels.first.locator(".fit-evidence img").count() >= 2
        checks["step5"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "panels": fit_panels.count(),
            "left_metrics": fit_text,
            "left_evidence_images": fit_panels.first.locator(".fit-evidence img").count(),
            "screenshot": shot("step5-fits.png", full_page=True),
        }

        page.locator('.step-button[data-step="6"]').click()
        page.locator('#stageTitle:has-text("像素")').wait_for()
        image = page.locator("#queryImageWrap img")
        image.wait_for(state="visible")
        box = image.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        u = float(page.locator("#queryU").input_value())
        v = float(page.locator("#queryV").input_value())
        # Browser hit-testing rounds to device pixels; this still proves the
        # render-to-original scaling, not a raw rendered-coordinate shortcut.
        assert abs(u - 640) <= 4 and abs(v - 480) <= 4, (u, v)
        assert page.locator("#depthModeRadio").is_disabled()
        page.locator("#queryBoardZ").fill("-0.002")
        page.get_by_role("button", name="查询左右臂坐标").click()
        page.locator("#queryResults .arm-result").first.wait_for(timeout=10_000)
        query_text = page.locator("#queryResults").inner_text()
        assert "base(m)" in query_text and "base(mm)" in query_text
        assert "-0.0020" in query_text
        assert page.locator("#queryResults .arm-result").count() == 2
        live_query_result = page.evaluate("() => JSON.parse(JSON.stringify(state.queryResult))")

        page.locator("#queryBoardZ").fill("-0.003")
        assert page.locator("#queryResults .arm-result").count() == 0
        assert "选择像素后查询" in page.locator("#queryResults").inner_text()
        assert page.locator("#queryBoardZ").input_value() == "-0.003"
        assert page.locator("#queryBoardZ").evaluate("element => element === document.activeElement")

        # Delay a query response in the browser while changing sessions. The
        # late task-A response must not repaint task B or repopulate its cache.
        page.evaluate("""() => {
          window.__calibrationOriginalFetch = window.fetch.bind(window);
          window.__delayCalibrationQueryOnce = true;
          window.fetch = (input, init) => {
            const pending = window.__calibrationOriginalFetch(input, init);
            const url = new URL(typeof input === 'string' ? input : input.url, location.href);
            if (window.__delayCalibrationQueryOnce && url.pathname.endsWith('/query')) {
              window.__delayCalibrationQueryOnce = false;
              return pending.then((response) => new Promise((resolve) => {
                setTimeout(() => resolve(response), 1200);
              }));
            }
            return pending;
          };
        }""")
        with page.expect_request(lambda request: request.url.endswith("/query"), timeout=10_000):
            page.get_by_role("button", name="查询左右臂坐标").click()
        assert page.locator("#queryResults .arm-result").count() == 0
        page.locator("#queryBoardZ").fill("-0.001")
        assert page.evaluate("() => state.queryResult") is None
        page.locator("#sessionSelect").select_option(race_target_session_id)
        expect(page.locator("#sessionBadge")).to_have_text(
            "浏览器 session race target", timeout=20_000)
        page.wait_for_timeout(1500)
        assert page.evaluate("() => state.queryResult") is None
        assert page.evaluate("() => state.session.session_id") == race_target_session_id
        page.evaluate("() => { window.fetch = window.__calibrationOriginalFetch; }")
        page.locator("#sessionSelect").select_option(synthetic_session_id)
        expect(page.locator("#sessionBadge")).to_have_text(
            "浏览器验收 synthetic 20261020", timeout=20_000)
        assert page.evaluate("() => state.queryResult") is None
        checks["step6"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "selected_pixel": [u, v],
            "depth_disabled": page.locator("#depthModeRadio").is_disabled(),
            "query_text": query_text,
            "late_query_response_ignored_after_task_switch": True,
            "race_target_session_id": race_target_session_id,
            "screenshot": shot("step6-pixel-query.png", full_page=True),
        }

        page.locator('.step-button[data-step="7"]').click()
        page.locator('#stageTitle:has-text("独立验收")').wait_for()
        verify_panels = page.locator(".verify-panel")
        assert verify_panels.count() == 2
        for index in range(2):
            text = verify_panels.nth(index).inner_text()
            assert "独立验收通过" in text and "independent_pass" in text
            assert "reference ≠ fit held-out" in text
            assert verify_panels.nth(index).locator("table.data tbody tr").count() >= 3
        first_reference_x = verify_panels.first.locator("table.data tbody tr").first.locator("input").nth(3)
        original_reference_x = first_reference_x.input_value()
        first_reference_x.fill("")
        verify_panels.first.get_by_role("button", name="保存验收草稿").click()
        expect(page.locator("#toastRegion .toast.error").last).to_contain_text("不能为空")
        first_reference_x.fill(original_reference_x)
        right_method = verify_panels.nth(1).locator('input[type="text"]')
        unsaved_method = "右臂未保存浏览器草稿 20261020"
        right_method.fill(unsaved_method)
        page.locator('.step-button[data-step="4"]').click()
        page.locator('.arm-tab[data-arm="left"]').click()
        left_panel = page.locator('.teach-panel[data-arm="left"]')
        left_panel.get_by_role("button", name="保存表格全部编辑").click()
        expect(page.locator("#toastRegion .toast.success").last).to_contain_text("左臂示教表已保存")
        page.locator('.step-button[data-step="7"]').click()
        verify_panels = page.locator(".verify-panel")
        preserved_method = verify_panels.nth(1).locator('input[type="text"]').input_value()
        assert preserved_method == unsaved_method, preserved_method
        assert page.evaluate("() => Boolean(state.dirtyVerify.right)") is True
        checks["step7"] = {
            "title": page.locator("#stageTitle").inner_text(),
            "panels": verify_panels.count(),
            "left_result": verify_panels.first.inner_text(),
            "before_export_screenshot": shot("step7-verification.png", full_page=True),
            "right_draft_preserved_after_left_save": preserved_method,
            "blank_independent_reference_rejected": True,
        }

        page.get_by_role("button", name="生成报告包").click()
        download_link = page.locator('#exportResult a[download]')
        download_link.wait_for(timeout=20_000)
        download_target = output / "calibration-browser-export.zip"
        with page.expect_download(timeout=30_000) as download_info:
            download_link.click()
        download = download_info.value
        download.save_as(str(download_target))
        assert download_target.is_file() and download_target.stat().st_size > 1000
        with zipfile.ZipFile(download_target) as archive:
            names = set(archive.namelist())
            assert {"camera/intrinsics.json", "board/observation.json",
                    "board/confirmation.json"}.issubset(names), sorted(names)
            intrinsics = json.loads(archive.read("camera/intrinsics.json"))
            assert len(intrinsics["K"]) == 3 and len(intrinsics["distortion"]) == 5
            manifest = json.loads(archive.read("manifest.json"))
            for name, digest in manifest["files"].items():
                assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        with tempfile.TemporaryDirectory(prefix="calibration-browser-export-") as extracted_dir:
            with zipfile.ZipFile(download_target) as archive:
                archive.extractall(extracted_dir)
            extracted = Path(extracted_dir)
            archived_intrinsics = json.loads(
                (extracted / "camera/intrinsics.json").read_text(encoding="utf-8"))
            archived_extrinsics = json.loads(
                (extracted / "arms/left/extrinsics.json").read_text(encoding="utf-8"))
            archived_observation = json.loads(
                (extracted / "board/observation.json").read_text(encoding="utf-8"))
            assert len(archived_observation["corners_uv"]) == 88
            archived_point, _ = table_math.pixel_to_plane(
                (live_query_result["u_px"], live_query_result["v_px"]),
                archived_intrinsics,
                np.asarray(archived_intrinsics["K"], dtype=float),
                np.asarray(archived_intrinsics["distortion"], dtype=float),
                np.asarray(archived_extrinsics["T_base_camera"], dtype=float),
                np.asarray(archived_extrinsics["T_base_board"], dtype=float),
                live_query_result["board_z_m"],
            )
            live_point = np.asarray(
                live_query_result["arms"]["left"]["point_base_m"], dtype=float)
            np.testing.assert_allclose(archived_point, live_point, atol=1e-12)
            archive_mapping_error_m = float(np.max(np.abs(archived_point - live_point)))
        checks["export"] = {
            "filename": download.suggested_filename,
            "size_bytes": download_target.stat().st_size,
            "sha256_text_present": "sha256:" in page.locator("#exportResult").inner_text(),
            "intrinsics_k_distortion_present": True,
            "manifest_hash_verified": True,
            "fresh_extraction_mapping_error_m": archive_mapping_error_m,
            "screenshot": shot("step7-export.png", full_page=True),
        }

        # Full reload proves the browser restores the session and validated
        # evidence from the server, not only in-memory JavaScript state.
        page.reload(wait_until="networkidle")
        page.locator("#sessionBadge").wait_for()
        expect(page.locator(".verify-panel")).to_have_count(2, timeout=30_000)
        expect(page.locator(".verify-panel").first).to_contain_text("独立验收通过", timeout=30_000)
        checks["reload"] = {
            "session_badge": page.locator("#sessionBadge").inner_text(),
            "synthetic_badge": page.locator("#syntheticBadge").inner_text(),
            "verification_restored": page.locator(".verify-panel").first.inner_text()[:800],
            "screenshot": shot("step7-after-reload.png", full_page=True),
        }

        # Deferred session GETs exercise same-arm drafts edited after the PUT
        # but before the refresh response arrives.
        page.locator('.step-button[data-step="4"]').click()
        page.locator('.arm-tab[data-arm="right"]').click()
        right_panel = page.locator('.teach-panel[data-arm="right"]')
        right_x = right_panel.locator("table.data tbody tr").first.locator("input").first
        submitted_x = right_x.input_value()
        right_x.fill(str(float(submitted_x) + 0.01))
        right_x.fill(submitted_x)
        page.evaluate("""() => {
          window.__calibrationOriginalFetch = window.fetch.bind(window);
          window.__delaySessionGetOnce = true;
          const current = state.sessionId;
          window.fetch = (input, init) => {
            const pending = window.__calibrationOriginalFetch(input, init);
            const url = new URL(typeof input === 'string' ? input : input.url, location.href);
            if (window.__delaySessionGetOnce && url.pathname === `/api/sessions/${current}`
                && (!init || !init.method || init.method === 'GET')) {
              window.__delaySessionGetOnce = false;
              return pending.then((response) => new Promise((resolve) => {
                setTimeout(() => {
                  const input = document.querySelector('.teach-panel[data-arm="right"] table.data tbody tr:first-child input');
                  input.value = '456.789';
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  resolve(response);
                }, 900);
              }));
            }
            return pending;
          };
        }""")
        right_panel.get_by_role("button", name="保存表格全部编辑").click()
        expect(page.locator("#toastRegion .toast.success").last).to_contain_text("示教表已保存")
        retained_teach_value = right_panel.locator("table.data tbody tr").first.locator("input").first.input_value()
        assert retained_teach_value == "456.789", retained_teach_value
        assert page.evaluate("() => state.dirtyTeach.right") is True

        page.locator('.step-button[data-step="7"]').click()
        left_verify = page.locator('.verify-panel[data-arm="left"]')
        method = left_verify.locator('input[type="text"]')
        submitted_method = method.input_value()
        method.fill(submitted_method + " temp")
        method.fill(submitted_method)
        page.evaluate("""() => {
          window.__delaySessionGetOnce = true;
          const current = state.sessionId;
          window.fetch = (input, init) => {
            const pending = window.__calibrationOriginalFetch(input, init);
            const url = new URL(typeof input === 'string' ? input : input.url, location.href);
            if (window.__delaySessionGetOnce && url.pathname === `/api/sessions/${current}`
                && (!init || !init.method || init.method === 'GET')) {
              window.__delaySessionGetOnce = false;
              return pending.then((response) => new Promise((resolve) => {
                setTimeout(() => {
                  const input = document.querySelector('.verify-panel[data-arm="left"] input[type="text"]');
                  input.value = '保存期间录入的后续方法';
                  input.dispatchEvent(new Event('input', { bubbles: true }));
                  resolve(response);
                }, 900);
              }));
            }
            return pending;
          };
        }""")
        left_verify.get_by_role("button", name="保存验收草稿").click()
        expect(page.locator("#toastRegion .toast.success").last).to_contain_text("验收草稿已保存")
        retained_method = left_verify.locator('input[type="text"]').input_value()
        assert retained_method == "保存期间录入的后续方法", retained_method
        assert page.evaluate("() => state.dirtyVerify.left") is True
        page.evaluate("() => { window.fetch = window.__calibrationOriginalFetch; }")
        checks["deferred_same_arm_drafts"] = {
            "teach_value_preserved": retained_teach_value,
            "verification_method_preserved": retained_method,
            "dirty_flags_retained": True,
        }

        # An accepted upstream edit invalidates visible XYZ and hides the old
        # export record because its saved dependency snapshot no longer matches.
        page.locator('.step-button[data-step="6"]').click()
        image = page.locator("#queryImageWrap img")
        box = image.bounding_box()
        assert box is not None
        page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
        page.locator("#queryBoardZ").fill("0")
        page.get_by_role("button", name="查询左右臂坐标").click()
        page.locator("#queryResults .arm-result").first.wait_for(timeout=10_000)
        page.locator('.step-button[data-step="4"]').click()
        page.locator('.arm-tab[data-arm="left"]').click()
        left_panel = page.locator('.teach-panel[data-arm="left"]')
        left_x = left_panel.locator("table.data tbody tr").first.locator("input").first
        left_x.fill(str(float(left_x.input_value()) + 1.0))
        left_panel.get_by_role("button", name="保存表格全部编辑").click()
        expect(page.locator("#toastRegion .toast.success").last).to_contain_text("示教表已保存")
        assert page.evaluate("() => state.queryResult") is None
        page.locator('.step-button[data-step="6"]').click()
        assert "选择像素后查询" in page.locator("#queryResults").inner_text()
        page.locator('.step-button[data-step="7"]').click()
        assert page.locator("#exportResult a[download]").count() == 0
        checks["upstream_invalidation"] = {
            "visible_query_cleared": page.evaluate("() => state.queryResult") is None,
            "old_export_hidden": page.locator("#exportResult a[download]").count() == 0,
        }

        checks["console_errors"] = console_errors
        checks["page_errors"] = page_errors
        checks["passed"] = not console_errors and not page_errors
        checks["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        (output / "browser_run.json").write_text(
            json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        browser.close()

    print(json.dumps({key: value for key, value in checks.items()
                      if key not in ("step5", "step7", "reload")},
                     ensure_ascii=False, indent=2))
    if page_errors or console_errors:
        print("BROWSER_ERRORS", json.dumps({"page": page_errors, "console": console_errors},
                                           ensure_ascii=False))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
