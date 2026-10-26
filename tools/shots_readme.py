"""README-style shots (like docs/images/app_*.jpg): 1200 px wide progressive JPG. usage: shots_readme.py URL OUTDIR OUTFOLDER"""
import sys, time, io
from pathlib import Path
from PIL import Image
from playwright.sync_api import sync_playwright
URL, OUT, DATA = sys.argv[1], Path(sys.argv[2]), sys.argv[3]; OUT.mkdir(exist_ok=True)
EX = Path(__file__).resolve().parents[1] / "example_data"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1440, "height": 855}, device_scale_factor=2)
    pg.on("pageerror", lambda e: print("PAGE ERROR", e))
    def idle(t=600): pg.wait_for_function("!document.body.style.cursor", timeout=t * 1000); time.sleep(1.5)
    def title(t): pg.wait_for_function(f"document.getElementById('stepTitle').textContent.startsWith({t!r})", timeout=600000); idle()
    def shot(n):
        im = Image.open(io.BytesIO(pg.screenshot())).convert("RGB")
        im = im.resize((1200, round(im.height * 1200 / im.width)), Image.LANCZOS)
        im.save(OUT / n, "JPEG", quality=88, optimize=True, progressive=True); print("shot", n, im.size)
    def main(): pg.click("#mainBtns button.main")
    pg.goto(URL); title("Files")
    pg.fill("#f_bse", str(EX / "10B_BSE_Image.tiff")); pg.dispatch_event("#f_bse", "change"); time.sleep(3)
    pg.fill("#f_out", DATA); pg.dispatch_event("#f_out", "change"); time.sleep(0.5); shot("app_web_files.jpg"); main(); title("Legend")
    pg.wait_for_selector("#lgT tr[data-i]"); time.sleep(1)
    nm = pg.locator("#lgT tr[data-i='6'] td.nm input"); v = nm.input_value(); nm.fill(v + " (edited)"); nm.dispatch_event("change"); time.sleep(1)
    pg.click("#lgAdd"); time.sleep(0.6); pg.fill("#eHex", "#008000"); pg.dispatch_event("#eHex", "change"); pg.fill("#eName", "Ilmenite B"); time.sleep(1.2); print("eErr:", pg.text_content("#eErr")); pg.click("#eOk", timeout=5000); time.sleep(1.5)
    pg.mouse.move(5, 5); shot("app_web_legend.jpg")
    for _ in range(2): pg.locator("button:has-text('Undo')").first.click(); time.sleep(0.8)
    main(); title("BSE"); main(); title("Non-grain"); time.sleep(2); shot("app_web_nongrain.jpg"); main()
    title("Scale"); pg.click("#mainBtns button:has-text('Skip')"); title("Grain seg")
    if "--run" in sys.argv:
        pg.click("button:has-text('Run segmentation')"); time.sleep(40); pg.mouse.move(5, 5); shot("app_web_segmentation_running.jpg"); idle(900)
    else:
        pg.click("#sgLoad"); idle()
    time.sleep(1.5)
    cv = pg.locator("#cv").bounding_box()
    for fx, fy in ((0.5, 0.5), (0.53, 0.47)):
        pg.keyboard.down("Shift"); pg.mouse.click(cv["x"] + cv["width"] * fx, cv["y"] + cv["height"] * fy); pg.keyboard.up("Shift"); time.sleep(0.6)
    pg.mouse.move(5, 5); time.sleep(1); shot("app_web_segmentation.jpg")
    # zoom / pan in the real browser: wheel zoom in x3, drag, then read image request timings
    cvb = pg.locator("#cv").bounding_box(); cx, cy = cvb["x"] + cvb["width"] / 2, cvb["y"] + cvb["height"] / 2
    t0 = time.perf_counter(); pg.mouse.move(cx, cy)
    for _ in range(6): pg.mouse.wheel(0, -300); time.sleep(0.15)
    pg.mouse.down(); pg.mouse.move(cx + 200, cy + 120, steps=10); pg.mouse.up(); time.sleep(2.5)
    print("zoom/pan wall", round(time.perf_counter() - t0, 2))
    print("img requests", pg.evaluate("performance.getEntriesByType('resource').filter(e=>e.name.includes('/api/img/')).map(e=>[e.name.split('/api/img/')[1].slice(0,40), Math.round(e.duration)])"))
    pg.keyboard.press("f"); time.sleep(1)
    pg.click("#sgDes"); time.sleep(0.5); pg.click("#sgUse"); title("Minerals"); time.sleep(2); shot("app_web_minerals.jpg")
    main(); title("Grain size"); time.sleep(2); shot("app_web_grainsize.jpg")
    main(); title("Outputs"); time.sleep(2); main(); idle(); time.sleep(2); shot("app_web_outputs.jpg")
    b.close()
