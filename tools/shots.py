"""Playwright screenshots of the 9 steps (real clicks on the page). usage: shots.py URL OUTDIR"""
import sys, time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL, OUT = sys.argv[1], Path(sys.argv[2]); OUT.mkdir(exist_ok=True)
EX = Path(__file__).resolve().parents[1] / "example_data"
with sync_playwright() as p:
    b = p.chromium.launch(); pg = b.new_page(viewport={"width": 1440, "height": 900})
    pg.on("pageerror", lambda e: print("PAGE ERROR", e))
    def idle(t=600):
        pg.wait_for_function("!document.body.style.cursor", timeout=t * 1000); time.sleep(1.2)
    def title(t): pg.wait_for_function(f"document.getElementById('stepTitle').textContent.startsWith({t!r})", timeout=600000); idle()
    def shot(n): pg.screenshot(path=str(OUT / n)); print("shot", n)
    def main(): pg.click("#mainBtns button.main")
    pg.goto(URL); title("Files")
    pg.fill("#f_bse", str(EX / "10B_BSE_Image.tiff")); pg.dispatch_event("#f_bse", "change"); time.sleep(3)
    pg.fill("#f_out", "/tmp/shotout"); shot("1_files.png"); main(); title("Legend")
    pg.wait_for_selector("#lgT tr[data-i]"); time.sleep(1); shot("2_legend.png")
    pg.click("#lgAdd"); time.sleep(0.6); pg.fill("#eName", "Quartz"); time.sleep(0.6); shot("2b_legend_add_entry.png"); pg.click("#eCancel")
    main(); title("BSE"); shot("3_footer.png"); main(); title("Non-grain"); time.sleep(2); shot("4_nongrain.png"); main()
    title("Scale"); cv = pg.locator("#cv").bounding_box()
    pg.mouse.click(cv["x"] + cv["width"] * 0.3, cv["y"] + cv["height"] * 0.94); pg.mouse.click(cv["x"] + cv["width"] * 0.45, cv["y"] + cv["height"] * 0.94)
    pg.fill("#scLen", "100"); pg.dispatch_event("#scLen", "change"); time.sleep(1); shot("5_scale.png")
    pg.click("#mainBtns button:has-text('Skip')"); title("Grain seg")
    pg.click("#sgLoad"); idle(); time.sleep(1)
    pg.mouse.click(cv["x"] + cv["width"] * 0.5, cv["y"] + cv["height"] * 0.5); time.sleep(1); shot("6_segmentation.png")
    pg.click("#sgUse"); title("Minerals"); time.sleep(2); shot("7_minerals.png")
    main(); title("Grain size"); time.sleep(2); shot("8_grain_size.png")
    main(); title("Outputs"); time.sleep(2); main(); idle(); time.sleep(2); shot("9_outputs.png")
    b.close()
