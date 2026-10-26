import shutil
from pathlib import Path
from PIL import Image
from common import EX
d = Path.home() / "pp_webapp_test_out" / "big"; d.mkdir(parents=True, exist_ok=True)
for a, b in (("10B_BSE_Image.tiff", "big_BSE_Image.tiff"), ("10B_Mineral_Image.tiff", "big_Mineral_Image.tiff")):
    im = Image.open(EX / a); im.resize((im.width * 2, im.height * 2), Image.NEAREST).save(d / b)
shutil.copy(EX / "10-Mineral_Legend.tiff", d)
print(Image.open(d / "big_BSE_Image.tiff").size)
