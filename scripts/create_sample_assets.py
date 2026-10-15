"""
Generate a synthetic multi-depth sample image for testing the bokeh pipeline.
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def create_sample_scene(output_path: str = "assets/sample_desk.jpg"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    w, h = 640, 480
    img = Image.new("RGB", (w, h), color=(220, 225, 230))
    draw = ImageDraw.Draw(img)

    # 1. Background wall / room with gradient (Far plane)
    for y in range(h // 2):
        shade = int(180 + (y / (h // 2)) * 40)
        draw.line([(0, y), (w, y)], fill=(shade - 30, shade - 20, shade))

    # Window in background
    draw.rectangle([w - 220, 40, w - 40, 200], fill=(135, 206, 235), outline=(100, 100, 100), width=4)
    draw.line([(w - 130, 40), (w - 130, 200)], fill=(100, 100, 100), width=3)
    draw.line([(w - 220, 120), (w - 40, 120)], fill=(100, 100, 100), width=3)

    # Background bookshelf
    draw.rectangle([40, 60, 200, 220], fill=(139, 69, 19), outline=(90, 40, 10), width=3)
    for shelf_y in [110, 160, 210]:
        draw.line([(40, shelf_y), (200, shelf_y)], fill=(90, 40, 10), width=4)
        for bx in range(50, 190, 25):
            book_colors = [(180, 50, 50), (40, 120, 180), (50, 150, 60), (200, 160, 40)]
            draw.rectangle([bx, shelf_y - 40, bx + 18, shelf_y], fill=book_colors[(bx // 25) % 4])

    # 2. Table surface (Midground to Foreground)
    table_top_y = h // 2
    for y in range(table_top_y, h):
        factor = (y - table_top_y) / (h - table_top_y)
        r = int(160 - factor * 40)
        g = int(110 - factor * 30)
        b = int(70 - factor * 20)
        draw.line([(0, y), (w, y)], fill=(r, g, b))

    # 3. Midground Object: A coffee mug & notebook
    # Notebook
    draw.polygon([(240, 270), (400, 270), (430, 360), (270, 360)], fill=(245, 245, 240), outline=(200, 200, 200))
    draw.line([(335, 270), (350, 360)], fill=(150, 150, 150), width=2)
    # Coffee mug
    draw.ellipse([160, 260, 230, 290], fill=(220, 60, 50))
    draw.rectangle([160, 275, 230, 340], fill=(220, 60, 50))
    draw.ellipse([160, 325, 230, 355], fill=(180, 40, 30))
    draw.ellipse([170, 265, 220, 285], fill=(70, 40, 20))  # Coffee surface

    # 4. Foreground Object: Camera / Plant leaves at very near depth
    # Near flowerpot on right
    draw.polygon([(460, 320), (580, 320), (560, 460), (480, 460)], fill=(180, 90, 50))
    # Green leaves
    for cx, cy, rx, ry, col in [
        (480, 300, 40, 25, (34, 139, 34)),
        (540, 290, 45, 30, (46, 139, 87)),
        (510, 260, 35, 45, (60, 179, 113)),
        (560, 330, 35, 20, (34, 139, 34)),
    ]:
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=col)

    img.save(output_path, quality=95)
    print(f"Sample scene generated at: {output_path}")


if __name__ == "__main__":
    create_sample_scene()
