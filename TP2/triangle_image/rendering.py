"""Renderizado de un `TriangleIndividual` a una imagen sobre canvas."""

from __future__ import annotations

from PIL import Image, ImageDraw

from triangle_image.gene import TriangleGene, TriangleIndividual

Color = tuple[int, int, int]


def render(
    individual: TriangleIndividual,
    width: int,
    height: int,
    background: Color = (255, 255, 255),
) -> Image.Image:
    canvas = Image.new("RGBA", (width, height), background + (255,))
    for triangle in individual.genome:
        canvas = _draw_triangle(canvas, triangle)
    return canvas.convert("RGB")


def _draw_triangle(canvas: Image.Image, triangle: TriangleGene) -> Image.Image:
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    alpha = round(triangle.alpha * 255)
    draw.polygon(triangle.vertices, fill=(triangle.r, triangle.g, triangle.b, alpha))
    return Image.alpha_composite(canvas, overlay)
