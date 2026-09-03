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
    canvas = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(canvas, "RGBA")
    for triangle in individual.genome:
        alpha = round(triangle.alpha * 255)
        draw.polygon(triangle.vertices, fill=(triangle.r, triangle.g, triangle.b, alpha))
    return canvas
