"""Shared fullscreen-canvas display helpers used by calibration and eval.

Kept private (`_display`) because nothing outside the package should depend on
the OpenCV rendering details. The functions are public-named (`draw_dot`,
not `_draw_dot`) so cross-module imports inside the package don't reach into
private names.
"""
from __future__ import annotations

import cv2
import numpy as np


def screen_size() -> tuple[int, int]:
    """Primary monitor logical size, via tkinter (works on X11 + Wayland).

    Caveat: on multi-monitor Wayland setups this returns the compositor's
    primary-display extent, which may not match the monitor cv2 fullscreens
    onto. If you hit that, override at the call site.
    """
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()
    w, h = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
    return w, h


def draw_dot(
    canvas: np.ndarray, pos: tuple[float, float], capturing: bool = False
) -> None:
    canvas[:] = 0
    x, y = int(pos[0]), int(pos[1])
    inner = (0, 200, 0) if capturing else (0, 0, 255)
    cv2.circle(canvas, (x, y), 25, (40, 40, 40), -1)
    cv2.circle(canvas, (x, y), 15, inner, -1)
    cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)


def draw_text(
    canvas: np.ndarray,
    text: str,
    y: int,
    color: tuple[int, int, int] = (200, 200, 200),
    scale: float = 0.9,
    centered: bool = False,
) -> None:
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x = (canvas.shape[1] - tw) // 2 if centered else 50
    cv2.putText(
        canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA
    )


def panel_y(dot_y_norm: float, screen_h: int) -> int:
    """Place an instruction panel vertically opposite the dot to avoid overlap."""
    return int(screen_h * 0.72) if dot_y_norm < 0.5 else int(screen_h * 0.28)
