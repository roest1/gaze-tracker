"""CLI entry point. Subcommands: calibrate, track, eval."""
from __future__ import annotations

import argparse

from .calibration import run_calibration
from .evaluate import run_evaluation
from .mapping import ALL_BASES, BASIS_CARTESIAN
from .snap import grid_targets
from .stream import run_tracking

DEFAULT_DPI = 96.0
DEFAULT_FACE_DISTANCE_CM = 50.0


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="gaze-tracker",
        description=(
            "Webcam gaze-to-screen tracker "
            "(MediaPipe Face Mesh + linear calibration + One Euro smoothing)."
        ),
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_cal = sub.add_parser(
        "calibrate", help="Run 9-point calibration and save the model.",
    )
    p_cal.add_argument(
        "--basis", choices=ALL_BASES, default=BASIS_CARTESIAN,
        help=(
            "Regression basis for the gaze->screen map. cartesian: [1, vx, vy, vz] "
            "(4 coefs/axis, default). polar: [1, yaw, pitch] (3, eye motor space). "
            "polynomial-polar: [1, yaw, pitch, yaw^2, pitch^2, yaw*pitch] (6, "
            "handles tangent stretch at screen edges). Use `gaze-tracker eval "
            "--bench` to A/B all three on the same captured fixations."
        ),
    )

    p_track = sub.add_parser("track", help="Run realtime gaze tracking with a saved calibration.")
    p_track.add_argument(
        "--min-cutoff", type=float, default=0.5,
        help="One Euro min cutoff (lower = smoother at rest)",
    )
    p_track.add_argument(
        "--beta", type=float, default=0.05, help="One Euro beta (higher = more responsive at speed)"
    )
    p_track.add_argument(
        "--feature-window",
        type=int,
        default=5,
        help="Median window (frames) over the raw 3D gaze feature; 1 disables",
    )
    p_track.add_argument(
        "--targets-demo",
        action="store_true",
        help=(
            "Render a 2x3 grid (top/bottom × left/center/right) of demo target "
            "boxes and snap gaze to the nearest one."
        ),
    )
    p_track.add_argument(
        "--snap-attractor-px", type=float, default=140.0,
        help="Outer snap radius in screen px (beyond: passthrough)",
    )
    p_track.add_argument(
        "--snap-lock-px", type=float, default=55.0,
        help="Inner snap radius in screen px (within: full snap to target center)",
    )
    p_track.add_argument(
        "--snap-unlock-px", type=float, default=95.0,
        help="Hysteresis radius in screen px (must be >= --snap-lock-px)",
    )
    p_track.add_argument(
        "--saccade-px-per-s", type=float, default=2500.0,
        help="Saccade velocity threshold in px/s; above this, target-snap is bypassed",
    )
    p_track.add_argument(
        "--no-saccade", action="store_true",
        help="Disable saccade-based snap suppression",
    )
    p_track.add_argument(
        "--click-weight-scale", type=float, default=20.0,
        help=(
            "Px of prediction error per unit click weight. Lower = clicks bend "
            "the fit harder for a given error. Default 20."
        ),
    )
    p_track.add_argument(
        "--click-weight-min", type=float, default=5.0,
        help=(
            "Floor for click weight (calibration samples are weight 1). "
            "A perfectly-on click still gets at least this much authority."
        ),
    )
    p_track.add_argument(
        "--click-weight-max", type=float, default=80.0,
        help=(
            "Ceiling for click weight. Caps how much one freak click can rotate "
            "the fit; protects against oscillation from outlier corrections."
        ),
    )

    p_eval = sub.add_parser(
        "eval",
        help="Run held-out evaluation against the saved calibration.",
    )
    p_eval.add_argument(
        "--monitor-dpi", type=float, default=DEFAULT_DPI,
        help=(
            "Pixels per inch of the display. Required for converting pixel error "
            "to degrees of visual angle. Default 96 with a startup warning — "
            "no portable Wayland DPI query exists."
        ),
    )
    p_eval.add_argument(
        "--face-distance-cm", type=float, default=DEFAULT_FACE_DISTANCE_CM,
        help=(
            "Estimated face-to-screen distance in cm. Used for px->degree "
            "conversion. Default 50."
        ),
    )
    p_eval.add_argument(
        "--seed", type=int, default=None,
        help=(
            "RNG seed for eval point layout. Default: derived from calibration "
            "file hash so the same calibration always sees the same points."
        ),
    )
    p_eval.add_argument(
        "--bench", action="store_true",
        help=(
            "A/B all bases (cartesian, polar, polynomial-polar) on the same "
            "captured fixations. Each basis is refit on the saved calibration's "
            "anchor samples; only the model architecture varies. Result is a "
            "side-by-side stdout table; the saved model + eval log are unchanged."
        ),
    )

    args = parser.parse_args()

    if args.cmd == "calibrate":
        model = run_calibration(camera_index=args.camera, basis=args.basis)
        if model is None:
            print("calibration aborted or insufficient points")
            return
        print(
            f"calibration saved for screen {model.screen_w}x{model.screen_h} "
            f"(basis={model.basis})"
        )
    elif args.cmd == "track":
        targets = None
        if args.targets_demo:
            from .mapping import GazeModel, calibration_path
            cal = GazeModel.load(calibration_path())
            # 2 rows × 3 cols = 6 demo zones (top/bottom × left/center/right).
            targets = grid_targets(
                cal.screen_w, cal.screen_h,
                cols=3, rows=2, box_w=320, box_h=220,
            )
        run_tracking(
            camera_index=args.camera,
            min_cutoff=args.min_cutoff,
            beta=args.beta,
            feature_window=args.feature_window,
            targets=targets,
            snap_attractor_px=args.snap_attractor_px,
            snap_lock_px=args.snap_lock_px,
            snap_unlock_px=args.snap_unlock_px,
            saccade_px_per_s=None if args.no_saccade else args.saccade_px_per_s,
            click_weight_scale=args.click_weight_scale,
            click_weight_min=args.click_weight_min,
            click_weight_max=args.click_weight_max,
        )
    elif args.cmd == "eval":
        if args.monitor_dpi == DEFAULT_DPI:
            # Without a real DPI the deg-of-visual-angle column is meaningless.
            # Still let it run — px metrics are valid — but call out the lie.
            print(
                f"[gaze-tracker] warning: --monitor-dpi not set, defaulting to "
                f"{DEFAULT_DPI:.0f}. Degree-of-visual-angle metrics will be "
                "wrong. Pass --monitor-dpi for accurate degrees."
            )
        report = run_evaluation(
            camera_index=args.camera,
            monitor_dpi=args.monitor_dpi,
            face_distance_cm=args.face_distance_cm,
            seed=args.seed,
            bench=args.bench,
        )
        if report is None:
            return


if __name__ == "__main__":
    main()
