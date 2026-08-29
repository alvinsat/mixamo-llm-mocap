"""RTMPose variant of the Mixamo XPU workflow GUI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from xpu_gui import MocapGui, PYTHON, REPO


RTMPOSE_RUNNER = REPO / "single-mode" / "rtmpose.py"
MOTIONBERT_RUNNER = REPO / "single-mode" / "MotionBERT" / "rtmpose.py"
SINGLE_MODE = REPO / "single-mode"


class RtmposeMocapGui(MocapGui):
    """Use RTMPose for extraction, then reuse the standard retarget stages."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Mocap Foundry / RTMPose")
        self.vars["landmarks"].set(str(SINGLE_MODE / "mocap_data.json"))
        self._replace_text(
            "Extract landmarks",
            "Extract RTMPose",
        )
        self._replace_text(
            "Run GVHMR, YOLO, ViTPose and HMR2 on XPU",
            "Run RTMPose landmark extraction on XPU",
        )

    def _replace_text(self, old: str, new: str) -> None:
        def visit(widget):
            try:
                if widget.cget("text") == old:
                    widget.configure(text=new)
            except tk_error:
                pass
            for child in widget.winfo_children():
                visit(child)

        visit(self)

    def run_extract(self):
        if self._check_paths("video"):
            self._run_rtmpose()

    def run_motionbert_preview(self):
        """Lift the current RTMPose result and show the combined preview."""
        input_path = SINGLE_MODE / "mocap_data.json"
        output_path = SINGLE_MODE / "motionbert_xpu.npy"
        if not input_path.exists():
            self._write("Run Step 2 first: RTMPose mocap_data.json is missing.\n")
            return
        try:
            metadata = json.loads(input_path.read_text(encoding="utf-8")).get("metadata", {})
            width, height = metadata.get("resolution", [864, 1080])
        except (OSError, ValueError, TypeError):
            width, height = 864, 1080
        self._run(
            "RTMPose + MotionBERT preview",
            [
                str(PYTHON), str(MOTIONBERT_RUNNER),
                "--input", str(input_path), "--output", str(output_path),
                "--width", str(width), "--height", str(height),
                "--preview", "--video", self.vars["video"].get(),
            ],
        )

    def _run_rtmpose(self, on_done=None):
        self._run(
            "RTMPose landmark extraction",
            [str(PYTHON), str(RTMPOSE_RUNNER), "--video", self.vars["video"].get()],
            cwd=SINGLE_MODE,
            on_done=lambda: self._rtmpose_ready(on_done),
        )

    def _rtmpose_ready(self, on_done=None):
        source = Path(self.vars["landmarks"].get())
        try:
            spec = json.loads(Path(self.vars["spec"].get()).read_text(encoding="utf-8"))
            target = Path(spec["landmarks"])
            target = target if target.is_absolute() else REPO / target
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            self._write(f"RTMPose output copied to the action spec landmark path: {target}\n")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            self._write(f"Could not sync RTMPose landmarks to the action spec: {exc}\n")
            return
        if on_done:
            on_done()

    def _full_extract(self):
        if not self.running:
            self._run_rtmpose(on_done=self._full_analyze)


try:
    import tkinter as tk
    tk_error = tk.TclError
except ImportError:
    tk_error = Exception


if __name__ == "__main__":
    app = RtmposeMocapGui()
    app.mainloop()
