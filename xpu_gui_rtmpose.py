"""RTMPose + MotionBERT variant of the Mixamo XPU workflow GUI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from xpu_gui import MOTIONBERT_RUNNER, MocapGui, PYTHON, REPO


RTMPOSE_RUNNER = REPO / "single-mode" / "rtmpose.py"
SINGLE_MODE = REPO / "single-mode"


class RtmposeMocapGui(MocapGui):
    """Use RTMPose and MotionBERT for extraction, then reuse stages 3-6."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Mocap Foundry / RTMPose + MotionBERT")
        self.vars["landmarks"].set(str(SINGLE_MODE / "mocap_data.json"))
        self.vars["motionbert_input"].set(str(SINGLE_MODE / "mocap_data.json"))
        self.vars["motionbert_output"].set(str(SINGLE_MODE / "motionbert_xpu.npy"))
        self._replace_text(
            "Extract landmarks",
            "Extract RTMPose + MotionBERT",
        )
        self._replace_text(
            "Run GVHMR, YOLO, ViTPose and HMR2 on XPU",
            "Run RTMPose, then lift its tracks with MotionBERT",
        )
        self._replace_text(
            "MotionBERT 3D preview",
            "Run MotionBERT only",
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
            self._run_rtmpose(on_done=self._run_motionbert_after_rtmpose)

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

    def _run_motionbert_after_rtmpose(self):
        self._run(
            "MotionBERT 3D lifting",
            self._motionbert_args(),
            on_done=self._log_extraction_ready,
        )

    def _motionbert_args(self):
        try:
            metadata = json.loads(Path(self.vars["motionbert_input"].get()).read_text(encoding="utf-8")).get("metadata", {})
            width, height = metadata.get("resolution", [864, 1080])
        except (OSError, ValueError, TypeError):
            width, height = 864, 1080
        return [
            str(PYTHON),
            str(MOTIONBERT_RUNNER),
            "--input", self.vars["motionbert_input"].get(),
            "--output", self.vars["motionbert_output"].get(),
            "--width", str(width),
            "--height", str(height),
            "--preview",
            "--video", self.vars["video"].get(),
        ]

    def _log_extraction_ready(self):
        self._write(
            "RTMPose landmarks are ready for steps 3-6. "
            "MotionBERT 3D output was saved separately for preview/use.\n"
        )

    def _full_extract(self):
        if not self.running:
            self._run_rtmpose(on_done=self._full_motionbert)

    def _full_motionbert(self):
        if not self.running:
            self._run(
                "MotionBERT 3D lifting",
                self._motionbert_args(),
                on_done=self._full_analyze,
            )


try:
    import tkinter as tk
    tk_error = tk.TclError
except ImportError:
    tk_error = Exception


if __name__ == "__main__":
    app = RtmposeMocapGui()
    app.mainloop()