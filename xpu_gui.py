"""Windows GUI launcher for the Mixamo LLM Mocap Intel XPU pipeline."""

from __future__ import annotations

import os
import json
import queue
import socket
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


REPO = Path(__file__).resolve().parent
PYTHON = REPO / ".venv" / "Scripts" / "python.exe"
XPU_LAUNCHER = REPO / "run-xpu.py"


class MocapGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Mocap Foundry / Intel XPU")
        self.geometry("1080x760")
        self.minsize(900, 650)
        self.configure(bg="#10161d")
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.run_log_window = None
        self.run_log = None
        self.running = False
        self.progress = None
        self.vars = {
            "fbx": tk.StringVar(value=str(REPO / "ybot.fbx")),
            "rig": tk.StringVar(value=str(REPO / "ybot_rest.blend")),
            "video": tk.StringVar(value=str(REPO / "plates" / "test" / "runner.mp4")),
            "spec": tk.StringVar(value=str(REPO / "action_specs" / "test.json")),
            "landmarks": tk.StringVar(value=str(REPO / "plates" / "test" / "landmarks.json")),
            "blender": tk.StringVar(value=""),
            "device": tk.StringVar(value="Checking XPU..."),
            "mcp": tk.StringVar(value="Checking Blender MCP..."),
            "run_status": tk.StringVar(value="Ready"),
        }
        self._style()
        self._build()
        self.after(100, self._drain_log)
        self.after(250, self._refresh_status)

    def _style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#10161d")
        style.configure("Panel.TFrame", background="#17212b")
        style.configure("TLabel", background="#10161d", foreground="#d7e2ea", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#10161d", foreground="#f1f6f8", font=("Segoe UI Semibold", 22))
        style.configure("Muted.TLabel", background="#10161d", foreground="#8193a1", font=("Segoe UI", 9))
        style.configure("PanelTitle.TLabel", background="#17212b", foreground="#f1f6f8", font=("Segoe UI Semibold", 11))
        style.configure("Accent.TButton", background="#22b573", foreground="#07130e", padding=(14, 9), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#36d18a")])
        style.configure("Action.TButton", background="#253442", foreground="#e8f0f4", padding=(10, 7))
        style.map("Action.TButton", background=[("active", "#324757")])
        style.configure("TEntry", fieldbackground="#0d1319", foreground="#edf4f7", insertcolor="#edf4f7", padding=7)
        style.configure("TCheckbutton", background="#17212b", foreground="#b9c9d3")

    def _build(self) -> None:
        outer = ttk.Frame(self, padding=24)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 18))
        ttk.Label(header, text="Mocap Foundry", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="A guided Mixamo retargeting desk for Intel Arc XPU", style="Muted.TLabel").pack(anchor="w", pady=(3, 0))

        status = ttk.Frame(header)
        status.pack(side="right", anchor="e", pady=(0, 4))
        ttk.Label(status, textvariable=self.vars["device"], foreground="#44d597").pack(anchor="e")
        self.mcp_label = ttk.Label(status, textvariable=self.vars["mcp"], foreground="#f2b84b")
        self.mcp_label.pack(anchor="e", pady=(3, 0))
        ttk.Label(status, textvariable=self.vars["run_status"], foreground="#f1f6f8", font=("Segoe UI Semibold", 10)).pack(anchor="e", pady=(10, 0))
        self.progress = ttk.Progressbar(status, mode="indeterminate", length=190)
        self.progress.pack(anchor="e", pady=(4, 0))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.pack(side="left", fill="y", padx=(0, 14))
        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True)

        ttk.Label(left, text="PROJECT INPUTS", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 12))
        self._path_row(left, "Mixamo FBX", "fbx", self._pick_fbx)
        self._path_row(left, "Rig .blend", "rig", self._pick_rig)
        self._path_row(left, "Source video", "video", self._pick_video)
        self._path_row(left, "Action spec", "spec", self._pick_spec)
        self._path_row(left, "Landmarks", "landmarks", self._pick_landmarks)
        self._path_row(left, "Blender", "blender", self._pick_blender)
        ttk.Separator(left).pack(fill="x", pady=14)
        ttk.Label(left, text="The GUI calls the project venv directly. Shell activation is not required.", style="Muted.TLabel", wraplength=270).pack(anchor="w")
        self.full_run = tk.BooleanVar(value=True)
        ttk.Checkbutton(left, text="Continue through Blender / QA / comparison / render", variable=self.full_run).pack(anchor="w", pady=(14, 8))
        ttk.Button(left, text="RUN FULL WORKFLOW", style="Accent.TButton", command=self.run_full).pack(fill="x", pady=(4, 8))
        ttk.Button(left, text="Refresh status", style="Action.TButton", command=self._refresh_status).pack(fill="x")

        ttk.Label(right, text="PIPELINE STAGES", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 10))
        stages = [
            ("01", "Prepare Mixamo rig", "Build ybot_rest.blend and rig_profile.json", self.run_rig),
            ("02", "Extract landmarks", "Run GVHMR, YOLO, ViTPose and HMR2 on XPU", self.run_extract),
            ("03", "Analyze motion", "Print contact and orientation evidence", self.run_analyze),
        ]
        motionbert_preview = getattr(self, "run_motionbert_preview", None)
        if callable(motionbert_preview):
            stages.append(("04", "RTMPose + MotionBERT preview",
                           "Lift RTMPose tracks and show the combined 2D/3D preview",
                           motionbert_preview))
        first_retarget_step = 5 if callable(motionbert_preview) else 4
        stages.extend([
            (f"{first_retarget_step:02d}", "Retarget motion", "Build joints_mixamo.json from the action spec", self.run_lift),
            (f"{first_retarget_step + 1:02d}", "Apply in Blender", "Send animation stages through MCP at localhost:9876", self.run_blender),
            (f"{first_retarget_step + 2:02d}", "QA clip", "Check curves for pops, foot skate, root drift, and rest pose", self.run_qa),
            (f"{first_retarget_step + 3:02d}", "Compare source", "Measure retarget pose differences against the source landmarks", self.run_comparison),
            (f"{first_retarget_step + 4:02d}", "Render showcase", "Create and open preview.mp4 and source-vs-retarget showcase.mp4", self.run_preview),
        ])
        for number, title, detail, command in stages:
            row = ttk.Frame(right, style="Panel.TFrame", padding=12)
            row.pack(fill="x", pady=4)
            ttk.Label(row, text=number, background="#22b573", foreground="#07130e", width=4, anchor="center", font=("Segoe UI Semibold", 10)).pack(side="left", padx=(0, 12))
            text = ttk.Frame(row, style="Panel.TFrame")
            text.pack(side="left", fill="x", expand=True)
            ttk.Label(text, text=title, style="PanelTitle.TLabel").pack(anchor="w")
            ttk.Label(text, text=detail, style="Muted.TLabel").pack(anchor="w", pady=(2, 0))
            ttk.Button(row, text="Run", style="Action.TButton", command=command).pack(side="right")

        ttk.Label(right, text="LIVE OUTPUT", style="PanelTitle.TLabel").pack(anchor="w", pady=(18, 8))
        log_frame = ttk.Frame(right, style="Panel.TFrame", padding=2)
        log_frame.pack(fill="both", expand=True)
        self.log = tk.Text(log_frame, bg="#0b1117", fg="#c5d5df", insertbackground="#c5d5df", relief="flat", wrap="word", font=("Consolas", 9), padx=12, pady=10)
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

    def _path_row(self, parent, label: str, key: str, picker) -> None:
        ttk.Label(parent, text=label, style="Muted.TLabel").pack(anchor="w", pady=(7, 3))
        row = ttk.Frame(parent, style="Panel.TFrame")
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.vars[key], width=29).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="...", width=3, command=picker).pack(side="right", padx=(5, 0))

    def _pick(self, key: str, **kwargs) -> None:
        value = filedialog.askopenfilename(**kwargs)
        if value:
            self.vars[key].set(value)

    def _pick_fbx(self): self._pick("fbx", filetypes=[("FBX files", "*.fbx"), ("All files", "*.*")])
    def _pick_rig(self): self._pick("rig", filetypes=[("Blender files", "*.blend"), ("All files", "*.*")])
    def _pick_video(self): self._pick("video", filetypes=[("Video files", "*.mp4 *.mov *.avi"), ("All files", "*.*")])
    def _pick_spec(self): self._pick("spec", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    def _pick_landmarks(self): self._pick("landmarks", filetypes=[("JSON files", "*.json"), ("All files", "*.*")])
    def _pick_blender(self): self._pick("blender", filetypes=[("Blender executable", "blender.exe"), ("All files", "*.*")])

    def _python(self, *args: str) -> list[str]:
        return [str(PYTHON), *args]

    def _check_paths(self, *keys: str) -> bool:
        missing = [self.vars[k].get() for k in keys if not Path(self.vars[k].get()).exists()]
        if missing:
            messagebox.showerror("Missing input", "Select or create:\n\n" + "\n".join(missing))
            return False
        return True

    def _run(self, title: str, args: list[str], cwd: Path = REPO, on_done=None) -> None:
        if self.running:
            messagebox.showinfo("Pipeline busy", "A pipeline step is already in progress. Please wait for it to finish.")
            return
        self.running = True
        self.vars["run_status"].set(f"In progress: {title}")
        if self.progress is not None:
            self.progress.start(12)
        self._open_run_log(title, args)
        self._write(f"\n>>> {title}\nStarting this step. It may take a while; live output will appear below.\nCommand: {' '.join(args)}\n")
        completion_callback = on_done

        def worker():
            success = False
            try:
                env = os.environ.copy()
                env["PYTHONPATH"] = str(REPO / "tools" / "GVHMR") + os.pathsep + env.get("PYTHONPATH", "")
                process = subprocess.Popen(args, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
                assert process.stdout is not None
                for line in process.stdout:
                    self.log_queue.put(line)
                code = process.wait()
                success = code == 0
                self.log_queue.put(f"\n{'Step finished successfully.' if success else 'Step finished with errors.'} Exit code: {code}\n")
            except Exception as exc:
                self.log_queue.put(f"\nStep could not finish: {exc}\n")
            finally:
                self.running = False
                self.log_queue.put("__READY__")
                if success and completion_callback:
                    self.after(0, completion_callback)

        threading.Thread(target=worker, daemon=True).start()

    def _write(self, text: str) -> None:
        self.log.insert("end", text)
        self.log.see("end")
        if self.run_log is not None and self.run_log.winfo_exists():
            self.run_log.insert("end", text)
            self.run_log.see("end")

    def _open_run_log(self, title: str, args: list[str]) -> None:
        if self.run_log_window is not None and self.run_log_window.winfo_exists():
            self.run_log_window.destroy()
        self.run_log_window = tk.Toplevel(self)
        self.run_log_window.title(f"Run log: {title}")
        self.run_log_window.geometry("760x420")
        self.run_log = tk.Text(
            self.run_log_window,
            bg="#0b1117",
            fg="#c5d5df",
            insertbackground="#c5d5df",
            relief="flat",
            wrap="word",
            font=("Consolas", 9),
            padx=12,
            pady=10,
        )
        self.run_log.pack(fill="both", expand=True)
        self.run_log.insert("end", f">>> {title}\n{' '.join(args)}\n\n")

    def _drain_log(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                if line == "__READY__":
                    self.running = False
                    if self.progress is not None:
                        self.progress.stop()
                    self.vars["run_status"].set("Ready")
                else:
                    self._write(line)
        except queue.Empty:
            pass
        self.after(100, self._drain_log)

    def _refresh_status(self) -> None:
        if PYTHON.exists():
            probe = [str(PYTHON), "-c", "import torch; print(torch.xpu.get_device_name(0) if hasattr(torch, 'xpu') and torch.xpu.is_available() else 'XPU unavailable')"]
            try:
                result = subprocess.run(probe, cwd=REPO, capture_output=True, text=True, timeout=8)
                self.vars["device"].set("XPU / " + result.stdout.strip() if result.returncode == 0 else "XPU check failed")
            except Exception:
                self.vars["device"].set("XPU check failed")
        else:
            self.vars["device"].set(".venv Python missing")
        try:
            with socket.create_connection(("localhost", 9876), timeout=0.3):
                self.vars["mcp"].set("MCP bridge / connected")
                self.mcp_label.configure(foreground="#44d597")
        except OSError:
            self.vars["mcp"].set("MCP bridge / offline")
            self.mcp_label.configure(foreground="#f05d5e")

    def run_rig(self):
        if self._check_paths("fbx"):
            blender = self.vars["blender"].get() or "blender"
            self._run("Prepare Mixamo rig", [blender, "--background", "--python", str(REPO / "pipeline" / "setup_rig.py"), "--", "--fbx", self.vars["fbx"].get(), "--out", self.vars["rig"].get()])

    def run_extract(self):
        if self._check_paths("video"):
            self._run("Extract landmarks on Intel XPU", [str(PYTHON), str(XPU_LAUNCHER), str(REPO / "pipeline" / "estimate_pose_gvhmr.py"), "--video", self.vars["video"].get(), "--out", self.vars["landmarks"].get()])

    def run_analyze(self):
        if self._check_paths("landmarks"):
            self._run("Analyze motion", self._python(str(REPO / "pipeline" / "analyze_landmarks.py"), "--landmarks", self.vars["landmarks"].get()))

    def run_lift(self):
        if self._check_paths("spec", "landmarks"):
            self._run("Retarget motion", self._python(str(REPO / "pipeline" / "lift_to_mixamo.py"), "--spec", self.vars["spec"].get()))

    def run_blender(self):
        if self._check_paths("spec"):
            self._run("Apply animation in Blender", self._python(str(REPO / "pipeline" / "run_in_blender.py"), "all", self.vars["spec"].get()))

    def run_qa(self):
        if self._check_paths("spec"):
            self._run("QA", self._python(str(REPO / "pipeline" / "qa_clip.py"), "--spec", self.vars["spec"].get()))

    def run_comparison(self):
        if self._check_paths("spec"):
            self._run("Compare retarget to source", self._python(
                str(REPO / "pipeline" / "compare_reference.py"), "--spec", self.vars["spec"].get()))

    def run_preview(self):
        if self._check_paths("spec", "video"):
            self._run("Render showcase", self._python(
                str(REPO / "pipeline" / "render_preview.py"), self.vars["spec"].get(),
                "--showcase", "--video", self.vars["video"].get()),
                on_done=self._open_showcase)

    def _open_showcase(self):
        try:
            spec = json.loads(Path(self.vars["spec"].get()).read_text(encoding="utf-8"))
            clip_dir = Path(spec["clip_dir"])
            clip_dir = clip_dir if clip_dir.is_absolute() else REPO / clip_dir
            showcase = clip_dir / "showcase.mp4"
            if not showcase.exists():
                self._write(f"Showcase render completed but output is missing: {showcase}\n")
                return
            os.startfile(str(showcase))
            self._write(f"Opened showcase video: {showcase}\n")
        except (KeyError, OSError, ValueError, TypeError) as exc:
            self._write(f"Could not open showcase video: {exc}\n")

    def run_full(self):
        if self._check_paths("fbx", "video", "spec"):
            blender = self.vars["blender"].get() or "blender"
            self._run("Prepare Mixamo rig", [blender, "--background", "--python", str(REPO / "pipeline" / "setup_rig.py"), "--", "--fbx", self.vars["fbx"].get(), "--out", self.vars["rig"].get()], on_done=self._full_extract)

    def _full_extract(self):
        if not self.running:
            self._run("Extract landmarks on Intel XPU", [str(PYTHON), str(XPU_LAUNCHER), str(REPO / "pipeline" / "estimate_pose_gvhmr.py"), "--video", self.vars["video"].get(), "--out", self.vars["landmarks"].get()], on_done=self._full_analyze)

    def _full_analyze(self):
        if not self.running:
            self._run("Analyze motion", self._python(str(REPO / "pipeline" / "analyze_landmarks.py"), "--landmarks", self.vars["landmarks"].get()), on_done=self._full_lift)

    def _full_lift(self):
        if not self.running:
            self._run("Retarget motion", self._python(str(REPO / "pipeline" / "lift_to_mixamo.py"), "--spec", self.vars["spec"].get()), on_done=self._full_blender)

    def _full_blender(self):
        if not self.running and self.full_run.get():
            self._run("Apply animation in Blender", self._python(str(REPO / "pipeline" / "run_in_blender.py"), "all", self.vars["spec"].get()), on_done=self._full_outputs)

    def _full_outputs(self):
        if not self.running and self.full_run.get():
            self._run("QA", self._python(str(REPO / "pipeline" / "qa_clip.py"),
                      "--spec", self.vars["spec"].get()), on_done=self._full_compare)

    def _full_compare(self):
        if not self.running and self.full_run.get():
            self._run("Compare retarget to source", self._python(
                str(REPO / "pipeline" / "compare_reference.py"), "--spec",
                self.vars["spec"].get()), on_done=self._full_preview)

    def _full_preview(self):
        if not self.running and self.full_run.get() and self._check_paths("video"):
            self._run("Render showcase", self._python(
                str(REPO / "pipeline" / "render_preview.py"), self.vars["spec"].get(),
                "--showcase", "--video", self.vars["video"].get()), on_done=self._open_showcase)


if __name__ == "__main__":
    app = MocapGui()
    app.mainloop()
