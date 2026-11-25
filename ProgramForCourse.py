import os
import logging
import threading
import traceback
from pathlib import Path
from enum import Enum

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

MAX_IMAGE_SIZE_FOR_DETECTION = 1280
TILE_SIZE = 640
TILE_OVERLAP = 100
NMS_IOU_THRESHOLD = 0.5

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)


class ImageType(Enum):
    AERIAL = "aerial"
    UAV = "uav"
    INFRARED = "infrared"
    THERMAL = "thermal"
    MIXED = "mixed-perspective"


DEFAULT_CLASS_NAMES = ["person", "car", "bicycle", "other_vehicle", "drone", "mine", "gun"]

CLASS_COLORS_RGB = {
    "person": (0, 255, 0),
    "car": (0, 120, 255),
    "bicycle": (255, 255, 0),
    "other_vehicle": (255, 0, 255),
    "drone": (255, 165, 0),
    "mine": (255, 0, 0),
    "gun": (128, 0, 128),
}


def rgb_to_bgr(rgb):
    return (rgb[2], rgb[1], rgb[0])


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def nms(detections, iou_threshold=NMS_IOU_THRESHOLD):
    if not detections:
        return []
    sorted_dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []
    while sorted_dets:
        best = sorted_dets.pop(0)
        kept.append(best)
        sorted_dets = [
            d for d in sorted_dets
            if not (
                d["class"] == best["class"]
                and iou(d["bbox"], best["bbox"]) > iou_threshold
            )
        ]
    return kept


class ImagePreprocessor:
    @staticmethod
    def preprocess(image, image_type):
        if image_type in (ImageType.THERMAL, ImageType.INFRARED):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            return cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
        if image_type in (ImageType.AERIAL, ImageType.UAV):
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            return cv2.filter2D(image, -1, kernel)
        return image.copy()


class DetectionRenderer:
    @staticmethod
    def draw_annotations(image, detections):
        annotated = image.copy()
        h, w = image.shape[:2]
        base = max(h, w)
        font_scale = max(0.4, min(1.2, base / 2000.0))
        thickness = max(1, int(round(font_scale * 1.5)))
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            class_name = det["class"]
            conf = det["confidence"]
            rgb = CLASS_COLORS_RGB.get(class_name, (255, 255, 255))
            bgr = rgb_to_bgr(rgb)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), bgr, thickness)
            label = f"{class_name}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
            cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 4, y1), bgr, -1)
            cv2.putText(
                annotated, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness,
            )
        return annotated


class ObjectDetector:
    def __init__(self):
        self.model = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.image_type = ImageType.MIXED
        self.class_names = list(DEFAULT_CLASS_NAMES)

    def load_model(self, model_path):
        if not YOLO_AVAILABLE:
            return False
        try:
            self.model = YOLO(model_path)
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                self.class_names = [names[i] for i in sorted(names.keys())]
            elif isinstance(names, list):
                self.class_names = list(names)
            log.info("Модель завантажено: %s", model_path)
            return True
        except Exception as e:
            log.error("Помилка завантаження моделі: %s", e)
            return False

    def detect(self, image):
        if self.model is None:
            return image, []
        h, w = image.shape[:2]
        if h <= MAX_IMAGE_SIZE_FOR_DETECTION and w <= MAX_IMAGE_SIZE_FOR_DETECTION:
            return self._detect_single(image)
        return self._detect_tiled(image)

    def _class_name(self, cls_id):
        if 0 <= cls_id < len(self.class_names):
            return self.class_names[cls_id]
        return f"Class_{cls_id}"

    def _parse_boxes(self, boxes, offset_x=0, offset_y=0):
        detections = []
        if boxes is None:
            return detections
        for box in boxes:
            tx1, ty1, tx2, ty2 = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            detections.append({
                "class": self._class_name(cls_id),
                "confidence": conf,
                "bbox": (tx1 + offset_x, ty1 + offset_y, tx2 + offset_x, ty2 + offset_y),
            })
        return detections

    def _detect_single(self, image):
        processed = ImagePreprocessor.preprocess(image, self.image_type)
        results = self.model.predict(
            source=processed,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )
        detections = self._parse_boxes(results[0].boxes)
        annotated = DetectionRenderer.draw_annotations(image, detections)
        return annotated, detections

    def _detect_tiled(self, image):
        h, w = image.shape[:2]
        log.info("Режим нарізки: %dx%d", w, h)
        all_detections = []
        step = TILE_SIZE - TILE_OVERLAP

        ys = list(range(0, max(h - TILE_SIZE, 0) + 1, step))
        xs = list(range(0, max(w - TILE_SIZE, 0) + 1, step))
        if not ys or ys[-1] + TILE_SIZE < h:
            ys.append(max(h - TILE_SIZE, 0))
        if not xs or xs[-1] + TILE_SIZE < w:
            xs.append(max(w - TILE_SIZE, 0))

        for y in ys:
            for x in xs:
                y2 = min(y + TILE_SIZE, h)
                x2 = min(x + TILE_SIZE, w)
                tile = image[y:y2, x:x2]
                if tile.size == 0:
                    continue
                pad_h = TILE_SIZE - tile.shape[0]
                pad_w = TILE_SIZE - tile.shape[1]
                if pad_h > 0 or pad_w > 0:
                    tile = cv2.copyMakeBorder(
                        tile, 0, max(pad_h, 0), 0, max(pad_w, 0),
                        cv2.BORDER_CONSTANT, value=(0, 0, 0),
                    )
                processed_tile = ImagePreprocessor.preprocess(tile, self.image_type)
                results = self.model.predict(
                    source=processed_tile,
                    conf=self.conf_threshold,
                    iou=self.iou_threshold,
                    verbose=False,
                )
                tile_dets = self._parse_boxes(results[0].boxes, offset_x=x, offset_y=y)
                all_detections.extend(tile_dets)

        all_detections = nms(all_detections)
        annotated = DetectionRenderer.draw_annotations(image, all_detections)
        return annotated, all_detections


class AppState:
    def __init__(self):
        self.current_image_bgr = None
        self.annotated_image_bgr = None
        self.current_detections = []

    def reset(self):
        self.__init__()


class CanvasViewer:
    def __init__(self, canvas):
        self.canvas = canvas
        self._photo = None
        self._pil_image = None
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._drag_start = None
        self._bind_events()

    def _bind_events(self):
        self.canvas.bind("<MouseWheel>", self._on_zoom)
        self.canvas.bind("<Button-4>", self._on_zoom)
        self.canvas.bind("<Button-5>", self._on_zoom)
        self.canvas.bind("<ButtonPress-1>", self._on_drag_start)
        self.canvas.bind("<B1-Motion>", self._on_drag_move)

    def _on_zoom(self, event):
        if self._pil_image is None:
            return
        factor = 1.1 if (event.delta > 0 or event.num == 4) else 1 / 1.1
        new_zoom = max(0.1, min(10.0, self._zoom * factor))
        if abs(new_zoom - self._zoom) < 1e-6:
            return
        mx, my = event.x, event.y
        cx = (mx - self._offset_x) / self._zoom
        cy = (my - self._offset_y) / self._zoom
        self._zoom = new_zoom
        self._offset_x = mx - cx * self._zoom
        self._offset_y = my - cy * self._zoom
        self._render()

    def _on_drag_start(self, event):
        self._drag_start = (event.x, event.y, self._offset_x, self._offset_y)

    def _on_drag_move(self, event):
        if self._drag_start is None:
            return
        sx, sy, ox, oy = self._drag_start
        self._offset_x = ox + (event.x - sx)
        self._offset_y = oy + (event.y - sy)
        self._render()

    def show(self, image_bgr=None):
        self.canvas.delete("all")
        if image_bgr is None:
            self._pil_image = None
            self._photo = None
            cw = self.canvas.winfo_width() or 400
            ch = self.canvas.winfo_height() or 250
            self.canvas.create_text(
                cw // 2, ch // 2,
                text="Натисніть «Відкрити зображення», щоб почати",
                fill="#888", font=("Segoe UI", 14),
            )
            return
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._pil_image = Image.fromarray(rgb)
        self._zoom = 1.0
        self._offset_x = 0
        self._offset_y = 0
        self._fit_initial()
        self._render()

    def _fit_initial(self):
        cw = max(self.canvas.winfo_width(), 1)
        ch = max(self.canvas.winfo_height(), 1)
        if self._pil_image is None:
            return
        scale = min(cw / self._pil_image.width, ch / self._pil_image.height, 1.0)
        self._zoom = scale
        self._offset_x = (cw - self._pil_image.width * scale) / 2
        self._offset_y = (ch - self._pil_image.height * scale) / 2

    def _render(self):
        self.canvas.delete("all")
        if self._pil_image is None:
            return
        w = max(int(self._pil_image.width * self._zoom), 1)
        h = max(int(self._pil_image.height * self._zoom), 1)
        resized = self._pil_image.resize((w, h), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(self._offset_x, self._offset_y, image=self._photo, anchor="nw")


class DetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Система виявлення об'єктів")
        self.root.geometry("1200x800")

        self.detector = ObjectDetector()
        self.state = AppState()
        self.viewer = None

        self._setup_ui()
        self._auto_load_model()

    def _setup_ui(self):
        top_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        top_frame.pack(fill="x")
        tk.Label(
            top_frame, text="Система виявлення об'єктів",
            bg="#2c3e50", fg="white", font=("Segoe UI", 16, "bold"),
        ).pack(pady=15)

        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        left = ttk.Frame(main, width=350)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        self._build_status_panel(left)
        self._build_params_panel(left)
        self._build_type_panel(left)
        self._build_actions_panel(left)

        right = ttk.Frame(main)
        right.pack(side="right", fill="both", expand=True)

        self.canvas = tk.Canvas(right, bg="#2b2b2b", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self._refresh_view())

        self.viewer = CanvasViewer(self.canvas)

        self.stats_text = tk.Text(right, height=7, font=("Consolas", 10), bg="#fafafa")
        self.stats_text.pack(fill="x", pady=(10, 0))
        self.stats_text.config(state="disabled")

    def _build_status_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Статус моделі ", padding=10)
        frame.pack(fill="x", pady=(0, 15))
        self.train_status = tk.StringVar(value="Пошук моделі...")
        ttk.Label(frame, textvariable=self.train_status, wraplength=300, justify="left").pack(anchor="w")
        ttk.Button(frame, text="Завантажити модель", command=self._load_model_manually).pack(fill="x", pady=(10, 0))

    def _build_params_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Параметри ", padding=10)
        frame.pack(fill="x", pady=(0, 15))

        ttk.Label(frame, text="Confidence:").pack(anchor="w")
        self.conf_var = tk.DoubleVar(value=0.25)
        ttk.Scale(frame, from_=0.05, to=0.95, variable=self.conf_var, orient="horizontal").pack(fill="x")
        ttk.Label(frame, textvariable=self.conf_var).pack(anchor="e")

        ttk.Label(frame, text="IoU:").pack(anchor="w", pady=(10, 0))
        self.iou_var = tk.DoubleVar(value=0.45)
        ttk.Scale(frame, from_=0.05, to=0.95, variable=self.iou_var, orient="horizontal").pack(fill="x")
        ttk.Label(frame, textvariable=self.iou_var).pack(anchor="e")

    def _build_type_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Тип зображення ", padding=10)
        frame.pack(fill="x", pady=(0, 15))
        self.image_type_var = tk.StringVar(value=ImageType.MIXED.value)
        for it in ImageType:
            ttk.Radiobutton(
                frame, text=it.value, value=it.value, variable=self.image_type_var,
            ).pack(anchor="w")

    def _build_actions_panel(self, parent):
        frame = ttk.Labelframe(parent, text=" Дії ", padding=10)
        frame.pack(fill="x", pady=(0, 15))
        ttk.Button(frame, text="Відкрити зображення", command=self._open_image).pack(fill="x", pady=(0, 10), ipady=5)
        self.btn_detect = ttk.Button(frame, text="ВИЯВИТИ ОБ'ЄКТИ", command=self._run_detection)
        self.btn_detect.pack(fill="x", pady=5, ipady=5)
        ttk.Button(frame, text="Скинути", command=self._reset).pack(fill="x", pady=(10, 0))

    def _auto_load_model(self):
        candidates = [Path("models/best.pt"), Path("best.pt"), Path("best_model_aerial.pt")]
        existing = [p for p in candidates if p.exists()]
        if not existing:
            self.train_status.set("Модель не знайдено. Покладіть best.pt у папку models.")
            return
        for path in existing:
            if self.detector.load_model(str(path)):
                self.train_status.set(f"Модель завантажена: {path.name}")
                return
        self.train_status.set(
            f"Знайдено {len(existing)} файл(ів) моделі, але жоден не завантажився. Перевірте формат."
        )

    def _load_model_manually(self):
        path = filedialog.askopenfilename(title="Виберіть файл моделі (.pt)", filetypes=[("YOLO model", "*.pt")])
        if not path:
            return
        if self.detector.load_model(path):
            self.train_status.set(f"Модель завантажена: {Path(path).name}")
            messagebox.showinfo("Успіх", "Модель успішно завантажено!")
        else:
            messagebox.showerror("Помилка", "Не вдалося завантажити модель. Перевірте файл і логи.")

    def _open_image(self):
        path = filedialog.askopenfilename(filetypes=[("Зображення", "*.jpg *.jpeg *.png *.bmp *.tif")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Помилка", "Не вдалося відкрити файл.")
            return
        self.state.current_image_bgr = img
        self.state.annotated_image_bgr = None
        self.state.current_detections = []
        self._refresh_view()

        h, w = img.shape[:2]
        mode = "нарізка" if h > MAX_IMAGE_SIZE_FOR_DETECTION or w > MAX_IMAGE_SIZE_FOR_DETECTION else "стандартний"
        self._update_stats(f"Зображення: {Path(path).name}\nРозмір: {w}x{h}\nРежим: {mode}")

    def _run_detection(self):
        if self.state.current_image_bgr is None:
            messagebox.showwarning("Увага", "Спочатку відкрийте зображення!")
            return
        if self.detector.model is None:
            messagebox.showwarning("Увага", "Модель не завантажена!")
            return

        self.detector.conf_threshold = self.conf_var.get()
        self.detector.iou_threshold = self.iou_var.get()
        self.detector.image_type = ImageType(self.image_type_var.get())

        self.btn_detect.config(state="disabled")
        threading.Thread(target=self._detect_worker, daemon=True).start()

    def _detect_worker(self):
        try:
            annotated, detections = self.detector.detect(self.state.current_image_bgr)
            self.state.annotated_image_bgr = annotated
            self.state.current_detections = detections
            self._safe_after(self._on_detection_done)
        except Exception:
            log.exception("Помилка детекції")
            self._safe_after(lambda: messagebox.showerror("Помилка", "Детекція завершилась з помилкою. Деталі в логах."))
        finally:
            self._safe_after(lambda: self.btn_detect.config(state="normal"))

    def _safe_after(self, fn):
        try:
            if self.root.winfo_exists():
                self.root.after(0, fn)
        except tk.TclError:
            pass

    def _on_detection_done(self):
        self._refresh_view()
        self._show_detections()

    def _refresh_view(self):
        img = self.state.annotated_image_bgr if self.state.annotated_image_bgr is not None else self.state.current_image_bgr
        self.viewer.show(img)

    def _show_detections(self):
        if not self.state.current_detections:
            self._update_stats("Об'єктів не виявлено.")
            return
        counts = {}
        for d in self.state.current_detections:
            counts[d["class"]] = counts.get(d["class"], 0) + 1
        lines = [f"Виявлено об'єктів: {len(self.state.current_detections)}", ""]
        for cls, n in sorted(counts.items(), key=lambda x: -x[1]):
            lines.append(f"{cls}: {n}")
        self._update_stats("\n".join(lines))

    def _update_stats(self, text):
        self.stats_text.config(state="normal")
        self.stats_text.delete("1.0", "end")
        self.stats_text.insert("1.0", text)
        self.stats_text.config(state="disabled")

    def _reset(self):
        self.state.reset()
        self._refresh_view()
        self._update_stats("")


def main():
    if not YOLO_AVAILABLE:
        log.error("Встановіть ultralytics: pip install ultralytics")
        return
    root = tk.Tk()
    DetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()