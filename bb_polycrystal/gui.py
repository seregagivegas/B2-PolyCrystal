"""Русский настольный интерфейс и геометрическая ручная правка масок.

Автоматический конвейер сосредоточен в core.py. Здесь хранятся только
состояние интерфейса, фоновые задания и явно инициированные пользователем правки.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QAction, QColor, QImage, QKeySequence, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGraphicsScene, QGraphicsView,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSizePolicy, QSpinBox, QSplitter, QTabWidget, QTableWidget, QTableWidgetItem,
    QToolButton, QVBoxLayout, QWidget,
)

from .core import (
    Config, Engine, evaluate_labels, load_image, load_result, refresh_result,
    save_result, stage_images, refilter_result, CLASS_COLORS, draw_label_boundaries,
)


ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
FILTER_IMAGES = "Изображения (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.webp);;Все файлы (*)"
STAGES = [
    ("Original", "01  Исходное"),
    ("Preprocessed", "02  Подготовка"),
    ("Edge map", "03  Границы"),
    ("Raw segmentation", "04  Маски SAM"),
    ("Postprocessed segmentation", "05  Отбор"),
    ("Overlay on original", "06  Правка"),
]
CLASS_NAMES = ["Зерно", "Артефакт", "Неопределено"]
HEADERS = {
    "id": "ID", "label": "ID", "class": "Класс", "class_name": "Класс",
    "area_px": "Площадь, px²", "area_px2": "Площадь, px²", "area_um2": "Площадь, мкм²",
    "perimeter_px": "Периметр, px", "perimeter_um": "Периметр, мкм",
    "centroid_x": "Центр X, px", "centroid_y": "Центр Y, px",
    "centroid_x_px": "Центр X, px", "centroid_y_px": "Центр Y, px",
    "major_axis_px": "Большая ось, px", "minor_axis_px": "Малая ось, px",
    "major_axis_length_px": "Большая ось, px", "minor_axis_length_px": "Малая ось, px",
    "major_axis_um": "Большая ось, мкм", "minor_axis_um": "Малая ось, мкм",
    "eccentricity": "Эксцентриситет", "feret_diameter_px": "Диаметр Фере, px",
    "feret_diameter_um": "Диаметр Фере, мкм", "feret_px": "Диаметр Фере, px",
    "feret_um": "Диаметр Фере, мкм", "boundary_homogeneity": "Однородность границы",
    "boundary_mean": "Средний градиент границы", "boundary_std": "Разброс градиента",
    "boundary_gradient_mean": "Средний градиент границы",
    "boundary_gradient_std": "Разброс градиента", "touches_border": "На краю",
    "equivalent_diameter_px": "Экв. диаметр, px", "equivalent_diameter_um": "Экв. диаметр, мкм",
    "solidity": "Заполненность", "mean_intensity": "Средняя яркость",
    "centroid_x_um": "Центр X, мкм", "centroid_y_um": "Центр Y, мкм",
    "feret_max_px": "Диаметр Фере, px", "feret_max_um": "Диаметр Фере, мкм",
    "boundary_intensity_mean": "Яркость границы · среднее",
    "boundary_intensity_std": "Яркость границы · σ",
    "edge_strength_mean": "Градиент границы · среднее",
    "edge_strength_std": "Градиент границы · σ",
    "predicted_iou": "Оценка IoU от SAM",
}




STYLE = """
QWidget { background: #F8E5D7; color: #432720; font-family: "Arial"; font-size: 12px; }
QMenuBar, QMenu { background: #FDF4EC; border: none; }
QMenuBar::item { padding: 7px 12px; }
QMenu::item { padding: 8px 24px; }
QMenuBar::item:selected, QMenu::item:selected { background: #ECAAB2; color: #65232B; }
QLabel { background: transparent; }
QLabel#Brand { font-size: 25px; font-weight: 700; color: #65232B; }
QLabel#Subtitle, QLabel#Muted { color: #85665C; }
QLabel#ImageTitle { font-size: 21px; font-weight: 700; color: #65232B; }
QLabel#CardValue { font-size: 22px; font-weight: 700; color: #65232B; }
QLabel#CardLabel { color: #85665C; font-size: 11px; }
QLabel#SectionTitle { color: #65232B; font-weight: 700; }
QFrame#Card { background: #FFF9F2; border: 1px solid #DFC3B4; border-radius: 12px; }
QFrame#Card QLabel { background: transparent; }
QGroupBox { background: #FDF1E7; border: 1px solid #DFC3B4; border-radius: 10px;
    margin-top: 15px; padding: 14px 11px 11px; font-weight: 600; color: #65232B; }
QGroupBox::title { subcontrol-origin: margin; left: 11px; padding: 0 5px; color: #65232B; background: #F8E5D7; }
QPushButton, QToolButton { background: #FFF9F2; border: 1px solid #D6ADA6;
    border-radius: 7px; padding: 8px 11px; color: #432720; }
QPushButton:hover, QToolButton:hover { background: #F4CDD0; border-color: #B97C83; }
QPushButton:pressed, QToolButton:pressed { background: #ECAAB2; }
QPushButton:disabled, QToolButton:disabled { background: #F2E4D9; color: #AA9186; border-color: #E0CFC2; }
QPushButton#Primary { background: #65232B; color: #FFF9F2; border-color: #65232B;
    font-weight: 700; padding: 10px 12px; }
QPushButton#Primary:hover { background: #813640; border-color: #813640; }
QPushButton#Primary:pressed { background: #4B1820; }
QPushButton#Primary:disabled { background: #DAB0AE; color: #785951; border-color: #DAB0AE; }
QToolButton#Advanced { text-align: left; background: #FDF1E7; color: #65232B; font-weight: 700; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #FFF9F2; color: #432720;
    border: 1px solid #DCC1B1; border-radius: 6px; padding: 6px 7px;
    selection-background-color: #ECAAB2; selection-color: #432720; min-height: 19px; }
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border: 1px solid #B66E7A; }
QComboBox QAbstractItemView { background: #FFF9F2; selection-background-color: #ECAAB2; selection-color: #432720; }
QCheckBox { spacing: 8px; color: #432720; }
QCheckBox::indicator { width: 17px; height: 17px; background: #FFF9F2; border: 1px solid #CBA39C; border-radius: 5px; }
QCheckBox::indicator:checked { background: #ECAAB2; border: 2px solid #65232B; }
QCheckBox::indicator:disabled { background: #E5D3C6; border-color: #CBB4A7; }
QListWidget, QTableWidget { background: #FFF9F2; color: #432720; border: 1px solid #DCC1B1;
    border-radius: 8px; alternate-background-color: #F9EAE0; gridline-color: #E8D3C6;
    selection-background-color: #F1C3C5; selection-color: #65232B; }
QListWidget::item { padding: 9px 6px; border-bottom: 1px solid #EEDBD0; }
QListWidget::item:selected, QTableWidget::item:selected { background: #F1C3C5; color: #65232B; }
QHeaderView::section { background: #F2DDD0; color: #65232B; border: none; border-bottom: 1px solid #DCC1B1;
    padding: 9px; font-weight: 700; }
QTabWidget::pane { border: 1px solid #DCC1B1; background: #FDF1E7; border-radius: 9px; top: -1px; }
QTabBar { background: #65232B; }
QTabBar::tab { background: #65232B; color: #F8E5D7; padding: 11px 9px;
    border: none; border-bottom: 3px solid transparent; }
QTabBar::tab:selected { background: #65232B; color: #FFFFFF;
    border-bottom: 3px solid #ECAAB2; font-weight: 700; }
QTabBar::tab:hover:!selected { background: #813640; color: #FFFFFF; }
QTabBar QToolButton { background: #65232B; color: #F8E5D7; border: none; }

QGraphicsView { background: #F3DED0; border: none; }
QScrollArea { background: transparent; border: none; }
QScrollBar:vertical { background: #F0D8CB; width: 9px; margin: 2px; border-radius: 4px; }
QScrollBar:horizontal { background: #F0D8CB; height: 9px; margin: 2px; border-radius: 4px; }
QScrollBar::handle { background: #ECAAB2; min-height: 30px; min-width: 30px; border-radius: 4px; }
QScrollBar::handle:hover { background: #D48D99; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QSplitter::handle { background: #DCC1B1; height: 5px; }
QSplitter::handle:hover { background: #ECAAB2; }
QProgressBar { background: #ECD1C4; border: none; border-radius: 5px; text-align: center; min-height: 16px; color: #432720; }
QProgressBar::chunk { background: #ECAAB2; border-radius: 5px; }
QStatusBar { background: #EFD7C9; color: #785951; border-top: 1px solid #DCC1B1; }
QStatusBar::item { border: none; }
QToolTip { background: #432720; color: #FFF9F2; border: 1px solid #65232B; padding: 6px; }
"""



def pixmap(rgb: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray(rgb, dtype=np.uint8)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    h, w = a.shape[:2]
    return QPixmap.fromImage(QImage(a.data, w, h, a.strides[0], QImage.Format.Format_RGB888).copy())


def fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, (bool, np.bool_)):
        return "Да" if value else "Нет"
    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return "—"
        if abs(value) >= 1_000_000_000:
            return f"{value:.4e}"
        if float(value).is_integer():
            return str(int(value))
        # Up to nine integer digits are always shown without an exponent.
        decimals = max(4, min(12, 4 - int(np.floor(np.log10(abs(value)))))) if value else 4
        return f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    return str(value)


class ImageFiles(QListWidget):
    filesDropped = Signal(list)

    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setMinimumHeight(130)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        self.filesDropped.emit(paths)
        event.acceptProposedAction()


class NumericItem(QTableWidgetItem):
    def __lt__(self, other):
        a = self.data(Qt.ItemDataRole.UserRole + 1)
        b = other.data(Qt.ItemDataRole.UserRole + 1)
        if a is not None and b is not None:
            return a < b
        return super().__lt__(other)


class ImageView(QGraphicsView):
    imageClicked = Signal(int, int, bool)
    pathFinished = Signal(list)
    cursorPosition = Signal(int, int)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.picture = None
        self.mode = "navigate"
        self.points: list[tuple[int, int]] = []
        self.annotations = []
        self.auto_fit = True

    def set_image(self, rgb, reset=True):
        previous = self.transform()
        self.scene().clear()
        self.annotations.clear()
        self.picture = self.scene().addPixmap(pixmap(rgb))
        self.scene().setSceneRect(self.picture.boundingRect())
        self.points.clear()
        if reset or self.auto_fit:
            self.fit_image()
        else:
            self.setTransform(previous)

    def fit_image(self):
        self.auto_fit = True
        if self.picture:
            self.fitInView(self.picture, Qt.AspectRatioMode.KeepAspectRatio)

    def zoom(self, factor):
        if self.picture:
            current = self.transform().m11()
            if 0.03 <= current * factor <= 30:
                self.auto_fit = False
                self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.auto_fit:
            self.fit_image()

    def wheelEvent(self, event):
        self.zoom(1.2 if event.angleDelta().y() > 0 else 1 / 1.2)
        event.accept()

    def set_mode(self, mode):
        self.cancel_path()
        self.mode = mode
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag if mode == "navigate" else QGraphicsView.DragMode.NoDrag)
        self.setCursor(Qt.CursorShape.OpenHandCursor if mode == "navigate" else Qt.CursorShape.CrossCursor)

    def point_at(self, position):
        p = self.mapToScene(position)
        if not self.picture or not self.picture.boundingRect().contains(p):
            return None
        return int(p.x()), int(p.y())

    def mousePressEvent(self, event):
        point = self.point_at(event.position().toPoint())
        if event.button() == Qt.MouseButton.LeftButton and point and self.mode != "navigate":
            if self.mode == "select":
                self.imageClicked.emit(*point, bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier))
            else:
                if not self.points or self.points[-1] != point:
                    self.points.append(point)
                    self.draw_path()
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton and self.mode in ("polygon", "reshape", "split"):
            self.cancel_path()
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        point = self.point_at(event.position().toPoint())
        if point:
            self.cursorPosition.emit(*point)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.mode in ("polygon", "reshape", "split"):
            self.finish_path()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and self.mode in ("polygon", "reshape", "split"):
            self.finish_path()
        elif event.key() == Qt.Key.Key_Escape:
            self.cancel_path()
        elif event.key() == Qt.Key.Key_Backspace and self.points:
            self.points.pop()
            self.draw_path()
        else:
            super().keyPressEvent(event)

    def draw_path(self):
        for item in self.annotations:
            self.scene().removeItem(item)
        self.annotations.clear()
        pen = QPen(QColor("#ECAAB2"), 0)
        pen.setWidthF(2)
        pen.setCosmetic(True)
        for a, b in zip(self.points, self.points[1:]):
            self.annotations.append(self.scene().addLine(*a, *b, pen))
        radius = 3 / max(0.05, self.transform().m11())
        for x, y in self.points:
            self.annotations.append(self.scene().addEllipse(x - radius, y - radius, 2 * radius, 2 * radius, pen))

    def cancel_path(self):
        self.points.clear()
        for item in self.annotations:
            self.scene().removeItem(item)
        self.annotations.clear()

    def finish_path(self):
        points = list(self.points)
        self.cancel_path()
        if len(points) >= (2 if self.mode == "split" else 3):
            self.pathFinished.emit(points)


class SegmentWorker(QThread):
    progress = Signal(int, int, str)
    succeeded = Signal(str, object)
    failed = Signal(str, str)
    complete = Signal()

    def __init__(self, checkpoint, config, paths, output_root=None, crop_by_path=None, scale_by_path=None):
        super().__init__()
        self.checkpoint, self.config = Path(checkpoint), config
        self.paths = list(paths)
        self.output_root = output_root
        self.crop_by_path = crop_by_path or {}
        self.scale_by_path = scale_by_path or {}
        self.cancelled = False

    def run(self):
        try:
            engine = Engine(self.checkpoint)
        except Exception as exc:
            self.failed.emit("Модель", f"{type(exc).__name__}: {exc}")
            self.complete.emit()
            return
        attempted = 0
        for index, path in enumerate(self.paths):
            if self.isInterruptionRequested():
                self.cancelled = True
                break
            self.progress.emit(index, len(self.paths), Path(path).name)
            try:
                config = replace(self.config, crop_bottom=self.crop_by_path.get(str(path), self.config.crop_bottom), scale_um_per_px=self.scale_by_path.get(str(path), self.config.scale_um_per_px))
                result = engine.segment(Path(path), config)
                if self.output_root:
                    folder = unique_folder(Path(self.output_root), Path(path).stem)
                    save_result(result, folder)
                self.succeeded.emit(str(path), result)
            except Exception as exc:
                self.failed.emit(Path(path).name, f"{type(exc).__name__}: {exc}")
            attempted = index + 1
        self.progress.emit(attempted, len(self.paths), "Остановлено" if self.cancelled else "Завершено")
        self.complete.emit()


def unique_folder(root: Path, name: str) -> Path:
    folder = root / name
    number = 2
    while folder.exists():
        folder = root / f"{name}_{number:02d}"
        number += 1
    return folder


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self.setWindowTitle("B² PolyCrystal · анализ зёрен")
        self.resize(1440, 960)
        self.setMinimumSize(1100, 740)
        self.result = None
        self.results = {}
        self.worker = None
        self.busy = False
        self.selected_ids: set[int] = set()
        self.history = []
        self.histories = {}
        self.dirty_results = set()
        self.crop_by_path = {}
        self.scale_by_path = {}
        self.loading_config = False
        try:
            self.crop_manifest = json.loads((ROOT / "data" / "crop_manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.crop_manifest = {}
        self.dirty = False
        self.table_keys = []
        self.failed_jobs = []
        self._build_ui()
        self._build_menu()
        self._connect()
        self.set_busy(False)
        self.statusBar().showMessage("Добавьте SEM-снимки. Параметры съёмки и масштаб задаются пользователем.")

    def _build_ui(self):
        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(14)
        self.setCentralWidget(central)
        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side = QWidget()
        s = QVBoxLayout(side)
        s.setContentsMargins(0, 0, 8, 0)
        s.setSpacing(11)
        brand = QLabel("B² PolyCrystal")
        brand.setObjectName("Brand")
        s.addWidget(brand)
        sub = QLabel("Сегментация и измерение зёрен")
        sub.setObjectName("Subtitle")
        s.addWidget(sub)

        files = QGroupBox("01 / Изображения")
        f = QVBoxLayout(files)
        self.add_button = QPushButton("＋  Добавить изображения")
        f.addWidget(self.add_button)
        hint = QLabel("Или перетащите файлы сюда · Ctrl для выбора")
        hint.setWordWrap(True)
        hint.setObjectName("Muted")
        f.addWidget(hint)
        self.files = ImageFiles()
        self.files.setFixedHeight(125)
        f.addWidget(self.files)
        frow = QHBoxLayout()
        self.remove_button = QPushButton("Убрать")
        self.open_session_button = QPushButton("Открыть сессию")
        frow.addWidget(self.remove_button)
        frow.addWidget(self.open_session_button)
        f.addLayout(frow)
        s.addWidget(files)

        model = QGroupBox("02 / Подготовка снимка")
        m = QFormLayout(model)
        m.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.crop = QSpinBox()
        self.crop.setRange(0, 100000)
        self.crop.setSuffix(" px")
        self.crop.setToolTip("Число удаляемых нижних строк. Проверьте по исходному снимку: служебная полоса не должна попадать в анализ.")
        m.addRow("Срез снизу", self.crop)
        self.crop_suggest = QPushButton("Срез набора / 0 для нового снимка")
        self.crop_suggest.setToolTip("Для предоставленного набора — проверенное значение crop_manifest.json. Для новых файлов срез задайте вручную.")
        m.addRow(self.crop_suggest)
        self.clahe = QDoubleSpinBox()
        self.clahe.setRange(0, 20)
        self.clahe.setSpecialValueText("Выкл.")
        self.clahe.setValue(2.0)
        self.clahe.setSingleStep(0.2)
        m.addRow("CLAHE · clip", self.clahe)
        self.denoise = QCheckBox("Подавлять мелкий шум · 3 × 3")
        self.denoise.setToolTip("Медианный фильтр перед CLAHE. Включайте при импульсном шуме; тонкие границы могут сгладиться.")
        m.addRow(self.denoise)
        s.addWidget(model)

        false_group = QGroupBox("03 / Ложные объекты")
        filters = QFormLayout(false_group)
        self.min_area = QSpinBox()
        self.min_area.setRange(0, 10000000)
        self.min_area.setSuffix(" px²")
        self.min_area.setValue(16)
        self.min_area.setToolTip("0 — фильтр площади выключен. Ненулевой порог допустим только при обосновании разрешением или разметкой.")
        filters.addRow("Мин. площадь", self.min_area)
        self.max_area = QDoubleSpinBox()
        self.max_area.setRange(1, 100)
        self.max_area.setDecimals(0)
        self.max_area.setSuffix(" % поля")
        self.max_area.setValue(85)
        self.max_area.setToolTip("Отбрасывать маски больше этой доли поля до разрешения пересечений. 100 % выключает фильтр. Защита от маски всего кадра работает независимо от исключения края.")
        filters.addRow("Макс. площадь", self.max_area)
        s.addWidget(false_group)

        border_group = QGroupBox("04 / Объекты у края")
        borders = QFormLayout(border_group)
        self.exclude_border = QCheckBox("Исключать объекты на краях")
        self.exclude_border.setChecked(True)
        borders.addRow(self.exclude_border)
        self.border_margin = QSpinBox()
        self.border_margin.setRange(0, 100)
        self.border_margin.setValue(1)
        self.border_margin.setSuffix(" px")
        self.border_margin.setToolTip("0 — касание самой рамки; 1 — также зазор в один пиксель. Проверяются все четыре края рабочей области после среза.")
        borders.addRow("Отступ от рамки", self.border_margin)
        self.filter_button = QPushButton("Применить фильтры к маскам")
        self.filter_button.setToolTip("Пересчитать отбор из исходных масок SAM без повторного запуска модели. Ручную геометрию можно восстановить отменой.")
        borders.addRow(self.filter_button)
        s.addWidget(border_group)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setObjectName("Advanced")
        self.advanced_toggle.setText("Параметры MobileSAM · CPU")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.advanced_toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        s.addWidget(self.advanced_toggle)
        self.advanced_panel = QWidget()
        advanced_form = QFormLayout(self.advanced_panel)
        advanced_form.setContentsMargins(7, 0, 7, 0)
        cp_row = QHBoxLayout()
        self.checkpoint = QLineEdit(str(ROOT / "models" / "mobile_sam.pt"))
        self.checkpoint.setToolTip("Веса MobileSAM. Инструкция получения — в README проекта.")
        self.checkpoint_button = QPushButton("…")
        self.checkpoint_button.setFixedWidth(32)
        cp_row.addWidget(self.checkpoint)
        cp_row.addWidget(self.checkpoint_button)
        advanced_form.addRow("Веса", cp_row)
        self.points = QSpinBox()
        self.points.setRange(4, 64)
        self.points.setValue(24)
        self.points.setToolTip("Количество точек SAM по каждой стороне. Увеличение повышает время и расход памяти.")
        advanced_form.addRow("Сетка SAM", self.points)
        self.iou = QDoubleSpinBox()
        self.iou.setRange(0, 1)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(Config().pred_iou_thresh)
        advanced_form.addRow("Порог pred. IoU", self.iou)
        self.stability = QDoubleSpinBox()
        self.stability.setRange(0, 1)
        self.stability.setSingleStep(0.05)
        self.stability.setValue(Config().stability_score_thresh)
        advanced_form.addRow("Стабильность", self.stability)
        self.nms = QDoubleSpinBox()
        self.nms.setRange(0, 1)
        self.nms.setSingleStep(0.05)
        self.nms.setValue(Config().box_nms_thresh)
        self.nms.setToolTip("Порог IoU рамок в стандартном подавлении пересекающихся предложений SAM. Меньшее значение удаляет больше пересекающихся кандидатов.")
        advanced_form.addRow("Порог NMS", self.nms)
        self.advanced_panel.hide()
        self.advanced_toggle.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_toggle.toggled.connect(lambda opened: self.advanced_toggle.setArrowType(Qt.ArrowType.DownArrow if opened else Qt.ArrowType.RightArrow))
        s.addWidget(self.advanced_panel)

        scale_group = QGroupBox("05 / Физический масштаб")
        sc = QFormLayout(scale_group)
        self.scale_enabled = QCheckBox("Использовать масштабную линейку")
        sc.addRow(self.scale_enabled)
        self.scale_px = QDoubleSpinBox()
        self.scale_px.setDecimals(3)
        self.scale_px.setRange(0.001, 1000000)
        self.scale_px.setValue(100)
        self.scale_px.setSuffix(" px")
        self.scale_um = QDoubleSpinBox()
        self.scale_um.setDecimals(6)
        self.scale_um.setRange(0.000001, 1000000)
        self.scale_um.setValue(1)
        self.scale_um.setSuffix(" мкм")
        sc.addRow("Длина на снимке", self.scale_px)
        sc.addRow("Реальная длина", self.scale_um)
        self.scale_label = QLabel("Без масштаба: измерения в пикселях")
        self.scale_label.setWordWrap(True)
        self.scale_label.setObjectName("Muted")
        sc.addRow(self.scale_label)
        self.apply_scale_button = QPushButton("Применить к текущему результату")
        sc.addRow(self.apply_scale_button)
        s.addWidget(scale_group)

        self.run_button = QPushButton("▶  Обработать текущий снимок")
        self.run_button.setObjectName("Primary")
        self.batch_button = QPushButton("Обработать выбранные / все")
        self.cancel_button = QPushButton("Остановить после текущего файла")
        self.cancel_button.hide()
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        note = QLabel("SAM предлагает маски. Проверяйте границы по исходному снимку и отмечайте артефакты вручную.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        s.addWidget(note)
        s.addStretch()
        sidebar_scroll.setWidget(side)
        left = QWidget()
        left.setFixedWidth(324)
        left_box = QVBoxLayout(left)
        left_box.setContentsMargins(0, 0, 0, 0)
        left_box.setSpacing(8)
        left_box.addWidget(sidebar_scroll, 1)
        for widget in [self.run_button, self.batch_button, self.cancel_button, self.progress]:
            left_box.addWidget(widget)
        layout.addWidget(left)

        right = QVBoxLayout()
        right.setSpacing(10)
        head = QHBoxLayout()
        self.image_title = QLabel("Рабочая область")
        self.image_title.setObjectName("ImageTitle")
        self.image_details = QLabel("Изображение не загружено")
        self.image_details.setObjectName("Muted")
        head_text = QVBoxLayout()
        head_text.addWidget(self.image_title)
        self.image_details.setWordWrap(True)
        head_text.addWidget(self.image_details)
        head.addLayout(head_text, 1)
        self.save_button = QPushButton("↓  Сохранить")
        self.save_button.setObjectName("Primary")
        head.addWidget(self.save_button)
        right.addLayout(head)

        cards = QHBoxLayout()
        self.card_values = []
        for label in ["Объекты / зёрна", "Средняя площадь", "Средний диаметр Фере", "Масштаб"]:
            card = QFrame()
            card.setObjectName("Card")
            box = QVBoxLayout(card)
            box.setContentsMargins(14, 9, 14, 9)
            value = QLabel("—")
            value.setObjectName("CardValue")
            caption = QLabel(label)
            caption.setObjectName("CardLabel")
            box.addWidget(value)
            box.addWidget(caption)
            self.card_values.append(value)
            cards.addWidget(card)
        right.addLayout(cards)

        self.main_splitter = QSplitter(Qt.Orientation.Vertical)
        image_area = QWidget()
        image_box = QVBoxLayout(image_area)
        image_box.setContentsMargins(0, 0, 0, 0)
        image_box.setSpacing(8)
        viewbar = QHBoxLayout()
        self.fit_button = QPushButton("Вписать")
        self.zoom_out = QPushButton("−")
        self.zoom_in = QPushButton("＋")
        self.zoom_out.setFixedWidth(38)
        self.zoom_in.setFixedWidth(38)
        self.mode = QComboBox()
        for name, mode in [("Перемещение", "navigate"), ("Выбор маски", "select"), ("Новая область · полигон", "polygon"), ("Разделить маску · линия", "split"), ("Изменить границу · полигон", "reshape")]:
            self.mode.addItem(name, mode)
        viewbar.addWidget(self.fit_button)
        viewbar.addWidget(self.zoom_out)
        viewbar.addWidget(self.zoom_in)
        viewbar.addSpacing(10)
        viewbar.addWidget(QLabel("Инструмент"))
        viewbar.addWidget(self.mode)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["Цвет: объекты", "Цвет: классы"])
        self.color_mode.setToolTip("Окраска итоговых масок и наложения по объектам или пользовательским классам")
        viewbar.addWidget(self.color_mode)
        viewbar.addStretch()
        self.coords = QLabel("X —   Y —")
        self.coords.setObjectName("Muted")
        viewbar.addWidget(self.coords)
        image_box.addLayout(viewbar)
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.tabBar().setExpanding(True)
        self.views = []
        for _, name in STAGES:
            view = ImageView()
            self.views.append(view)
            self.tabs.addTab(view, name)
            self.tabs.setTabToolTip(len(self.views) - 1, key if (key := STAGES[len(self.views) - 1][0]) != "Edge map" else "Edge map · величина градиента Sobel, диагностическая карта, не вероятность SAM")
        image_box.addWidget(self.tabs, 1)
        self.empty_label = QLabel("Добавьте снимок слева или перетащите его в список файлов.\nВсе шесть стадий появятся после обработки.")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setObjectName("Muted")
        self.empty_label.setWordWrap(True)
        image_box.addWidget(self.empty_label)
        self.stage_info = QLabel("B² PolyCrystal · подготовьте снимок для анализа")
        self.stage_info.setWordWrap(True)
        self.stage_info.setObjectName("Muted")
        image_box.addWidget(self.stage_info)
        self.legend = QLabel("● Зерно    ● Артефакт    ● Неопределено")
        self.legend.setText('<span style="color:#65232B">● Зерно</span> &nbsp; <span style="color:#AC7041">● Артефакт</span> &nbsp; <span style="color:#5C7775">● Неопределено</span>')
        self.legend.hide()
        image_box.addWidget(self.legend)
        self.edit_hint = QLabel("Правка доступна на вкладке 06. Ctrl + щелчок — несколько масок; Enter / двойной щелчок — завершить линию или полигон; Esc — отменить рисование.")
        self.edit_hint.setWordWrap(True)
        self.edit_hint.setObjectName("Muted")
        image_box.addWidget(self.edit_hint)
        self.main_splitter.addWidget(image_area)

        table_area = QWidget()
        table_box = QVBoxLayout(table_area)
        table_box.setContentsMargins(0, 0, 0, 0)
        editbar = QHBoxLayout()
        editbar.setSpacing(6)
        self.selection_label = QLabel("Маски не выбраны")
        self.selection_label.setMinimumWidth(95)
        self.selection_label.setMaximumWidth(170)
        editbar.addWidget(self.selection_label)
        self.class_combo = QComboBox()
        self.class_combo.addItems(CLASS_NAMES)
        self.assign_button = QPushButton("Назначить класс")
        self.merge_button = QPushButton("Объединить")
        self.delete_button = QPushButton("Удалить")
        self.undo_button = QPushButton("↶  Отменить")
        self.class_combo.setMinimumWidth(95)
        for widget in (self.class_combo, self.assign_button, self.merge_button, self.delete_button, self.undo_button):
            editbar.addWidget(widget)
        editbar.addStretch()
        table_box.addLayout(editbar)
        self.table = QTableWidget()
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        self.table.horizontalHeader().setResizeContentsPrecision(100)
        self.table.setMinimumHeight(115)
        table_box.addWidget(self.table, 1)
        bottom = QHBoxLayout()
        self.table_label = QLabel("Измерения появятся после сегментации")
        self.table_label.setObjectName("Muted")
        bottom.addWidget(self.table_label)
        bottom.addStretch()
        self.reference_button = QPushButton("Сравнить с эталонной маской")
        bottom.addWidget(self.reference_button)
        table_box.addLayout(bottom)
        self.main_splitter.addWidget(table_area)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setSizes([520, 240])
        right.addWidget(self.main_splitter, 1)
        layout.addLayout(right, 1)

    def _build_menu(self):
        file_menu = self.menuBar().addMenu("Файл")
        edit_menu = self.menuBar().addMenu("Правка")
        help_menu = self.menuBar().addMenu("Справка")
        for menu, text, shortcut, handler in [
            (file_menu, "Добавить изображения…", "Ctrl+O", self.choose_files),
            (file_menu, "Открыть сохранённую сессию…", "Ctrl+Shift+O", self.open_session),
            (file_menu, "Сохранить результат…", "Ctrl+S", self.save_current),
            (edit_menu, "Отменить правку", "Ctrl+Z", self.undo),
            (edit_menu, "Удалить выбранные маски", "Delete", self.delete_selected),
            (help_menu, "Как работать с программой", "F1", self.show_help),
        ]:
            action = QAction(text, self)
            action.setShortcut(QKeySequence(shortcut))
            action.triggered.connect(handler)
            menu.addAction(action)

    def _connect(self):
        self.add_button.clicked.connect(self.choose_files)
        self.files.filesDropped.connect(self.add_files)
        self.files.currentItemChanged.connect(self.file_changed)
        self.remove_button.clicked.connect(self.remove_files)
        self.open_session_button.clicked.connect(self.open_session)
        self.checkpoint_button.clicked.connect(self.choose_checkpoint)
        self.crop_suggest.clicked.connect(self.suggest_crop)
        self.crop.valueChanged.connect(self.store_crop)
        self.run_button.clicked.connect(lambda: self.run_processing(False))
        self.batch_button.clicked.connect(lambda: self.run_processing(True))
        self.cancel_button.clicked.connect(self.cancel_processing)
        self.save_button.clicked.connect(self.save_current)
        self.fit_button.clicked.connect(lambda: self.current_view().fit_image())
        self.zoom_out.clicked.connect(lambda: self.current_view().zoom(1 / 1.2))
        self.zoom_in.clicked.connect(lambda: self.current_view().zoom(1.2))
        self.mode.currentIndexChanged.connect(self.tool_changed)
        self.tabs.currentChanged.connect(self.tab_changed)
        self.views[-1].imageClicked.connect(self.select_pixel)
        self.views[-1].pathFinished.connect(self.finish_edit_path)
        for view in self.views:
            view.cursorPosition.connect(lambda x, y: self.coords.setText(f"X {x}   Y {y}"))
        self.delete_button.clicked.connect(self.delete_selected)
        self.merge_button.clicked.connect(self.merge_selected)
        self.assign_button.clicked.connect(self.assign_class)
        self.undo_button.clicked.connect(self.undo)
        self.table.itemSelectionChanged.connect(self.table_selected)
        self.reference_button.clicked.connect(self.compare_reference)
        self.scale_enabled.toggled.connect(self.update_scale_label)
        self.scale_px.valueChanged.connect(self.update_scale_label)
        self.scale_um.valueChanged.connect(self.update_scale_label)
        self.apply_scale_button.clicked.connect(self.apply_scale)
        self.filter_button.clicked.connect(self.apply_filters)
        self.color_mode.currentIndexChanged.connect(self.refresh_colors)
        self.update_scale_label()

    def current_view(self):
        return self.views[self.tabs.currentIndex()]

    def error(self, title, message):
        QMessageBox.warning(self, title, str(message))

    def choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Добавить SEM-изображения", str(ROOT / "data"), FILTER_IMAGES)
        self.add_files(paths)

    def add_files(self, paths):
        known = {self.files.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.files.count())}
        candidates = []
        for name in paths:
            path = Path(name).expanduser()
            if path.is_dir():
                candidates.extend(sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES))
            elif path.suffix.lower() in IMAGE_SUFFIXES:
                candidates.append(path)
        first = None
        for path in candidates:
            key = str(path.resolve())
            if key in known:
                continue
            item = QListWidgetItem(path.name)
            item.setData(Qt.ItemDataRole.UserRole, key)
            item.setToolTip(key)
            self.files.addItem(item)
            self.crop_by_path[key] = int(self.crop_manifest.get(path.name, 0))
            known.add(key)
            first = first or item
        if first:
            self.files.setCurrentItem(first)
        elif paths:
            self.statusBar().showMessage("Новых поддерживаемых изображений нет.")

    def remove_files(self):
        for item in self.files.selectedItems():
            self.files.takeItem(self.files.row(item))
        if not self.files.count():
            self.result = None
            self.set_busy(False)

    def selected_path(self):
        item = self.files.currentItem()
        return Path(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def file_changed(self, current, previous=None):
        if current is None:
            return
        path = Path(current.data(Qt.ItemDataRole.UserRole))
        if str(path) in self.results:
            self.show_result(self.results[str(path)])
            return
        try:
            rgb = load_image(path)
        except Exception as exc:
            self.error("Не удалось открыть изображение", exc)
            return
        self.result = None
        self.selected_ids.clear()
        self.history = []
        self.dirty = False
        h, w = rgb.shape[:2]
        self.image_title.setText(path.name)
        self.image_details.setText(f"{w} × {h} px · RGB-представление для обработки · результат ещё не рассчитан")
        self.views[0].set_image(rgb)
        for view in self.views[1:]:
            view.scene().clear()
            view.picture = None
            view.annotations.clear()
            view.points.clear()
        self.tabs.setCurrentIndex(0)
        self.empty_label.setText("Проверьте нижнюю служебную полосу, параметры SAM и масштаб. Затем запустите обработку.")
        self.empty_label.show()
        self.loading_config = True
        self.crop.setMaximum(max(0, h - 16))
        self.crop.setValue(self.crop_by_path.get(str(path), int(self.crop_manifest.get(path.name, 0))))
        self.set_scale_controls(self.scale_by_path.get(str(path)))
        self.loading_config = False
        if path.name not in self.crop_manifest:
            self.statusBar().showMessage("Новый снимок: проверьте служебную полосу и вручную задайте срез нижних строк. По умолчанию срез 0.")
        self.table.setRowCount(0)
        self.table.setColumnCount(0)
        for card in self.card_values:
            card.setText("—")
        self.table_label.setText("Измерения появятся после сегментации")
        self.set_busy(bool(self.worker and self.worker.isRunning()))
        self.update_edit_buttons()

    def choose_checkpoint(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите веса MobileSAM", str(ROOT / "models"), "Веса PyTorch (*.pt *.pth);;Все файлы (*)")
        if path:
            self.checkpoint.setText(path)

    def suggest_crop(self):
        path = self.selected_path()
        if path:
            try:
                self.crop.setValue(int(self.crop_manifest.get(path.name, 0)))
                self.statusBar().showMessage("Загружен срез из crop_manifest.json." if path.name in self.crop_manifest else "Для нового снимка срез не определён: проверьте нижнюю полосу и задайте число строк вручную.")
            except Exception as exc:
                self.error("Ошибка чтения", exc)

    def store_crop(self):
        path = self.selected_path()
        if path is not None and not self.loading_config:
            self.crop_by_path[str(path)] = self.crop.value()

    def config(self):
        return Config(
            crop_bottom=self.crop.value(), clahe_clip=self.clahe.value(), denoise=self.denoise.isChecked(),
            points_per_side=self.points.value(), pred_iou_thresh=self.iou.value(),
            stability_score_thresh=self.stability.value(), min_area=self.min_area.value(),
            box_nms_thresh=self.nms.value(),
            exclude_border=self.exclude_border.isChecked(),
            scale_um_per_px=self.scale_um.value() / self.scale_px.value() if self.scale_enabled.isChecked() else None,
            max_area_fraction=self.max_area.value() / 100,
            border_margin=self.border_margin.value(),
        )

    def run_processing(self, batch):
        if self.worker and self.worker.isRunning():
            return
        if batch:
            items = self.files.selectedItems() or [self.files.item(i) for i in range(self.files.count())]
        else:
            items = [self.files.currentItem()] if self.files.currentItem() else []
        paths = [Path(i.data(Qt.ItemDataRole.UserRole)) for i in items]
        if not paths:
            self.error("Нет изображения", "Добавьте хотя бы один SEM-снимок.")
            return
        unsaved = [path for path in paths if str(path) in self.dirty_results]
        if unsaved:
            choice = QMessageBox.question(self, "Повторная обработка", "У выбранных снимков есть несохранённые результаты или правки. Повторный расчёт заменит их. Продолжить?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                return
        checkpoint = Path(self.checkpoint.text()).expanduser()
        if not checkpoint.is_file():
            self.error("Не найдены веса MobileSAM", "Выберите файл весов .pt/.pth. Инструкция загрузки и проверка окружения приведены в README проекта.")
            return
        output = None
        if batch:
            folder = QFileDialog.getExistingDirectory(self, "Папка для пакетных результатов", str(ROOT / "results"))
            if not folder:
                return
            output = folder
        self.failed_jobs = []
        self.set_busy(True)
        self.progress.setRange(0, 0)
        self.statusBar().showMessage("Загрузка MobileSAM… Первый запуск может занять некоторое время.")
        crops = {str(path): self.crop_by_path.get(str(path), int(self.crop_manifest.get(path.name, 0))) for path in paths}
        scales = {str(path): self.scale_by_path.get(str(path)) for path in paths}
        self.worker = SegmentWorker(checkpoint, self.config(), paths, output, crops, scales)
        self.worker.progress.connect(self.processing_progress)
        self.worker.succeeded.connect(self.processing_succeeded)
        self.worker.failed.connect(self.processing_failed)
        self.worker.complete.connect(self.processing_complete)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()

    def set_busy(self, busy):
        self.busy = busy
        has_file = self.files.count() > 0
        for widget in [self.run_button, self.batch_button]:
            widget.setEnabled(not busy and has_file)
        for widget in [self.add_button, self.remove_button, self.open_session_button, self.checkpoint_button, self.checkpoint, self.crop, self.crop_suggest, self.clahe, self.denoise, self.points, self.iou, self.stability, self.nms, self.min_area, self.max_area, self.border_margin, self.exclude_border, self.scale_enabled, self.scale_px, self.scale_um, self.files]:
            widget.setEnabled(not busy)
        self.cancel_button.setVisible(busy)
        self.cancel_button.setEnabled(busy)
        self.save_button.setEnabled(not busy and self.result is not None)
        self.reference_button.setEnabled(not busy and self.result is not None)
        self.apply_scale_button.setEnabled(not busy and self.result is not None)
        self.filter_button.setEnabled(not busy and self.result is not None)
        self.mode.setEnabled(not busy and self.result is not None)
        self.update_scale_label()
        self.update_edit_buttons()

    def processing_progress(self, done, total, filename):
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        self.progress.setFormat(f"{done} / {total}")
        self.statusBar().showMessage(f"Обработка: {filename} · {done} / {total}. Окно остаётся доступным.")

    def processing_succeeded(self, path, result):
        self.dirty_results.discard(path)
        if self.worker is not None and not self.worker.output_root:
            self.dirty_results.add(path)
        self.histories.pop(path, None)
        self.crop_by_path[path] = result.config.crop_bottom
        self.scale_by_path[path] = result.config.scale_um_per_px
        self.results[path] = result
        for i in range(self.files.count()):
            item = self.files.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == path:
                item.setText("✓  " + Path(path).name)
                self.files.blockSignals(True)
                self.files.setCurrentItem(item)
                self.files.blockSignals(False)
        self.show_result(result)

    def processing_failed(self, name, reason):
        self.failed_jobs.append(f"{name}: {reason}")

    def processing_complete(self):
        if self.failed_jobs:
            self.error("Не все файлы обработаны", "\n\n".join(self.failed_jobs))
        stopped = bool(self.worker and self.worker.cancelled)
        message = "Обработка завершена." if not self.failed_jobs else f"Обработка завершена с ошибками: {len(self.failed_jobs)}."
        if stopped:
            message = "Очередь остановлена. Завершённые файлы сохранены." if self.worker.output_root else "Обработка завершена. Сохраните результат кнопкой «Сохранить результат»."
        elif self.result is not None and self.dirty:
            message += " Сохраните результат."
        self.statusBar().showMessage(message)

    def worker_finished(self):
        self.set_busy(False)
        if self.worker:
            self.worker.deleteLater()
            self.worker = None

    def cancel_processing(self):
        if self.worker:
            self.worker.requestInterruption()
            self.cancel_button.setEnabled(False)
            self.statusBar().showMessage("Остановка запрошена. Текущий снимок будет завершён; пакетный результат сохранится автоматически, одиночный сохраните кнопкой.")

    def load_config(self, config):
        self.loading_config = True
        self.crop.setValue(config.crop_bottom)
        self.clahe.setValue(config.clahe_clip)
        self.denoise.setChecked(config.denoise)
        self.points.setValue(config.points_per_side)
        self.iou.setValue(config.pred_iou_thresh)
        self.stability.setValue(config.stability_score_thresh)
        self.nms.setValue(config.box_nms_thresh)
        self.min_area.setValue(config.min_area)
        self.exclude_border.setChecked(config.exclude_border)
        self.max_area.setValue(config.max_area_fraction * 100)
        self.border_margin.setValue(config.border_margin)
        self.scale_enabled.setChecked(config.scale_um_per_px is not None)
        if config.scale_um_per_px is not None:
            self.scale_px.setValue(100)
            self.scale_um.setValue(config.scale_um_per_px * 100)
        self.update_scale_label()
        self.loading_config = False

    def show_result(self, result):
        self.result = result
        self.results[str(Path(result.source).resolve())] = result
        self.selected_ids.clear()
        key = str(Path(result.source).resolve())
        self.history = self.histories.setdefault(key, [])
        self.dirty = key in self.dirty_results
        self.loading_config = True
        self.crop.setMaximum(max(0, result.original.shape[0] - 16))
        self.load_config(result.config)
        self.loading_config = True
        self.crop.setValue(self.crop_by_path.get(key, result.config.crop_bottom))
        self.set_scale_controls(self.scale_by_path.get(key, result.config.scale_um_per_px))
        self.loading_config = False
        self.refresh_views(reset=True)
        self.tabs.setCurrentIndex(5)
        self.mode.setCurrentIndex(1)
        self.image_title.setText(Path(result.source).name)
        self.empty_label.hide()
        self.set_busy(bool(self.worker and self.worker.isRunning()))

    def refresh_views(self, reset=False):
        if self.result is None:
            return
        images = stage_images(self.result, self.color_mode.currentIndex() == 1)
        self.overlay_image = images["Overlay on original"]
        for view, (key, _) in zip(self.views, STAGES):
            if key in images:
                view.set_image(images[key], reset=reset)
        self.render_selection()
        self.fill_table()
        self.update_cards()
        h, w = self.result.labels.shape
        original_h = self.result.original.shape[0]
        self.image_details.setText(f"Рабочая область {w} × {h} px · срез снизу {original_h - h} px · MobileSAM · " + ("есть несохранённые правки" if self.dirty else "расчёт выполнен"))
        self.update_edit_buttons()
        self.update_stage_info()

    def fill_table(self):
        records = self.result.records
        keys = list(records[0]) if records else ["id", "class", "area_px", "perimeter_px", "feret_diameter_px"]
        for record in records[1:]:
            for key in record:
                if key not in keys:
                    keys.append(key)
        self.table_keys = keys
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.setColumnCount(len(keys))
        self.table.setHorizontalHeaderLabels([HEADERS.get(k, k) for k in keys])
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            ident = int(record.get("id", record.get("label", row + 1)))
            for col, key in enumerate(keys):
                value = record.get(key)
                item = NumericItem(fmt(value))
                item.setData(Qt.ItemDataRole.UserRole, ident)
                if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
                    item.setData(Qt.ItemDataRole.UserRole + 1, float(value))
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.table.setItem(row, col, item)
                if key in ("class", "class_name"):
                    item.setForeground(QColor(*CLASS_COLORS.get(str(value), (67, 39, 32))))
        self.table.resizeColumnsToContents()
        self.table.setSortingEnabled(True)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) in self.selected_ids:
                for col in range(self.table.columnCount()):
                    self.table.item(row, col).setSelected(True)
        self.table.setUpdatesEnabled(True)
        self.table.blockSignals(False)
        self.table_label.setText(f"{len(records)} объектов · все характеристики экспортируются в CSV · классы редактируются вручную")

    def update_cards(self):
        records = self.result.records
        grain_records = [r for r in records if r.get("class", r.get("class_name", "Зерно")) == "Зерно"]
        self.card_values[0].setText(f"{len(records)} / {len(grain_records)}")
        scale = self.result.config.scale_um_per_px
        def mean_of(candidates):
            for key in candidates:
                values = [float(r[key]) for r in grain_records if r.get(key) is not None]
                if values:
                    return float(np.mean(values))
            return None
        area = mean_of(["area_um2"] if scale is not None else ["area_px", "area_px2"])
        feret = mean_of(["feret_max_um", "feret_diameter_um", "feret_um"] if scale is not None else ["feret_max_px", "feret_diameter_px", "feret_px"])
        self.card_values[1].setText(f"{fmt(round(area, 2) if area is not None else None)} {'мкм²' if scale is not None else 'px²'}")
        self.card_values[2].setText(f"{fmt(round(feret, 2) if feret is not None else None)} {'мкм' if scale is not None else 'px'}")
        self.card_values[3].setText(f"{scale:.5g} мкм/px" if scale is not None else "Пиксели")

    def tab_changed(self, index):
        if index != 5:
            self.views[-1].cancel_path()
        self.update_stage_info()
        QTimer.singleShot(0, self.current_view().fit_image)

    def tool_changed(self):
        mode = self.mode.currentData()
        self.views[-1].set_mode(mode)
        if mode != "navigate" and self.result is not None:
            self.tabs.setCurrentIndex(5)
        hints = {
            "navigate": "Колесо — масштаб. Перетаскивание левой кнопкой — перемещение.",
            "select": "Щёлкните по маске. Ctrl + щелчок добавляет или снимает выбор.",
            "polygon": "Щёлкайте вершины новой области. Enter / двойной щелчок — закончить; Esc — отменить.",
            "split": "Выберите одну маску, затем проведите ломаную через неё. Enter — выполнить геометрическое разделение.",
            "reshape": "Выберите одну маску, затем обведите её новую границу полигоном. Enter — заменить границу выбранной маски.",
        }
        self.statusBar().showMessage(hints[mode])

    def select_pixel(self, x, y, additive):
        if self.result is None or y >= self.result.labels.shape[0] or x >= self.result.labels.shape[1]:
            return
        ident = int(self.result.labels[y, x])
        if not additive:
            self.selected_ids.clear()
        if ident:
            if additive and ident in self.selected_ids:
                self.selected_ids.remove(ident)
            else:
                self.selected_ids.add(ident)
        self.render_selection()
        self.table.blockSignals(True)
        self.table.setUpdatesEnabled(False)
        self.table.clearSelection()
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.ItemDataRole.UserRole) in self.selected_ids:
                for col in range(self.table.columnCount()):
                    self.table.item(row, col).setSelected(True)
        self.table.setUpdatesEnabled(True)
        self.table.blockSignals(False)
        self.update_edit_buttons()

    def table_selected(self):
        self.selected_ids = {int(item.data(Qt.ItemDataRole.UserRole)) for item in self.table.selectedItems() if item.data(Qt.ItemDataRole.UserRole) is not None}
        self.render_selection()
        self.update_edit_buttons()

    def render_selection(self):
        if self.result is None:
            return
        image = self.overlay_image.copy()
        if self.selected_ids:
            h, w = self.result.labels.shape
            mask = np.isin(self.result.labels, list(self.selected_ids))
            image[:h, :w][mask] = (0.25 * image[:h, :w][mask] + 0.75 * np.array([236, 170, 178])).astype(np.uint8)
        draw_label_boundaries(image, self.result.labels)
        self.views[-1].set_image(image, reset=False)

    def update_edit_buttons(self):
        busy = self.busy
        has_result = self.result is not None and not busy
        n = len(self.selected_ids)
        ids = ", ".join(map(str, sorted(self.selected_ids)))
        self.selection_label.setText(("Выбрано: " + ids if n <= 3 else f"Выбрано объектов: {n}") if n else "Маски не выбраны")
        self.selection_label.setToolTip(ids)
        self.delete_button.setEnabled(has_result and n > 0)
        self.assign_button.setEnabled(has_result and n > 0)
        self.merge_button.setEnabled(has_result and n > 1)
        self.class_combo.setEnabled(has_result)
        self.undo_button.setEnabled(has_result and bool(self.history))

    def snapshot(self):
        self.history.append({name: copy.deepcopy(getattr(self.result, name)) for name in
            ("labels", "raw_labels", "classes", "config", "scores", "edit_log", "proposal_ids", "original_border_ids", "summary")})
        if len(self.history) > 20:
            self.history.pop(0)

    def edited(self, description, geometry_ids=None):
        if geometry_ids:
            for ident in geometry_ids:
                self.result.scores[int(ident)] = None
                self.result.proposal_ids.pop(int(ident), None)
        self.result.edit_log.append({"time": datetime.now().astimezone().isoformat(), "operation": description})
        self.result.summary["reference_metrics"] = None
        refresh_result(self.result)
        self.dirty = True
        self.dirty_results.add(str(Path(self.result.source).resolve()))
        self.selected_ids &= set(map(int, np.unique(self.result.labels)))
        self.refresh_views()
        self.statusBar().showMessage(description + " Сохраните результат, чтобы сохранить исправления.")

    def delete_selected(self):
        if self.result is None or not self.selected_ids or (self.worker and self.worker.isRunning()):
            return
        self.snapshot()
        self.result.labels[np.isin(self.result.labels, list(self.selected_ids))] = 0
        for ident in self.selected_ids:
            self.result.classes.pop(ident, None)
            self.result.scores.pop(ident, None)
            self.result.proposal_ids.pop(ident, None)
        self.selected_ids.clear()
        self.edited("Выбранные маски удалены.")

    def merge_selected(self):
        if self.result is None or len(self.selected_ids) < 2:
            return
        self.snapshot()
        target = min(self.selected_ids)
        self.result.labels[np.isin(self.result.labels, list(self.selected_ids))] = target
        for ident in self.selected_ids - {target}:
            self.result.classes.pop(ident, None)
            self.result.scores.pop(ident, None)
            self.result.proposal_ids.pop(ident, None)
        self.selected_ids = {target}
        self.edited("Выбранные маски объединены по решению пользователя.", {target})

    def assign_class(self):
        if self.result is None or not self.selected_ids:
            return
        self.snapshot()
        for ident in self.selected_ids:
            self.result.classes[ident] = self.class_combo.currentText()
        self.edited(f"Назначен класс «{self.class_combo.currentText()}».")

    def finish_edit_path(self, points):
        if self.result is None or (self.worker and self.worker.isRunning()):
            return
        from skimage.draw import polygon, line
        from skimage.measure import label
        mode = self.mode.currentData()
        height, width = self.result.labels.shape
        if any(x < 0 or x >= width or y < 0 or y >= height for x, y in points):
            self.error("Область вне рабочей части", "Рисуйте только внутри рабочей области, выше удалённой служебной полосы.")
            return
        if mode in ("polygon", "reshape"):
            if mode == "reshape" and len(self.selected_ids) != 1:
                self.error("Выберите одну маску", "Сначала выберите одну маску инструментом «Выбор маски», затем обведите её новую границу.")
                return
            rr, cc = polygon([p[1] for p in points], [p[0] for p in points], (height, width))
            if not len(rr):
                return
            target = next(iter(self.selected_ids)) if mode == "reshape" else None
            overlap = np.count_nonzero((self.result.labels[rr, cc] != 0) & (self.result.labels[rr, cc] != target))
            affected = set(map(int, np.unique(self.result.labels[rr, cc]))) - {0}
            if overlap:
                choice = QMessageBox.question(self, "Полигон пересекает существующие маски", f"В полигоне уже размечено {overlap} пикселей. Заменить их новой областью?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if choice != QMessageBox.StandardButton.Yes:
                    return
            self.snapshot()
            ident = target if target is not None else self.new_mask_id()
            if target is not None:
                self.result.labels[self.result.labels == ident] = 0
            self.result.labels[rr, cc] = ident
            if target is None:
                self.result.classes[ident] = self.class_combo.currentText()
            self.selected_ids = {ident}
            self.edited("Граница маски заменена полигоном пользователя." if mode == "reshape" else "Новая область добавлена по полигону пользователя.", {ident} | affected)
        elif mode == "split":
            if len(self.selected_ids) != 1:
                self.error("Выберите одну маску", "Сначала выберите одну маску инструментом «Выбор маски», затем нарисуйте разделяющую линию.")
                return
            ident = next(iter(self.selected_ids))
            region = self.result.labels == ident
            divided = region.copy()
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                rr, cc = line(y1, x1, y2, x2)
                divided[rr, cc] = False
            components = label(divided, connectivity=1)
            count = int(components.max())
            if count < 2:
                self.error("Разделение не выполнено", "Линия не разделила маску. Проведите её от одного внешнего края маски до другого.")
                return
            self.snapshot()
            self.result.labels[region] = 0
            old_class = self.result.classes.pop(ident, "Зерно")
            self.result.scores.pop(ident, None)
            self.result.proposal_ids.pop(ident, None)
            next_id = self.new_mask_id()
            self.selected_ids.clear()
            for component in range(1, count + 1):
                new_id = next_id + component - 1
                self.result.labels[components == component] = new_id
                self.result.classes[new_id] = old_class
                self.selected_ids.add(new_id)
            self.edited(f"Маска вручную разделена на {count} областей. Пиксели линии оставлены фоном.", self.selected_ids)

    def undo(self):
        if self.result is None or not self.history or (self.worker and self.worker.isRunning()):
            return
        state = self.history.pop()
        for name, value in state.items():
            setattr(self.result, name, value)
        config = self.result.config
        self.load_config(config)
        key = str(Path(self.result.source).resolve())
        self.crop_by_path[key] = config.crop_bottom
        self.scale_by_path[key] = config.scale_um_per_px
        self.selected_ids.clear()
        self.edited("Последняя правка отменена.")

    def new_mask_id(self):
        return max(int(self.result.labels.max()), max(self.result.scores, default=0), max(self.result.proposal_ids, default=0)) + 1

    def update_scale_label(self):
        enabled = self.scale_enabled.isChecked()
        self.scale_px.setEnabled(enabled and not self.busy)
        self.scale_um.setEnabled(enabled and not self.busy)
        if enabled:
            self.scale_label.setText(f"1 px = {self.scale_um.value() / self.scale_px.value():.7g} мкм\nВведите длину линейки со своего снимка.")
        else:
            self.scale_label.setText("Без масштаба: измерения в пикселях")
        path = self.selected_path()
        if path is not None and not self.loading_config:
            self.scale_by_path[str(path)] = self.scale_um.value() / self.scale_px.value() if enabled else None

    def set_scale_controls(self, scale):
        previous = self.loading_config
        self.loading_config = True
        self.scale_enabled.setChecked(scale is not None)
        self.scale_px.setValue(100)
        self.scale_um.setValue(scale * 100 if scale is not None else 1)
        self.update_scale_label()
        self.loading_config = previous

    def apply_scale(self):
        if self.result is None:
            return
        self.snapshot()
        scale = self.scale_um.value() / self.scale_px.value() if self.scale_enabled.isChecked() else None
        self.result.config = replace(self.result.config, scale_um_per_px=scale)
        self.edited("Масштаб измерений обновлён.")

    def save_current(self):
        if self.result is None:
            self.error("Нет результата", "Сначала выполните сегментацию или откройте сохранённую сессию.")
            return
        if self.worker and self.worker.isRunning():
            return
        root = QFileDialog.getExistingDirectory(self, "Родительская папка для результата", str(ROOT / "results"))
        if not root:
            return
        folder = unique_folder(Path(root), Path(self.result.source).stem)
        try:
            save_result(self.result, folder)
            self.dirty = False
            self.dirty_results.discard(str(Path(self.result.source).resolve()))
            self.refresh_views()
            self.statusBar().showMessage(f"Сохранено: {folder}")
            QMessageBox.information(self, "Результат сохранён", f"Папка: {folder}\n\nСохранены стадии, индексная маска, измерения CSV, параметры и пользовательские исправления. Папку можно открыть как сессию.")
        except Exception as exc:
            self.error("Не удалось сохранить", exc)

    def open_session(self, folder=None):
        if isinstance(folder, bool):
            folder = None
        if self.worker and self.worker.isRunning():
            return
        folder = folder or QFileDialog.getExistingDirectory(self, "Открыть папку сохранённого результата", str(ROOT / "results"))
        if not folder:
            return
        try:
            result = load_result(Path(folder))
            legacy_state = None
            if result.summary.get("pipeline_version", 1) < 2 and result.proposal_archive and not result.edit_log:
                legacy_state = {name: copy.deepcopy(getattr(result, name)) for name in
                    ("labels", "raw_labels", "classes", "config", "scores", "edit_log", "proposal_ids", "original_border_ids", "summary")}
                refilter_result(result, result.config)
            key = str(Path(result.source).resolve())
            if key in self.dirty_results:
                choice = QMessageBox.question(self, "Открытие другой сессии", "У этого снимка есть несохранённые исправления. Заменить их открываемой сессией?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
                if choice != QMessageBox.StandardButton.Yes:
                    return
            self.dirty_results.discard(key)
            self.histories.pop(key, None)
            if legacy_state is not None:
                self.histories[key] = [legacy_state]
                self.dirty_results.add(key)
            self.crop_by_path[key] = result.config.crop_bottom
            self.scale_by_path[key] = result.config.scale_um_per_px
            self.results[key] = result
            existing = next((self.files.item(i) for i in range(self.files.count()) if self.files.item(i).data(Qt.ItemDataRole.UserRole) == key), None)
            if existing is None:
                existing = QListWidgetItem("✓  " + Path(result.source).name)
                existing.setData(Qt.ItemDataRole.UserRole, key)
                existing.setToolTip(key)
                self.files.addItem(existing)
            self.files.setCurrentItem(existing)
            self.show_result(result)
            if legacy_state is not None:
                self.statusBar().showMessage("Старая сессия: отбор масок обновлён. Проверьте результат и сохраните; Ctrl+Z вернёт прежние маски.")
            else:
                self.statusBar().showMessage(f"Открыта сессия: {folder}")
        except Exception as exc:
            self.error("Не удалось открыть сессию", exc)

    def compare_reference(self):
        if self.result is None:
            return
        path, _ = QFileDialog.getOpenFileName(self, "Эталон: 2D индексная маска (0 — фон, ID — объекты)", str(ROOT), "Индексные маски (*.npy *.tif *.tiff *.png);;Все файлы (*)")
        if not path:
            return
        try:
            if Path(path).suffix.lower() == ".npy":
                reference = np.load(path, allow_pickle=False)
            else:
                import imageio.v3 as iio
                reference = iio.imread(path)
            if reference.ndim != 2 or not np.issubdtype(reference.dtype, np.integer) or np.any(reference < 0):
                raise ValueError("Нужна двумерная целочисленная индексная маска: фон 0, каждый объект с отдельным положительным ID.")
            if reference.shape == self.result.original.shape[:2]:
                reference = reference[:self.result.labels.shape[0], :self.result.labels.shape[1]]
            if reference.shape != self.result.labels.shape:
                raise ValueError(f"Размер эталона {reference.shape} не совпадает с рабочей областью {self.result.labels.shape}.")
            grain_ids = [r["id"] for r in self.result.records if r["class"] == "Зерно"]
            grain_labels = np.where(np.isin(self.result.labels, grain_ids), self.result.labels, 0)
            metrics = evaluate_labels(grain_labels, reference)
            self.result.summary["reference_metrics"] = metrics
            self.result.summary["evaluation_target"] = "Зерно; Артефакт и Неопределено считаются фоном"
            self.result.summary["reference_source"] = str(Path(path).resolve())
            self.dirty = True
            self.dirty_results.add(str(Path(self.result.source).resolve()))
            self.refresh_views()
            dialog = QDialog(self)
            dialog.setWindowTitle("Метрики относительно эталона")
            dialog.resize(490, 470)
            box = QVBoxLayout(dialog)
            label = QLabel(f"Эталон зёрен: {Path(path).name}\nОценивается класс «Зерно»; остальные классы — фон.\nСопоставление объектов: IoU ≥ 0,5.\nМетрики показывают согласие с этой разметкой.")
            label.setWordWrap(True)
            box.addWidget(label)
            table = QTableWidget(0, 2)
            table.setHorizontalHeaderLabels(["Метрика", "Значение"])
            names = {"iou": "IoU (пиксели)", "dice": "Dice", "precision": "Precision (пиксели)", "recall": "Recall (пиксели)", "tp": "TP объектов", "fp": "FP объектов", "fn": "FN объектов", "f1": "F1 объектов", "object_f1": "F1 объектов", "object_tp": "TP объектов", "object_fp": "FP объектов", "object_fn": "FN объектов", "object_mean_matched_iou": "Средний IoU найденных пар", "object_iou_threshold": "Порог IoU объектов"}
            def add_rows(values, prefix=""):
                for key, value in values.items():
                    if key == "note":
                        continue
                    if isinstance(value, dict):
                        add_rows(value, prefix + key + " / ")
                    else:
                        row = table.rowCount()
                        table.insertRow(row)
                        table.setItem(row, 0, QTableWidgetItem(prefix + names.get(key, key)))
                        table.setItem(row, 1, QTableWidgetItem(fmt(value)))
            add_rows(metrics)
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
            table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
            box.addWidget(table)
            buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
            buttons.rejected.connect(dialog.reject)
            box.addWidget(buttons)
            dialog.exec()
        except Exception as exc:
            self.error("Сравнение не выполнено", exc)

    def refresh_colors(self):
        if self.result is not None:
            images = stage_images(self.result, self.color_mode.currentIndex() == 1)
            self.overlay_image = images["Overlay on original"]
            self.views[4].set_image(images["Postprocessed segmentation"], reset=False)
            self.render_selection()
            self.update_stage_info()
        self.legend.setVisible(self.color_mode.currentIndex() == 1 and self.tabs.currentIndex() >= 4)

    def update_stage_info(self):
        if self.result is None:
            return
        summary = self.result.summary
        index = self.tabs.currentIndex()
        notes = [
            "Исходный снимок · нижняя служебная полоса удаляется только в рабочей области.",
            "Серая рабочая область · медианный фильтр по настройке · локальный контраст CLAHE.",
            "Градиент Собеля · контраст отображения усилен; исходные значения градиента сохранены.",
            f"Маски SAM до отбора по краю и минимальной площади · масок поля отклонено: {summary.get('excluded_field_masks', 0)}. Все исходные предложения сохранены в сессии.",
            f"Отбор · край: {summary.get('excluded_border', 0)} · малые: {summary.get('excluded_small', 0)} · маски поля: {summary.get('excluded_field_masks', 0)} · осталось: {len(self.result.records)}.",
            "Результат на исходнике · выберите объект или строку таблицы для правки границы и класса.",
        ]
        self.stage_info.setText(notes[index])
        self.edit_hint.setVisible(index == 5)
        self.legend.setVisible(self.color_mode.currentIndex() == 1 and index >= 4)

    def apply_filters(self):
        if self.result is None or self.busy:
            return
        geometry_changed = any(score is None for score in self.result.scores.values())
        if geometry_changed or self.result.edit_log:
            choice = QMessageBox.question(self, "Повторный отбор исходных масок",
                "Отбор восстанавливает маски SAM и заменяет ручную геометрию и удаления. Классы сохранятся у неизменённых масок. Действие можно отменить. Применить?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                return
        config = replace(self.result.config, min_area=self.min_area.value(),
                         max_area_fraction=self.max_area.value()/100,
                         exclude_border=self.exclude_border.isChecked(), border_margin=self.border_margin.value())
        snapshot_added = False
        try:
            # Validate before changing the undo stack or result.
            config.validate(self.result.original.shape)
            self.snapshot()
            snapshot_added = True
            refilter_result(self.result, config)
        except Exception as exc:
            if snapshot_added:
                state = self.history.pop()
                for name, value in state.items():
                    setattr(self.result, name, value)
            self.error("Фильтры не применены", exc)
            return
        self.selected_ids.clear()
        self.edited("Фильтры применены к исходным маскам SAM без запуска модели.")

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and not self.busy:
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        if not self.busy:
            self.add_files([url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()])
            event.acceptProposedAction()

    def show_help(self):
        QMessageBox.information(self, "Работа с B² PolyCrystal", "1. Добавьте изображения; список принимает перетаскивание файлов и папок.\n\n2. Проверьте число удаляемых нижних строк. Для предоставленного набора берётся проверенный срез из crop_manifest.json, для новых файлов по умолчанию 0. Перед обработкой задайте срез вручную для каждого нового снимка.\n\n3. Модель работает на CPU. Веса и точность отбора настраиваются в раскрывающемся блоке MobileSAM. Для пакетного режима выделите несколько файлов или снимите выделение, чтобы обработать все. У каждого файла сохраняется собственное число удаляемых строк.\n\n4. Укажите масштаб по линейке своего снимка либо оставьте пиксели.\n\n5. Изучите все шесть стадий. Карта Sobel диагностическая; это не вероятность SAM.\n\n6. Во вкладке 06 выберите маски, удалите артефакты, назначьте классы, объедините области. Полигон добавляет область; ломаная разделяет одну выбранную маску. Enter завершает рисование, Esc отменяет; Ctrl+Z отменяет до 20 правок.\n\n7. Сохраните результат: стадии, CSV, индексная маска, настройки и классы. Сохранённая папка открывается как сессия. Сравнение с эталоном принимает 2D маску с ID объектов; без эталона качество численно не заявляется.")

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.error("Выполняется обработка", "Дождитесь завершения текущего снимка. Для остановки очереди нажмите «Остановить после текущего файла».")
            event.ignore()
            return
        if self.dirty_results:
            choice = QMessageBox.question(self, "Несохранённые исправления", "Закрыть окно и потерять несохранённые исправления?", QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        event.accept()


def launch(result_folder=None, screenshot_path=None) -> int:
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("B² PolyCrystal")
    app.setOrganizationName("B² PolyCrystal")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    window = MainWindow()
    if result_folder is not None:
        window.open_session(result_folder)
    window.show()
    if screenshot_path is not None:
        target = Path(screenshot_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        def capture():
            window.grab().save(str(target))
            app.quit()
        QTimer.singleShot(1100, capture)
    return app.exec()
