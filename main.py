import os
import random
import sys

import cv2
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QHeaderView, QLabel, QMainWindow, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from utils import METHODS, MODEL_DIR, load_models, predict_all

BG, PANEL, TEXT, GREY = "#0f1117", "#181b24", "#e6e8ee", "#8b92a5"
BLUE, GREEN, RED = "#6c8cff", "#34d3a0", "#ff6b81"

STYLE = f"""
* {{ color: {TEXT}; font-size: 13px; }}
QMainWindow, QWidget#root {{ background: {BG}; }}
QFrame#panel {{ background: {PANEL}; border-radius: 14px; }}
QFrame#card {{ background: {BG}; border-radius: 10px; }}
QLabel#title {{ font-size: 22px; font-weight: bold; }}
QLabel#grey {{ color: {GREY}; }}
QComboBox {{ background: {BG}; border: 1px solid #2a2f3d; border-radius: 8px; padding: 7px; }}
QComboBox QAbstractItemView {{ background: {PANEL}; selection-background-color: #2c3245; }}
QPushButton {{ background: #232837; border: none; border-radius: 8px; padding: 9px 16px; }}
QPushButton:hover {{ background: #2c3245; }}
QPushButton:disabled {{ background: #1c2030; color: #555c70; }}
QTabWidget::pane {{ border: none; }}
QTabBar::tab {{ background: transparent; color: {GREY}; padding: 9px 18px; border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {TEXT}; border-bottom: 2px solid {BLUE}; }}
QTableWidget {{ background: transparent; border: none; gridline-color: #232837; }}
QHeaderView::section {{ background: transparent; color: {GREY}; border: none; border-bottom: 1px solid #232837; padding: 8px; }}
"""


def make_label(text, name=None):
    label = QLabel(text)
    if name:
        label.setObjectName(name)
    return label


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CIFAR-10: classical descriptors vs Deep Learning")
        self.resize(1050, 760)
        self.info = None
        self.models = {}
        self.matrices = {}
        self.samples = None

        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(18)
        layout.addWidget(self.build_panel(), 1)
        self.load_saved_models()

    def build_panel(self):
        panel = QFrame()
        panel.setObjectName("panel")
        box = QVBoxLayout(panel)
        box.setContentsMargins(20, 14, 20, 20)

        head = QHBoxLayout()
        head.addWidget(make_label("CIFAR-10: HOG, LBP, ORB, CNN", "title"))
        head.addStretch()
        self.status = make_label("", "grey")
        head.addWidget(self.status)
        box.addLayout(head)

        tabs = QTabWidget()
        box.addWidget(tabs)

        # tab 1: test one image on the 4 models
        page = QWidget()
        v = QVBoxLayout(page)
        top = QHBoxLayout()
        pick = QPushButton("Choose an image...")
        pick.clicked.connect(self.choose_image)
        rand = QPushButton("Random test image")
        rand.clicked.connect(self.random_image)
        top.addWidget(pick)
        top.addWidget(rand)
        top.addStretch()
        v.addLayout(top)

        row = QHBoxLayout()
        left = QVBoxLayout()
        self.preview = QLabel("No image")
        self.preview.setFixedSize(300, 300)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setStyleSheet(f"background: {BG}; border-radius: 12px;")
        self.truth = QLabel("")
        left.addWidget(self.preview)
        left.addWidget(self.truth)
        left.addStretch()
        row.addLayout(left)
        row.addSpacing(20)

        grid = QGridLayout()
        grid.setSpacing(12)
        self.card_value = {}
        for i, method in enumerate(METHODS):
            card = QFrame()
            card.setObjectName("card")
            cv = QVBoxLayout(card)
            cv.addWidget(make_label(method, "grey"))
            value = QLabel("-")
            value.setStyleSheet("font-size: 18px; font-weight: bold;")
            cv.addWidget(value)
            self.card_value[method] = value
            grid.addWidget(card, i // 2, i % 2)
        row.addLayout(grid, 1)
        v.addLayout(row)
        v.addStretch()
        tabs.addTab(page, "Test an image")

        # tab 2: metrics
        page = QWidget()
        v = QVBoxLayout(page)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["Method", "Dim", "Accuracy", "F1 macro", "Extraction (s)", "Training (s)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setFixedHeight(200)
        self.bar_fig = Figure(facecolor=PANEL)
        self.bar_canvas = Canvas(self.bar_fig)
        v.addWidget(self.table)
        v.addWidget(self.bar_canvas, 1)
        tabs.addTab(page, "Results")

        # tab 3: confusion matrix
        page = QWidget()
        v = QVBoxLayout(page)
        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))
        self.cm_choice = QComboBox()
        self.cm_choice.setFixedWidth(200)
        self.cm_choice.currentIndexChanged.connect(self.draw_matrix)
        top.addWidget(self.cm_choice)
        top.addStretch()
        self.cm_fig = Figure(facecolor=PANEL)
        self.cm_canvas = Canvas(self.cm_fig)
        v.addLayout(top)
        v.addWidget(self.cm_canvas, 1)
        tabs.addTab(page, "Confusion matrix")
        return panel

    # ---------- loading saved models
    def load_saved_models(self):
        loaded = load_models()
        if loaded is None:
            self.status.setText("No trained models found. Run: uv run python train.py")
            return
        self.info, self.models, self.matrices = loaded
        self.status.setText(f"Models loaded ({len(self.models)} methods).")

        sample_file = os.path.join(MODEL_DIR, "samples.npz")
        if os.path.exists(sample_file):
            self.samples = np.load(sample_file)

        rows = self.info["results"]
        self.table.setRowCount(len(rows))
        for i, r in enumerate(rows):
            values = [r["method"], r["dim"], f"{r['accuracy']:.3f}", f"{r['f1']:.3f}",
                      f"{r['t_feat']:.1f}", f"{r['t_fit']:.1f}"]
            for j, v in enumerate(values):
                self.table.setItem(i, j, QTableWidgetItem(str(v)))
        self.draw_bars(rows)

        self.cm_choice.blockSignals(True)
        self.cm_choice.clear()
        self.cm_choice.addItems(list(self.matrices))
        self.cm_choice.blockSignals(False)
        self.draw_matrix()

    # ---------- plots
    def style_axes(self, ax):
        ax.set_facecolor(PANEL)
        for side in ax.spines.values():
            side.set_color("#2a2f3d")
        ax.tick_params(colors=GREY)
        ax.title.set_color(TEXT)

    def draw_bars(self, rows):
        self.bar_fig.clear()
        ax = self.bar_fig.add_subplot(111)
        self.style_axes(ax)
        x = np.arange(len(rows))
        ax.bar(x - 0.18, [r["accuracy"] for r in rows], 0.36, color=BLUE, label="Accuracy")
        ax.bar(x + 0.18, [r["f1"] for r in rows], 0.36, color=GREEN, label="F1 macro")
        ax.set_xticks(x)
        ax.set_xticklabels([r["method"] for r in rows])
        ax.set_ylim(0, 1)
        ax.set_title("Test set performance")
        ax.legend(facecolor=PANEL, edgecolor="#2a2f3d", labelcolor=TEXT)
        self.bar_fig.tight_layout()
        self.bar_canvas.draw()

    def draw_matrix(self):
        method = self.cm_choice.currentText()
        if method not in self.matrices:
            return
        classes = self.info["classes"]
        cm = self.matrices[method].astype(float)
        cm = cm / (cm.sum(axis=1, keepdims=True) + 1e-9)  # each row sums to 1

        self.cm_fig.clear()
        ax = self.cm_fig.add_subplot(111)
        self.style_axes(ax)
        img = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(method)
        ax.set_xlabel("Predicted", color=GREY)
        ax.set_ylabel("True", color=GREY)
        ax.set_xticks(range(len(classes)))
        ax.set_yticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=90, fontsize=8)
        ax.set_yticklabels(classes, fontsize=8)
        self.cm_fig.colorbar(img, ax=ax).ax.tick_params(colors=GREY)
        self.cm_fig.tight_layout()
        self.cm_canvas.draw()

    # ---------- testing an image
    def choose_image(self):
        if not self.models:
            QMessageBox.information(self, "Info", "No trained models found. Run train.py first.")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Image", "", "Images (*.jpg *.jpeg *.png *.bmp)")
        if path:
            img = cv2.imread(path)
            if img is not None:
                self.show_prediction(img)

    def random_image(self):
        if not self.models or self.samples is None:
            QMessageBox.information(self, "Info", "No trained models found. Run train.py first.")
            return
        i = random.randrange(len(self.samples["X"]))
        true = self.info["classes"][int(self.samples["y"][i])]
        self.show_prediction(self.samples["X"][i], true)

    def show_prediction(self, img, true=None):
        rgb = cv2.cvtColor(cv2.resize(img, (300, 300)), cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        qimg = QImage(rgb.data, 300, 300, 900, QImage.Format.Format_RGB888)
        self.preview.setPixmap(QPixmap.fromImage(qimg.copy()))
        self.truth.setText(f"True class: {true}" if true else "")

        preds = predict_all(self.models, self.info["classes"], img)
        for method in METHODS:
            label = self.card_value[method]
            if method not in preds:
                label.setText("not trained")
                label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {GREY};")
                continue
            if true is None:
                color = BLUE
            else:
                color = GREEN if preds[method] == true else RED
            label.setText(preds[method])
            label.setStyleSheet(f"font-size: 18px; font-weight: bold; color: {color};")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    window = Window()
    window.show()
    sys.exit(app.exec())
