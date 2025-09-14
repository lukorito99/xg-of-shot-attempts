import copy

import tensorflow as tf
import json
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import List, Tuple, Optional, Dict, Set

import cv2
import numpy as np
import vlc
from PyQt5.QtCore import (
    Qt,
    QThread,
    pyqtSignal,
    QSettings,
    QTimer,
    QEvent,
    QPropertyAnimation,
    QParallelAnimationGroup,
    QRect,
    QPoint,
    QRectF,
    QAbstractAnimation,
    pyqtProperty,
    QMetaObject,
    Q_ARG,
    pyqtSlot, QThreadPool, QWaitCondition, QMutex, QMutexLocker, QPointF, QSize
)
from PyQt5.QtGui import (
    QFont,
    QIcon,
    QKeySequence,
    QColor,
    QPen,
    QPainter,
    QBrush,
    QImage,
    QPixmap, QCursor, QRadialGradient, QPainterPath, QPolygonF
)
from PyQt5.QtWidgets import (
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QDialog,
    QMessageBox,
    QApplication,
    QSlider,
    QFrame,
    QSizePolicy,
    QScrollArea,
    QFileDialog,
    QProgressBar,
    QRadioButton,
    QButtonGroup,
    QGridLayout,
    QToolButton,
    QMenu,
    QStyle,
    QToolTip, QDialogButtonBox, QTextEdit, QShortcut, QProgressDialog, QGroupBox
)

from action_recognition3 import inference_on_video
from xg_module2 import XGModel

class WindowState(Enum):
    NORMAL = 1
    MAXIMIZED = 2
    FULLSCREEN = 3

# ---------------------------
# PreviewPopup: A popup preview dialog
# ---------------------------
class PreviewPopup(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Use frameless and always-on-top flags so the popup is visible
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(10, 10, 10, 10)
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.preview_label)
        self.resize(800, 450)
        # self.hide()
        self.setVisible(False)

    def show_preview(self, pixmap: QPixmap):
        if not pixmap or pixmap.isNull():
            self.hide()
            return
        # Scale the pixmap to fill the popup while keeping aspect ratio.
        scaled = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled)
        # Center the popup over the main window
        if self.parent():
            parent_rect = self.parent().frameGeometry()
            new_x = parent_rect.center().x() - self.width() // 2
            new_y = parent_rect.center().y() - self.height() // 2
            self.move(new_x, new_y)
        self.setVisible(True)
        self.show()
        self.raise_()
        self.activateWindow()

    def hide_preview(self):
        self.setVisible(False)
        # self.hide()

class FineTuneDialog(QDialog):
    def __init__(self, parent, segment, which_boundary: str, step=0.1):
        """
        parent: main window (xG)
        segment: the VideoSegment to be fine-tuned
        which_boundary: "start" or "end"
        step: the adjustment increment in seconds
        """
        super().__init__(parent)
        self.setWindowTitle(f"Fine Tune {which_boundary.capitalize()} Boundary")
        if parent and hasattr(parent, 'windowIcon'):
            self.setWindowIcon(parent.windowIcon())
        self.setModal(True)
        self.setStyleSheet("""
            QDialog {
                background-color: #FFFACD;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border: 2px solid #000000;
                border-radius: 4px;
                padding: 5px 15px;
                margin: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)
        self.segment = segment
        self.which_boundary = which_boundary  # "start" or "end"
        self.step = step

        # Saving the original value to revert if the user cancels.
        if self.which_boundary == "start":
            self.original_value = segment.start
        else:
            self.original_value = segment.end

        layout = QVBoxLayout(self)

        # current boundary value
        self.value_label = QLabel(self)
        self.value_label = QLabel(self)
        self.value_label.setStyleSheet("background-color: rgba(255,255,255,0.8); padding: 2px; border: 1px solid #000;")
        self.value_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.value_label, alignment=Qt.AlignCenter)
        self.update_value_label()

        # adjustment buttons
        btn_layout = QHBoxLayout()
        self.btn_minus = QPushButton(f"-{self.step:.5f}s", self)
        self.btn_plus = QPushButton(f"+{self.step:.5f}s", self)
        btn_layout.addWidget(self.btn_minus)
        btn_layout.addWidget(self.btn_plus)
        layout.addLayout(btn_layout)

        # adjustment functions
        self.btn_minus.clicked.connect(lambda: self.adjust_boundary(-self.step))
        self.btn_plus.clicked.connect(lambda: self.adjust_boundary(self.step))

        # OK/Cancel button box
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        layout.addWidget(self.button_box)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

    def update_value_label(self):
        if self.which_boundary == "start":
            current_val = self.segment.start
        else:
            current_val = self.segment.end
        self.value_label.setText(f"Current {self.which_boundary.capitalize()}: {current_val:.2f}s")

    def adjust_boundary(self, delta: float):
        if self.which_boundary == "start":
            new_value = self.segment.start + delta
            new_value = max(0, min(new_value, self.segment.end - 0.1))
            self.segment.start = new_value
            if self.segment.start_marker:
                self.segment.start_marker.updatePosition()
        else:
            new_value = self.segment.end + delta
            new_value = min(self.parent().timeline.duration, max(new_value, self.segment.start + 0.1))
            self.segment.end = new_value
            if self.segment.end_marker:
                self.segment.end_marker.updatePosition()
        self.update_value_label()
        self.parent().timeline.clear_thumbnail_cache()
        current_time = self.segment.start if self.which_boundary == "start" else self.segment.end
        # Show the frame in the preview popup
        frame = self.parent().extract_frame(self.parent().fileName, current_time)
        if frame is not None:
            pixmap = self.parent().bgr_to_pixmap(frame)
            self.parent().preview_popup.show_preview(pixmap)
        else:
            self.parent().preview_popup.hide_preview()

    def reject(self):
        # Revert boundary to original value if the user cancels
        if self.which_boundary == "start":
            self.segment.start = self.original_value
            if self.segment.start_marker:
                self.segment.start_marker.updatePosition()
        else:
            self.segment.end = self.original_value
            if self.segment.end_marker:
                self.segment.end_marker.updatePosition()
        # Update preview to reflect the original value.
        if self.parent() and hasattr(self.parent(), "video_player"):
            orig_time = self.original_value
            self.parent().video_player.set_time(int(orig_time * 1000))
            self.parent().timeline.request_thumbnail(orig_time)
        if self.parent() and hasattr(self.parent(), "preview_popup"):
            self.parent().preview_popup.hide_preview()
        super().reject()

class PointAnnotationDialog(QDialog):
    def __init__(self, frame_pixmap, boundary_label, timestamp, parent=None, existing_annotation=None):
        super().__init__(parent)
        self.setWindowIcon(QIcon('black.png'))
        self.setFont(QFont('Segoe Script', 12))
        if frame_pixmap is None or frame_pixmap.isNull():
            raise ValueError("PointAnnotationDialog received an invalid pixmap.")

        self.setWindowTitle("Expected Goals(xG)")
        self.boundary_label = boundary_label  # e.g., 'Shot Start' or 'Shot End'
        self.timestamp = timestamp
        self.annotation = None  # This will store the point as {"x":..., "y":...}
        self.existing_annotation = existing_annotation

        self.resize(800, 600)
        self.setModal(True)
        self.setStyleSheet("""
            QDialog { background-color: #2D2D2D; color: #FFFFFF; }
            QLabel { background-color: #1A1A1A; color: #FFFFFF; border: 1px solid #444444; }
            QDialogButtonBox { padding: 10px; }
            QPushButton { background-color: #444444; color: white; border: 1px solid #666666; 
                       border-radius: 4px; padding: 5px 15px; margin: 8px; font-size: 14px; }
            QPushButton:hover { background-color: #666666; }
            QToolButton { background-color: #444444; color: white; border: 1px solid #666666; 
                         border-radius: 4px; padding: 5px; margin: 2px; }
            QToolButton:hover { background-color: #666666; }
        """)

        main_layout = QVBoxLayout(self)

        instructions = QLabel("Click on the image to select the ball position.", self)
        instructions.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(instructions)

        # Toolbar for zoom controls
        toolbar = QHBoxLayout()
        zoom_in_btn = QToolButton(self)
        zoom_in_btn.setText("+")
        zoom_in_btn.setToolTip("Zoom In")
        zoom_in_btn.clicked.connect(self.zoom_in)
        zoom_out_btn = QToolButton(self)
        zoom_out_btn.setText("-")
        zoom_out_btn.setToolTip("Zoom Out")
        zoom_out_btn.clicked.connect(self.zoom_out)
        reset_zoom_btn = QToolButton(self)
        reset_zoom_btn.setText("1:1")
        reset_zoom_btn.setToolTip("Reset Zoom")
        reset_zoom_btn.clicked.connect(self.reset_zoom)
        toolbar.addWidget(zoom_in_btn)
        toolbar.addWidget(zoom_out_btn)
        toolbar.addWidget(reset_zoom_btn)
        toolbar.addStretch()
        main_layout.addLayout(toolbar)

        # Status bar to show mouse position
        self.status_bar = QLabel("Ready to annotate", self)
        self.status_bar.setStyleSheet("background-color: #333333; border: none; padding: 2px;")
        main_layout.addWidget(self.status_bar)

        # scroll area for the image
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.scroll_area)

        self.image_label = PointAnnotationLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.original_pixmap = frame_pixmap
        self.zoom_level = 1.0

        if not frame_pixmap.isNull():
            self.image_label.set_original_size(frame_pixmap.width(), frame_pixmap.height())
        else:
            self.image_label.set_original_size(1, 1)

        self.update_zoom()
        if self.existing_annotation:
            self.image_label.point = self.existing_annotation

        self.scroll_area.setWidget(self.image_label)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.image_label.installEventFilter(self)

    def update_zoom(self):
        if self.original_pixmap.isNull():
            return
        scaled_pixmap = self.original_pixmap.scaled(
            max(1, int(self.original_pixmap.width() * self.zoom_level)),
            max(1, int(self.original_pixmap.height() * self.zoom_level)),
            Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.image_label.adjustSize()
        self.status_bar.setText(f"Zoom: {int(self.zoom_level * 100)}%")

    def zoom_in(self):
        self.zoom_level = min(2.0, self.zoom_level * 1.2)
        self.update_zoom()

    def zoom_out(self):
        self.zoom_level = max(0.2, self.zoom_level / 1.2)
        self.update_zoom()

    def reset_zoom(self):
        self.zoom_level = 1.0
        self.update_zoom()

    def eventFilter(self, obj, event):
        if obj == self.image_label and event.type() == QEvent.MouseMove:
            pos = event.pos()
            if not self.original_pixmap.isNull():
                scale_x = self.original_pixmap.width() / self.image_label.width()
                scale_y = self.original_pixmap.height() / self.image_label.height()
                img_x = pos.x() * scale_x
                img_y = pos.y() * scale_y
                self.status_bar.setText(f"Position: ({int(img_x)}, {int(img_y)})")
        return super().eventFilter(obj, event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Delete:
            self.image_label.point = None
            self.image_label.update()
        else:
            super().keyPressEvent(event)

    def accept(self):
        self.annotation = self.image_label.point
        super().accept()

class PointAnnotationLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.point = None  # the annotation as a dict: {"x": float, "y": float}
        # Keep track of original image dimensions for scaling.
        self.original_width = 1
        self.original_height = 1
        # Visual properties for the point marker.
        self.marker_radius = 5
        self.marker_color = QColor(255, 0, 0)

    def set_original_size(self, width, height):
        self.original_width = max(1, width)
        self.original_height = max(1, height)

    def paintEvent(self, event):
        super().paintEvent(event)
        # Only draw the point if it exists
        if self.point is not None:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)
            pen = QPen(self.marker_color, 2)
            painter.setPen(pen)
            scale_x = self.width() / self.original_width
            scale_y = self.height() / self.original_height
            x = int(self.point["x"] * scale_x)
            y = int(self.point["y"] * scale_y)
            painter.drawEllipse(QPoint(x, y), self.marker_radius, self.marker_radius)
            font = QFont()
            font.setPointSize(8)
            painter.setFont(font)
            text = f"x:{self.point['x']:.1f}, y:{self.point['y']:.1f}"
            painter.drawText(x + self.marker_radius + 2, y, text)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # When clicked, record the point, converting widget coords to original image coords.
            scale_x = self.original_width / self.width()
            scale_y = self.original_height / self.height()
            original_x = event.pos().x() * scale_x
            original_y = event.pos().y() * scale_y
            self.point = {"x": float(original_x), "y": float(original_y)}
            self.update()

class FrameSelectorDialog(QDialog):
    def __init__(self, frames, parent=None, segment=None):
        super().__init__(parent)
        self.frames = frames
        self.current_index = 0
        self.selected_frame = None
        self.segment = segment

        # Calculate frame timestamps if segment is provided
        if segment:
            self.frame_timestamps = np.linspace(segment.start, segment.end, len(frames))
        else:
            self.frame_timestamps = None

        self.setWindowTitle("Select Frame")
        self.setup_ui()
        self.update_frame()

    def setup_ui(self):
        layout = QVBoxLayout()

        # Image display
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(800, 450)  # 16:9 aspect ratio
        self.image_label.setStyleSheet("QLabel { background-color: black; }")
        layout.addWidget(self.image_label)

        # Frame info
        info_layout = QHBoxLayout()
        self.frame_counter = QLabel()
        info_layout.addWidget(self.frame_counter)

        if self.frame_timestamps is not None:
            self.timestamp_label = QLabel()
            info_layout.addWidget(self.timestamp_label)

        info_layout.addStretch()
        layout.addLayout(info_layout)

        # Navigation buttons
        nav_layout = QHBoxLayout()

        self.prev_button = QPushButton("Previous Frame")
        self.prev_button.clicked.connect(self.prev_frame)
        nav_layout.addWidget(self.prev_button)

        self.next_button = QPushButton("Next Frame")
        self.next_button.clicked.connect(self.next_frame)
        nav_layout.addWidget(self.next_button)

        nav_layout.addStretch()

        # Action buttons
        self.select_button = QPushButton("Select This Frame")
        self.select_button.clicked.connect(self.accept_frame)
        nav_layout.addWidget(self.select_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        nav_layout.addWidget(self.cancel_button)

        layout.addLayout(nav_layout)
        self.setLayout(layout)

    def update_frame(self):
        frame = self.frames[self.current_index]
        # Convert BGR to RGB before displaying
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width = rgb_frame.shape[:2]
        bytes_per_line = 3 * width
        q_img = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Scale pixmap to fit label while maintaining aspect ratio
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # Update frame counter
        self.frame_counter.setText(f"Frame {self.current_index + 1} of {len(self.frames)}")

        # Update timestamp if available
        if self.frame_timestamps is not None:
            timestamp = self.frame_timestamps[self.current_index]
            self.timestamp_label.setText(f"Time: {timestamp:.2f}s")

        # Update button states
        self.prev_button.setEnabled(self.current_index > 0)
        self.next_button.setEnabled(self.current_index < len(self.frames) - 1)

    def prev_frame(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_frame()

    def next_frame(self):
        if self.current_index < len(self.frames) - 1:
            self.current_index += 1
            self.update_frame()

    def accept_frame(self):
        self.selected_frame = self.frames[self.current_index]
        self.accept()

@dataclass
class SoccerPitchConfiguration:
    width: float = 80.0
    length: float = 120.0
    penalty_box_length: float = 16.5
    goal_box_length: float = 5.5
    penalty_box_width: float = 40.32
    goal_box_width: float = 18.32
    penalty_spot_distance: float = 11.0
    centre_circle_radius: float = 9.15

    # Top‑level list of landmark names in the exact order of IDs 1…32:
    LANDMARK_NAMES = [
        "Top left corner",  # 1
        "Left 18yb boundary top",  # 2
        "Left 6yd boundary top",  # 3
        "Left 6yd boundary bottom",  # 4
        "Left 18yb boundary bottom",  # 5
        "Left penalty spot",  # 6
        "Left 18yd box top",  # 7
        "Left 6yd box top",  # 8
        "Left 6yd box bottom",  # 9
        "Left arc upper",  # 10
        "Left arc lower",  # 11
        "Left 18yd box bottom",  # 12
        "Bottom left corner",  # 13

        "Top center line",  # 14
        "Center circle left",  # 15
        "Center circle top",  # 16
        "Center circle bottom",  # 17
        "Center circle right",  # 18
        "Bottom center line",  # 19

        "Top right corner",  # 20
        "Right 18yd box top",  # 21
        "Right 6yd box top",  # 22
        "Right 6yd box bottom",  # 23
        "Right arc upper",  # 24
        "Right arc lower",  # 25
        "Right 18yb boundary top",  # 26
        "Right 6yd boundary top",  # 27
        "Right 6yd boundary bottom",  # 28
        "Right 18yb boundary bottom",  # 29
        "Right penalty spot",  # 30
        "Right 18yd box bottom",  # 31
        "Bottom right corner",  # 32
    ]
    all_landmarks: Dict[str, Tuple[int, float, float]] = field(default_factory=dict)

    @property
    def vertices(self) -> List[Tuple[float, float]]:
        return [(self.all_landmarks[name][1], self.all_landmarks[name][2])
                for name in self.LANDMARK_NAMES]


class SchematicPitchGridWidget(QWidget):
    landmarkDragged = pyqtSignal(int, QPoint)
    landmarkDropped = pyqtSignal(int, QPoint)

    def __init__(self, landmarks, edges, pitch_config=None, parent=None, margin=20):
        super().__init__(parent)
        self.landmarks = dict(landmarks)
        next_id = max(self.landmarks.keys(), default=0) + 1
        self.landmarks[next_id] = ("Centre Pitch", (60.0, 40.0))
        self.edges = edges
        self.margin = margin
        self.selected = set()
        self.hovered = None
        self.radius = 10  # Slightly smaller radius for cleaner look
        self.pitch_config = pitch_config or SoccerPitchConfiguration()
        self.setMouseTracking(True)
        self.drag_start_lid = None
        self.drag_start_pos = None

        # Set a fixed size for the grid widget based on the pitch dimensions
        ppm = 7  # pixels per meter
        fixed_width = int(self.pitch_config.length * ppm + 2 * margin)
        fixed_height = int(self.pitch_config.width * ppm + 2 * margin)
        self.setFixedSize(fixed_width, fixed_height)  # Use fixed size instead of minimum

        # Change size policy to prevent automatic expansion
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

    def sizeHint(self):
        # Return our fixed size
        ppm = 7  # preferred pixels per meter
        width = int(self.pitch_config.length * ppm + 2 * self.margin)
        height = int(self.pitch_config.width * ppm + 2 * self.margin)
        return QSize(width, height)

    def setPreferredSize(self, size):
        self._preferred_size = size

    def get_pitch_rect(self):
        w, h = self.width(), self.height()
        m = self.margin
        pitch_ratio = self.pitch_config.length / self.pitch_config.width
        avail_w = w - 2 * m
        avail_h = h - 2 * m

        if avail_w / avail_h > pitch_ratio:
            # Height is limiting
            ph = avail_h
            pw = ph * pitch_ratio
        else:
            # Width is limiting
            pw = avail_w
            ph = pw / pitch_ratio

        x0 = m + (avail_w - pw) / 2
        y0 = m + (avail_h - ph) / 2
        return QRectF(x0, y0, pw, ph)

    def mx(self, x_m):
        rect = self.get_pitch_rect()
        return rect.x() + (x_m / self.pitch_config.length) * rect.width()

    def my(self, y_m):
        rect = self.get_pitch_rect()
        return rect.y() + (y_m / self.pitch_config.width) * rect.height()

    def get_landmark_at(self, pos: QPoint) -> Optional[int]:
        """
        Return the landmark ID whose on‐screen circle contains pos,
        or None if none.
        """
        for lid, (_, (x_m, y_m)) in self.landmarks.items():
            px, py = self.mx(x_m), self.my(y_m)
            # distance squared ≤ radius²?
            dx, dy = pos.x() - px, pos.y() - py
            if dx * dx + dy * dy <= self.radius * self.radius:
                return lid
        return None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.get_pitch_rect()

        # Draw pitch background
        painter.fillRect(rect, QColor(0, 100, 0))

        # Draw pitch outline
        painter.setPen(QPen(Qt.white, 2))
        painter.drawRect(rect)

        # Draw center line
        center_x = self.mx(self.pitch_config.length / 2)
        painter.drawLine(
            QPointF(center_x, rect.y()),
            QPointF(center_x, rect.y() + rect.height())
        )

        # Draw all edges (lines)
        for id1, id2 in self.edges:
            if id1 in self.landmarks and id2 in self.landmarks:
                _, (x1, y1) = self.landmarks[id1]
                _, (x2, y2) = self.landmarks[id2]
                painter.setPen(QPen(Qt.white, 1.5))
                painter.drawLine(
                    QPointF(self.mx(x1), self.my(y1)),
                    QPointF(self.mx(x2), self.my(y2))
                )

        # Draw centre circle
        centre_x, centre_y = self.pitch_config.length / 2, self.pitch_config.width / 2
        arc_radius = self.pitch_config.centre_circle_radius

        # Calculate QRectF for the circle to account for potential non-square scaling
        width_in_pixels = arc_radius * 2 * rect.width() / self.pitch_config.length
        height_in_pixels = arc_radius * 2 * rect.height() / self.pitch_config.width

        circle_rect = QRectF(
            self.mx(centre_x - arc_radius),
            self.my(centre_y - arc_radius),
            width_in_pixels,
            height_in_pixels
        )

        painter.setPen(QPen(Qt.white, 2))
        painter.drawEllipse(circle_rect)

        # Draw penalty arcs for both sides
        # Left penalty area
        left_penalty_area = QRectF(
            rect.x(),
            self.my(centre_y - self.pitch_config.penalty_box_width / 2),
            self.pitch_config.penalty_box_length * rect.width() / self.pitch_config.length,
            self.pitch_config.penalty_box_width * rect.height() / self.pitch_config.width
        )
        painter.drawRect(left_penalty_area)

        # Right penalty area
        right_penalty_area = QRectF(
            rect.right() - self.pitch_config.penalty_box_length * rect.width() / self.pitch_config.length,
            self.my(centre_y - self.pitch_config.penalty_box_width / 2),
            self.pitch_config.penalty_box_length * rect.width() / self.pitch_config.length,
            self.pitch_config.penalty_box_width * rect.height() / self.pitch_config.width
        )
        painter.drawRect(right_penalty_area)

        # Draw points
        for lid, (name, (x, y)) in self.landmarks.items():
            px, py = self.mx(x), self.my(y)
            if lid in self.selected:
                painter.setBrush(QBrush(Qt.green))
            elif lid == self.hovered:
                painter.setBrush(QBrush(Qt.yellow))
            else:
                painter.setBrush(QBrush(Qt.white))

            painter.setPen(QPen(Qt.blue if lid in self.selected else Qt.black, 2))
            painter.drawEllipse(QPointF(px, py), self.radius, self.radius)

    def mousePressEvent(self, event):
        lid = self.get_landmark_at(event.pos())
        if lid is not None:
            self._dragging_id = lid
            self._orig_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        lid = self.get_landmark_at(event.pos())
        if lid != self.hovered:
            self.hovered = lid
            self.update()
            if lid is not None:
                name, _ = self.landmarks[lid]
                QToolTip.showText(event.globalPos(), name, self)
            else:
                QToolTip.hideText()

        if hasattr(self, "_dragging_id"):
            # fire a signal with the landmark and current global pos
            self.landmarkDragged.emit(self._dragging_id, event.globalPos())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if hasattr(self, "_dragging_id"):
            self.unsetCursor()
            self.landmarkDropped.emit(self._dragging_id, event.globalPos())
            del self._dragging_id
        else:
            super().mouseReleaseEvent(event)

class DraggableFrameLabel(QLabel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(False)  # we'll handle drops via signals now
        self.mapped_points = {}  # lid -> (x_img, y_img)
        self.undo_stack = []  # tuples of (lid, previous_value)
        self.redo_stack = []
        self.grid_widget = None
        self._dragging_landmark = None
        self._drag_start_global = None
        self._adjusting_landmark = None  # For fine-tuning existing points
        self.setMouseTracking(True)  # Enable mouse tracking for hover effects

    def set_grid_widget(self, widget):
        self.grid_widget = widget

    def get_landmark_at(self, pos: QPoint) -> Optional[int]:
        """
        Return the landmark ID whose on-screen circle contains pos,
        or None if none.
        """
        if not self.pixmap():
            return None

        # Convert to image coordinates
        lbl_w, lbl_h = self.width(), self.height()
        img_w, img_h = self.pixmap().width(), self.pixmap().height()
        scale_x = lbl_w / img_w
        scale_y = lbl_h / img_h

        for lid, (x_img, y_img) in self.mapped_points.items():
            # Convert image coordinates to label coordinates
            x_lbl = x_img * scale_x
            y_lbl = y_img * scale_y

            # Check if within radius
            dx, dy = pos.x() - x_lbl, pos.y() - y_lbl
            if dx * dx + dy * dy <= 10 * 10:  # 10px radius for selection
                return lid

        return None

    @pyqtSlot(int, QPoint)
    def startDrag(self, lid, global_pos):
        """
        Called when the user mouses‐down on a grid point and starts dragging.
        We just record which landmark and where we grabbed it.
        """
        self._dragging_landmark = lid
        self._drag_start_global = global_pos
        # Optionally change cursor:
        self.setCursor(Qt.ClosedHandCursor)

    @pyqtSlot(int, QPoint)
    def endDrag(self, lid, global_pos):
        """
        Called when the user releases the mouse after dragging a landmark.
        We convert the global pos into image coords, record undo/redo,
        and update both frame and grid.
        """
        # only handle if it matches the one we started
        if self._dragging_landmark != lid:
            return

        # map into our widget coordinates
        local_pt = self.mapFromGlobal(global_pos)
        x_lbl = max(0, min(local_pt.x(), self.width() - 1))
        y_lbl = max(0, min(local_pt.y(), self.height() - 1))

        # if there's a pixmap, convert label coords → pixmap coords
        if self.pixmap() is not None:
            lbl_w, lbl_h = self.width(), self.height()
            img_w, img_h = self.pixmap().width(), self.pixmap().height()
            # assume scaled to fit QLabel (keep aspect)?
            # adjust if you use scaledContents or a different scaling policy
            scale_x = img_w / lbl_w
            scale_y = img_h / lbl_h
            x_img = x_lbl * scale_x
            y_img = y_lbl * scale_y
        else:
            # no pixmap: just use label coords
            x_img, y_img = x_lbl, y_lbl

        # record undo/redo
        prev = self.mapped_points.get(lid, None)
        self.undo_stack.append((lid, prev))
        self.redo_stack.clear()

        # store new mapping
        self.mapped_points[lid] = (x_img, y_img)

        # tell the grid to mark this one selected/highlighted
        if self.grid_widget:
            self.grid_widget.selected.add(lid)
            self.grid_widget.update()

        # cleanup
        self._dragging_landmark = None
        self.unsetCursor()
        self.update()  # repaint to draw the dashed circle

    def mousePressEvent(self, event):
        """Handle mouse press for adjusting existing points."""
        if event.button() == Qt.LeftButton:
            lid = self.get_landmark_at(event.pos())
            if lid is not None:
                # Start adjusting an existing point
                self._adjusting_landmark = lid
                self.setCursor(Qt.ClosedHandCursor)
                # Record position for undo
                prev = self.mapped_points.get(lid, None)
                self.undo_stack.append((lid, prev))
                self.redo_stack.clear()

                if self.grid_widget:
                    self.grid_widget.selected.add(lid)
                    self.grid_widget.update()

                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse movement for adjusting points or hovering effects."""
        # Handle adjusting (dragging) existing points
        if self._adjusting_landmark is not None:
            lid = self._adjusting_landmark

            # Get current position in label coordinates
            x_lbl = max(0, min(event.x(), self.width() - 1))
            y_lbl = max(0, min(event.y(), self.height() - 1))

            # Convert to image coordinates
            if self.pixmap() is not None:
                lbl_w, lbl_h = self.width(), self.height()
                img_w, img_h = self.pixmap().width(), self.pixmap().height()
                scale_x = img_w / lbl_w
                scale_y = img_h / lbl_h
                x_img = x_lbl * scale_x
                y_img = y_lbl * scale_y
            else:
                x_img, y_img = x_lbl, y_lbl

            # Update the point position
            self.mapped_points[lid] = (x_img, y_img)
            self.update()
            event.accept()
            return

        # Handle hover effects for existing points
        lid = self.get_landmark_at(event.pos())
        if lid is not None:
            self.setCursor(Qt.OpenHandCursor)
            # Show tooltip with landmark name
            if self.grid_widget and lid in self.grid_widget.landmarks:
                name = self.grid_widget.landmarks[lid][0]
                QToolTip.showText(event.globalPos(), name, self)
        else:
            self.setCursor(Qt.ArrowCursor)
            QToolTip.hideText()

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release for fine adjustments."""
        if event.button() == Qt.LeftButton and self._adjusting_landmark is not None:
            self._adjusting_landmark = None
            self.unsetCursor()
            self.update()
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        """
        Draw the frame (via QLabel) then overplot any mapped points
        with a dashed outline if they're newly dropped.
        """
        super().paintEvent(event)

        if not self.pixmap():
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        for lid, (x_img, y_img) in self.mapped_points.items():
            # convert back into label coords
            lbl_w, lbl_h = self.width(), self.height()
            img_w, img_h = self.pixmap().width(), self.pixmap().height()
            scale_x = lbl_w / img_w
            scale_y = lbl_h / img_h
            x_lbl = x_img * scale_x
            y_lbl = y_img * scale_y

            # Make the point semi-transparent if currently being adjusted
            if lid == self._adjusting_landmark:
                # Draw a transparent filled circle
                painter.setBrush(QBrush(QColor(255, 0, 0, 120)))  # Semi-transparent red
                painter.setPen(QPen(QColor(255, 0, 0, 200), 2))  # More opaque outline
                r = 10
                painter.drawEllipse(QPointF(x_lbl, y_lbl), r, r)

                # Draw crosshairs for precise positioning
                painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
                painter.drawLine(QPointF(x_lbl - 12, y_lbl), QPointF(x_lbl + 12, y_lbl))
                painter.drawLine(QPointF(x_lbl, y_lbl - 12), QPointF(x_lbl, y_lbl + 12))
            else:
                # Standard appearance for normal points
                is_selected = self.grid_widget and lid in self.grid_widget.selected
                is_hovered = self.get_landmark_at(self.mapFromGlobal(QCursor.pos())) == lid

                if is_hovered:
                    # Highlight on hover
                    painter.setBrush(QBrush(QColor(255, 255, 0, 120)))  # Yellow semi-transparent
                    painter.setPen(QPen(QColor(255, 255, 0, 200), 2))
                elif is_selected:
                    # Selected points
                    painter.setBrush(QBrush(QColor(0, 255, 0, 120)))  # Green semi-transparent
                    painter.setPen(QPen(QColor(0, 255, 0, 200), 2))
                else:
                    # Normal points
                    painter.setBrush(QBrush(QColor(255, 255, 255, 120)))  # White semi-transparent
                    painter.setPen(QPen(QColor(0, 0, 0, 180), 2, Qt.DashLine))

                r = 8
                painter.drawEllipse(QPointF(x_lbl, y_lbl), r, r)

                # Draw a small ID number by the point for better identification
                if self.grid_widget and lid in self.grid_widget.landmarks:
                    name = self.grid_widget.landmarks[lid][0]
                    font = painter.font()
                    font.setBold(True)
                    painter.setFont(font)
                    painter.setPen(QPen(QColor(0, 0, 0, 200), 1))

                    # Draw a small background for better readability
                    text_rect = QRectF(x_lbl + 10, y_lbl - 6, 25, 15)
                    painter.fillRect(text_rect, QBrush(QColor(255, 255, 255, 150)))

                    # Draw the landmark ID number
                    painter.drawText(text_rect, Qt.AlignCenter, str(lid))

    def undo(self):
        if not self.undo_stack:
            return
        lid, prev = self.undo_stack.pop()
        if lid in self.mapped_points:
            self.redo_stack.append((lid, self.mapped_points[lid]))
            if prev is None:
                del self.mapped_points[lid]
            else:
                self.mapped_points[lid] = prev
            if self.grid_widget:
                self.grid_widget.selected.discard(lid)
                self.grid_widget.update()
            self.update()

    def redo(self):
        if not self.redo_stack:
            return
        lid, pos = self.redo_stack.pop()
        self.undo_stack.append((lid, self.mapped_points.get(lid, None)))
        self.mapped_points[lid] = pos
        if self.grid_widget:
            self.grid_widget.selected.add(lid)
            self.grid_widget.update()
        self.update()


class DraggableFullPitch(QWidget):
    markerChanged = pyqtSignal()

    def __init__(self, marker_color=Qt.red, parent=None):
        super().__init__(parent)
        self.color = marker_color
        self.half = 'left'  # 'left' or 'right'
        self.marker_pos = None  # normalized (0–1,0–1) within selected half
        self.undo_stack = []
        self.redo_stack = []
        self.dragging = False
        # new marker interaction settings (tweakable)
        self.snap_enabled = True
        self.snap_threshold_px = 12.0  # pixel distance to snap
        self.nudge_step_norm = 0.005  # keyboard nudge step in normalized coords (~0.5% of half width)
        self._near_candidate = None  # (norm_x, norm_y) if within snap range (for visual hint)
        self._drag_candidate_highlight = None

        self.setMouseTracking(True)

        # Pitch dimensions in meters (120m x 80m)
        self.pitch_length = 120.0
        self.pitch_width = 80.0

    def set_half(self, half):
        if half not in ('left', 'right'): return
        self.half = half
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.marker_pos = None
        self.markerChanged.emit()
        self.update()

    def _pitch_rect(self):
        """Get the rectangle for the displayed half pitch - fills entire available area"""
        margin = 20
        return self.rect().adjusted(margin, margin, -margin, -margin)

    def _half_rect(self):
        """Get the rectangle for the selected half (same as pitch rect now)"""
        return self._pitch_rect()

    def _to_norm(self, pos):
        """Convert widget position to normalized coordinates within selected half"""
        hr = self._half_rect()
        if not hr.contains(pos):
            return None

        x = (pos.x() - hr.left()) / hr.width()
        y = (pos.y() - hr.top()) / hr.height()
        return (max(0.0, min(x, 1.0)), max(0.0, min(y, 1.0)))

    def _to_widget(self, x, y):
        """Convert normalized coordinates to widget position"""
        hr = self._half_rect()
        return QPointF(hr.left() + x * hr.width(), hr.top() + y * hr.height())

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        hr = self._half_rect()
        if not hr.contains(e.pos()):
            return

        # compute distance from click to marker widget position
        clicked_pt = e.pos()

        # if existing marker close to click, begin dragging
        if self.marker_pos:
            cur_widget_pt = self._to_widget(*self.marker_pos)
            dist = (cur_widget_pt - QPointF(clicked_pt)).manhattanLength()
            if dist < 16:  # enlarged hit area (pixels)
                self.dragging = True
                # refresh candidate highlight while dragging
                self._update_snap_candidate(clicked_pt)
                return

        # otherwise, create a new marker (push undo)
        self.undo_stack.append(self.marker_pos)
        self.redo_stack.clear()
        norm_pos = self._to_norm(e.pos())
        if norm_pos:
            # If snap is enabled, consider snapping on press if close enough
            if self.snap_enabled:
                snapped = self._maybe_snap(norm_pos)
                if snapped:
                    self.marker_pos = snapped
                else:
                    self.marker_pos = norm_pos
            else:
                self.marker_pos = norm_pos
            self.markerChanged.emit()
            self.update()

    def mouseMoveEvent(self, e):
        if self.dragging:
            norm_pos = self._to_norm(e.pos())
            if norm_pos:
                # while dragging, live-snap only for visual hint; commit on release
                if self.snap_enabled:
                    snap = self._maybe_snap(norm_pos, only_hint=True)
                    if snap:
                        # show highlight but do not commit until release
                        self._near_candidate = snap
                    else:
                        self._near_candidate = None
                    # update marker to follow cursor for fluid control (but keep _near_candidate for commit)
                    self.marker_pos = norm_pos
                else:
                    self.marker_pos = norm_pos
                self.markerChanged.emit()
                self.update()
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e):
        if self.dragging:
            self.dragging = False
            # on release, if there was a near candidate and snapping is enabled, snap to it
            if self.snap_enabled and self._near_candidate:
                self.undo_stack.append(self.marker_pos)
                self.marker_pos = self._near_candidate
                self._near_candidate = None
            # clear transient highlight
            self._drag_candidate_highlight = None
            self.markerChanged.emit()
            self.update()
        super().mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        # arrow keys nudge the marker in small normalized steps
        if self.marker_pos is None:
            return super().keyPressEvent(e)

        nx, ny = self.marker_pos
        changed = False
        if e.key() == Qt.Key_Left:
            nx = max(0.0, nx - self.nudge_step_norm)
            changed = True
        elif e.key() == Qt.Key_Right:
            nx = min(1.0, nx + self.nudge_step_norm)
            changed = True
        elif e.key() == Qt.Key_Up:
            ny = max(0.0, ny - self.nudge_step_norm)
            changed = True
        elif e.key() == Qt.Key_Down:
            ny = min(1.0, ny + self.nudge_step_norm)
            changed = True

        if changed:
            self.undo_stack.append(self.marker_pos)
            self.marker_pos = (nx, ny)
            # if snapping is enabled, allow immediate snap if close
            if self.snap_enabled:
                maybe = self._maybe_snap(self.marker_pos)
                if maybe:
                    self.marker_pos = maybe
            self.markerChanged.emit()
            self.update()
        else:
            super().keyPressEvent(e)

    def undo(self):
        if not self.undo_stack: return
        prev = self.undo_stack.pop()
        self.redo_stack.append(self.marker_pos)
        self.marker_pos = prev
        self.markerChanged.emit()
        self.update()

    def redo(self):
        if not self.redo_stack: return
        nxt = self.redo_stack.pop()
        self.undo_stack.append(self.marker_pos)
        self.marker_pos = nxt
        self.markerChanged.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(40, 100, 40))  # Darker grass background

        # Get pitch rectangle (now represents only the selected half)
        pr = self._pitch_rect()

        # Fill pitch with grass color
        painter.fillRect(pr, QColor(34, 139, 34))

        # Draw half pitch markings
        self.draw_half_pitch(painter, pr)

        # Draw crosshair marker if exists
        if self.marker_pos:
            self.draw_crosshair_marker(painter)

        painter.end()

    def draw_coords_overlay(self, painter, half_outer: QRectF, side: str):
        """
        Draw a small translucent coords box just *inside* the selected half,
        immediately next to the halfway border (center line).
        side: 'left' or 'right' - which half is active.
        """
        # nothing to draw if no marker
        if not self.marker_pos:
            return

        # Prepare text: normalized + absolute coordinates
        nx, ny = self.marker_pos
        abs_coords = self.get_absolute_pitch_coordinates()
        if not abs_coords:
            return
        ax, ay = abs_coords

        # Two-line label
        text_lines = [f"{nx:.3f}, {ny:.3f}", f"{ax:.1f} m, {ay:.1f} m"]

        f = painter.font()
        f.setPointSize(max(10, f.pointSize()))  # keep at least small readable size
        painter.setFont(f)
        fm = painter.fontMetrics()

        # compute text block size
        text_width = max(fm.horizontalAdvance(line) for line in text_lines)
        text_height = fm.height() * len(text_lines)

        pad_x = 10
        pad_y = 6
        box_w = text_width + pad_x * 2
        box_h = text_height + pad_y * 2

        # center vertically around the pitch centerline midpoint
        center_y = half_outer.top() + half_outer.height() / 2.0
        box_y = center_y - box_h / 2.0

        margin_from_border = 6  # pixels inside the half
        if side == 'left':
            box_x = half_outer.right() - box_w - margin_from_border
        else:  # 'right'
            box_x = half_outer.left() + margin_from_border

        # Keep the box fully inside half_outer vertically (clamp)
        if box_y < half_outer.top() + 4:
            box_y = half_outer.top() + 4
        if box_y + box_h > half_outer.bottom() - 4:
            box_y = half_outer.bottom() - 4 - box_h

        box_rect = QRectF(box_x, box_y, box_w, box_h)

        # Draw background rounded rect + border + text
        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 170))  # translucent dark background
        painter.drawRoundedRect(box_rect, 6, 6)

        painter.setPen(QPen(QColor(220, 220, 220, 220)))
        # draw text lines
        tx = box_rect.left() + pad_x
        ty = box_rect.top() + pad_y + fm.ascent()
        for line in text_lines:
            painter.drawText(QPointF(tx, ty), line)
            ty += fm.height()

        painter.restore()

    def draw_coords_overlay(self, painter, half_outer: QRectF, side: str):
        """
        Draw a small translucent coords box positioned so it does not cover the marker.
        If the preferred position would overlap the marker, try corner positions or nudge
        the box away from the marker.
        """
        # only draw when there's a marker
        if not self.marker_pos:
            return

        # Prepare text lines and metrics
        nx, ny = self.marker_pos
        abs_coords = self.get_absolute_pitch_coordinates()
        if not abs_coords:
            return
        ax, ay = abs_coords
        text_lines = [f"{nx:.3f}, {ny:.3f}", f"{ax:.1f} m, {ay:.1f} m"]

        # Setup font smaller to reduce obstruction
        f = painter.font()
        f.setPointSize(max(8, f.pointSize() - 1))
        painter.setFont(f)
        fm = painter.fontMetrics()
        text_width = max(fm.horizontalAdvance(line) for line in text_lines)
        text_height = fm.height() * len(text_lines)

        pad_x = 6
        pad_y = 5
        box_w = text_width + pad_x * 2
        box_h = text_height + pad_y * 2

        widget_rect = QRectF(self.rect())
        # marker bounding rect (small circle around marker)
        marker_pt = self._to_widget(*self.marker_pos)
        marker_radius = 10.0
        marker_rect = QRectF(marker_pt.x() - marker_radius, marker_pt.y() - marker_radius,
                             marker_radius * 2, marker_radius * 2)

        # Candidate positions (try outside the half first near the centerline as before)
        margin_from_border = 6
        candidates = []

        # Preferred: outside the half near the centerline (same logic as before)
        if side == 'left':
            pref_x = half_outer.right() + margin_from_border
            pref_y = half_outer.top() + (half_outer.height() - box_h) / 2.0
            candidates.append(QRectF(pref_x, pref_y, box_w, box_h))
        else:
            pref_x = half_outer.left() - box_w - margin_from_border
            pref_y = half_outer.top() + (half_outer.height() - box_h) / 2.0
            candidates.append(QRectF(pref_x, pref_y, box_w, box_h))

        # Corner candidates inside the widget but outside the half area
        # top-right, bottom-right, top-left, bottom-left (relative to widget)
        candidates += [
            QRectF(widget_rect.right() - box_w - 8, widget_rect.top() + 8, box_w, box_h),  # top-right
            QRectF(widget_rect.right() - box_w - 8, widget_rect.bottom() - box_h - 8, box_w, box_h),  # bottom-right
            QRectF(widget_rect.left() + 8, widget_rect.top() + 8, box_w, box_h),  # top-left
            QRectF(widget_rect.left() + 8, widget_rect.bottom() - box_h - 8, box_w, box_h),  # bottom-left
        ]

        # Also try positions just inside the half next to the centerline (fallback)
        if side == 'left':
            inside_x = half_outer.right() - box_w - margin_from_border
        else:
            inside_x = half_outer.left() + margin_from_border
        inside_y = half_outer.top() + (half_outer.height() - box_h) / 2.0
        candidates.append(QRectF(inside_x, inside_y, box_w, box_h))

        # Choose the first candidate that doesn't intersect marker_rect and fits inside widget_rect
        chosen = None
        for cand in candidates:
            if not cand.intersects(marker_rect) and widget_rect.contains(cand):
                chosen = cand
                break

        # If none found, nudge away from marker in the direction opposite marker->centerline
        if chosen is None:
            # default to placing to the right of marker if space, else left, else above, else below
            try_right = QRectF(marker_rect.right() + 8, marker_rect.top(), box_w, box_h)
            try_left = QRectF(marker_rect.left() - 8 - box_w, marker_rect.top(), box_w, box_h)
            try_above = QRectF(marker_rect.left(), marker_rect.top() - 8 - box_h, box_w, box_h)
            try_below = QRectF(marker_rect.left(), marker_rect.bottom() + 8, box_w, box_h)
            for cand in (try_right, try_left, try_above, try_below):
                if widget_rect.contains(cand) and not cand.intersects(marker_rect):
                    chosen = cand
                    break

        # Final fallback: clamp inside widget_rect
        if chosen is None:
            # place top-right but keep it inside widget
            x = min(widget_rect.right() - box_w - 8, max(widget_rect.left() + 8, widget_rect.right() - box_w - 8))
            y = widget_rect.top() + 8
            chosen = QRectF(x, y, box_w, box_h)

        # Draw the box and text
        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        # translucent background but less opaque so it is less intrusive
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 140))  # semi-transparent
        painter.drawRoundedRect(chosen, 6, 6)

        # thin light border
        painter.setPen(QPen(QColor(200, 200, 200, 140), 1))
        painter.drawRoundedRect(chosen, 6, 6)

        # draw text lines
        painter.setPen(QPen(QColor(240, 240, 240, 220)))
        tx = chosen.left() + pad_x
        ty = chosen.top() + pad_y + fm.ascent()
        for line in text_lines:
            painter.drawText(QPointF(tx, ty), line)
            ty += fm.height()

        painter.restore()

    def draw_crosshair_marker(self, painter):
        """Draw an improved marker: soft shadow + outer ring + inner dot + small ticks + small unobtrusive label."""
        if not self.marker_pos:
            return

        pt = self._to_widget(*self.marker_pos)
        x, y = pt.x(), pt.y()

        # Visual parameters (pixels)
        shadow_radius = 12
        outer_radius = 7
        inner_radius = 3
        ring_width = 2
        tick_len = 6

        # DRAW soft shadow (semi-transparent big circle)
        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        shadow_color = QColor(0, 0, 0, 90)
        painter.setPen(Qt.NoPen)
        painter.setBrush(shadow_color)
        painter.drawEllipse(QPointF(x + 1.5, y + 1.5), shadow_radius, shadow_radius)

        # Outer ring
        ring_color = QColor(self.color)
        ring_color.setAlpha(220)
        painter.setPen(QPen(ring_color, ring_width))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(QPointF(x, y), outer_radius, outer_radius)

        # Inner filled core
        core_color = QColor(self.color)
        core_color.setAlpha(240)
        painter.setPen(Qt.NoPen)
        painter.setBrush(core_color)
        painter.drawEllipse(QPointF(x, y), inner_radius, inner_radius)

        # Small cross ticks to indicate precision
        tick_pen = QPen(ring_color, 1)
        painter.setPen(tick_pen)
        painter.drawLine(QPointF(x - tick_len, y), QPointF(x - outer_radius - 2, y))
        painter.drawLine(QPointF(x + tick_len, y), QPointF(x + outer_radius + 2, y))
        painter.drawLine(QPointF(x, y - tick_len), QPointF(x, y - outer_radius - 2))
        painter.drawLine(QPointF(x, y + tick_len), QPointF(x, y + outer_radius + 2))

        painter.restore()

        # If a snap candidate is nearby, draw a subtle halo at candidate location
        if self._near_candidate:
            try:
                cand_pt = self._to_widget(*self._near_candidate)
                painter.save()
                halo_col = QColor(255, 215, 0, 100)  # soft gold-ish halo
                painter.setPen(Qt.NoPen)
                painter.setBrush(halo_col)
                painter.drawEllipse(cand_pt, outer_radius + 6, outer_radius + 6)
                painter.restore()
            except Exception:
                pass

        # Draw small unobtrusive coordinate tooltip near marker, auto-offset so it doesn't cover the marker
        try:
            nx, ny = self.marker_pos
            abs_coords = self.get_absolute_pitch_coordinates()
            if abs_coords:
                ax, ay = abs_coords
                txt = f"{ax:.1f}m, {ay:.1f}m"
                # font smaller
                f = painter.font()
                f.setPointSize(max(8, f.pointSize() - 1))
                painter.setFont(f)
                fm = painter.fontMetrics()
                tw = fm.horizontalAdvance(txt)
                th = fm.height()

                # prefer top-right offset; if that would overlap marker, push further away
                offset_x, offset_y = 12, - (inner_radius + 8)
                tooltip_x = x + offset_x
                tooltip_y = y + offset_y - th

                # compute tooltip rect
                padding = 6
                rect = QRectF(tooltip_x, tooltip_y, tw + padding * 2, th + padding * 2)

                # if tooltip intersects marker bounding circle, move it to a corner (bottom-right)
                marker_rect = QRectF(x - shadow_radius, y - shadow_radius, shadow_radius * 2, shadow_radius * 2)
                if rect.intersects(marker_rect):
                    tooltip_x = x + 12
                    tooltip_y = y + 12
                    rect = QRectF(tooltip_x, tooltip_y, tw + padding * 2, th + padding * 2)

                # clamp to widget rect
                wrect = QRectF(self.rect())
                if rect.right() > wrect.right() - 4:
                    rect.moveRight(wrect.right() - 4)
                if rect.left() < wrect.left() + 4:
                    rect.moveLeft(wrect.left() + 4)
                if rect.top() < wrect.top() + 4:
                    rect.moveTop(wrect.top() + 4)
                if rect.bottom() > wrect.bottom() - 4:
                    rect.moveBottom(wrect.bottom() - 4)

                # draw translucent background
                painter.save()
                painter.setRenderHint(QPainter.TextAntialiasing, True)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 150))
                painter.drawRoundedRect(rect, 5, 5)
                # draw text
                painter.setPen(QPen(QColor(240, 240, 240, 230)))
                tx = rect.left() + padding
                ty = rect.top() + padding + fm.ascent()
                painter.drawText(QPointF(tx, ty), txt)
                painter.restore()
        except Exception:
            pass

    def draw_half_pitch(self, painter, pr):
        # Set up pen for field lines
        pen = QPen(Qt.white, 2)
        painter.setPen(pen)

        if self.half == 'left':
            self.draw_left_half(painter, pr)
        else:
            self.draw_right_half(painter, pr)

    def draw_left_half(self, painter, pr):
        """Draw left half of the pitch, edge-to-edge on the left, vertically centered."""
        # Real-world half pitch meters
        half_length_m = 60.0
        full_width_m = self.pitch_width  # 80.0m

        # compute uniform scale so circles remain circular
        scale_x = pr.width() / half_length_m
        scale_y = pr.height() / full_width_m
        scale = min(scale_x, scale_y)

        total_half_px_w = half_length_m * scale
        total_pitch_px_h = full_width_m * scale

        # EDGE-TO-EDGE behavior: left half should touch pr.left()
        x_offset = pr.left()
        # Center vertically so pitch looks well proportioned
        y_offset = pr.top() + (pr.height() - total_pitch_px_h) / 2.0

        def px(x_m, y_m):
            return QPointF(x_offset + x_m * scale, y_offset + y_m * scale)

        # outer rect of the half (used for goal alignment)
        half_outer = QRectF(x_offset, y_offset, total_half_px_w, total_pitch_px_h)
        # painter.drawRect(half_outer)
        painter.save()
        painter.setBrush(QColor(50, 160, 60))  # lighter grass for the active half
        painter.setPen(Qt.NoPen)
        painter.drawRect(half_outer)
        painter.restore()

        # now draw the outer rect / pitch lines
        painter.setPen(QPen(Qt.white, 2))
        painter.drawRect(half_outer)

        # center line (right edge of our half)
        painter.drawLine(QPointF(x_offset + total_half_px_w, y_offset),
                         QPointF(x_offset + total_half_px_w, y_offset + total_pitch_px_h))

        # center semicircle (on center line)
        center_y = y_offset + total_pitch_px_h / 2.0
        center_x = x_offset + total_half_px_w
        arc_radius_px = 9.15 * scale
        circle_rect = QRectF(center_x - arc_radius_px, center_y - arc_radius_px,
                             arc_radius_px * 2, arc_radius_px * 2)
        painter.drawArc(circle_rect, 90 * 16, 180 * 16)

        # center spot
        painter.drawEllipse(QPointF(center_x, center_y), 3, 3)

        # penalty area (18-yard box)
        penalty_width_m = 40.32
        penalty_depth_m = 16.5
        box_top_y = y_offset + (full_width_m / 2.0 - penalty_width_m / 2.0) * scale
        box_left_x = x_offset  # goal line at left edge
        penalty_box = QRectF(box_left_x, box_top_y, penalty_depth_m * scale, penalty_width_m * scale)
        painter.drawRect(penalty_box)

        # goal area (6-yard box)
        goal_area_width_m = 18.32
        goal_area_depth_m = 5.5
        ga_top_y = y_offset + (full_width_m / 2.0 - goal_area_width_m / 2.0) * scale
        ga_left_x = x_offset
        painter.drawRect(QRectF(ga_left_x, ga_top_y, goal_area_depth_m * scale, goal_area_width_m * scale))

        # penalty spot (11m from goal)
        penalty_spot_x = x_offset + 11.0 * scale
        penalty_spot_y = center_y
        painter.drawEllipse(QPointF(penalty_spot_x, penalty_spot_y), 3, 3)

        # penalty arc drawn only outside the box
        self.draw_penalty_arc(painter, QPointF(penalty_spot_x, penalty_spot_y),
                              arc_radius_px, box_left_x + penalty_depth_m * scale, side='left')

        # draw the goal aligned to half_outer (ensures no gap)
        self.draw_goal(painter, half_outer, 'left')
        self.draw_coords_overlay(painter, half_outer, 'left')

    def draw_right_half(self, painter, pr):
        """Draw right half of the pitch, edge-to-edge on the right, vertically centered."""
        half_length_m = 60.0
        full_width_m = self.pitch_width  # 80.0m

        scale_x = pr.width() / half_length_m
        scale_y = pr.height() / full_width_m
        scale = min(scale_x, scale_y)

        total_half_px_w = half_length_m * scale
        total_pitch_px_h = full_width_m * scale

        # EDGE-TO-EDGE behavior: right half should touch pr.right()
        x_offset = pr.right() - total_half_px_w
        y_offset = pr.top() + (pr.height() - total_pitch_px_h) / 2.0

        def px(x_m, y_m):
            return QPointF(x_offset + x_m * scale, y_offset + y_m * scale)

        half_outer = QRectF(x_offset, y_offset, total_half_px_w, total_pitch_px_h)
        # painter.drawRect(half_outer)
        painter.save()
        painter.setBrush(QColor(50, 160, 60))  # lighter grass for the active half
        painter.setPen(Qt.NoPen)
        painter.drawRect(half_outer)
        painter.restore()

        # now draw the outer rect / pitch lines
        painter.setPen(QPen(Qt.white, 2))
        painter.drawRect(half_outer)

        # center line (left edge of our half)
        painter.drawLine(QPointF(x_offset, y_offset), QPointF(x_offset, y_offset + total_pitch_px_h))

        # center semicircle (on center line)
        center_y = y_offset + total_pitch_px_h / 2.0
        center_x = x_offset
        arc_radius_px = 9.15 * scale
        circle_rect = QRectF(center_x - arc_radius_px, center_y - arc_radius_px,
                             arc_radius_px * 2, arc_radius_px * 2)
        painter.drawArc(circle_rect, 270 * 16, 180 * 16)

        # center spot
        painter.drawEllipse(QPointF(center_x, center_y), 3, 3)

        # penalty area (18-yard box) for right half (box at rightmost side)
        penalty_width_m = 40.32
        penalty_depth_m = 16.5
        box_top_y = y_offset + (full_width_m / 2.0 - penalty_width_m / 2.0) * scale
        box_right_x = x_offset + total_half_px_w
        penalty_box = QRectF(box_right_x - penalty_depth_m * scale, box_top_y,
                             penalty_depth_m * scale, penalty_width_m * scale)
        painter.drawRect(penalty_box)

        # goal area (6-yard box)
        goal_area_width_m = 18.32
        goal_area_depth_m = 5.5
        ga_top_y = y_offset + (full_width_m / 2.0 - goal_area_width_m / 2.0) * scale
        ga_right_x = x_offset + total_half_px_w
        painter.drawRect(QRectF(ga_right_x - goal_area_depth_m * scale, ga_top_y,
                                goal_area_depth_m * scale, goal_area_width_m * scale))

        # penalty spot at 11m from goal (from right edge)
        penalty_spot_x = x_offset + total_half_px_w - 11.0 * scale
        penalty_spot_y = center_y
        painter.drawEllipse(QPointF(penalty_spot_x, penalty_spot_y), 3, 3)

        # penalty arc
        self.draw_penalty_arc(painter, QPointF(penalty_spot_x, penalty_spot_y),
                              arc_radius_px, box_right_x - penalty_depth_m * scale, side='right')

        # draw goal aligned to half_outer
        self.draw_goal(painter, half_outer, 'right')
        self.draw_coords_overlay(painter, half_outer, 'right')

    def draw_penalty_arc(self, painter, spot_pt, arc_radius_px, penalty_box_edge_x, side):
        """
        Draw the penalty arc centered at spot_pt with radius arc_radius_px, but only the portion
        that lies outside the penalty box. penalty_box_edge_x is the x coordinate (widget coords)
        of the inner edge of the penalty area (for left side this is positive to the right of the spot).
        side is 'left' or 'right'.
        """

        cx = spot_pt.x()
        cy = spot_pt.y()
        r = arc_radius_px

        # If the vertical line of the penalty box doesn't intersect the circle, draw a full semicircle
        if side == 'left':
            dx = penalty_box_edge_x - cx  # positive if box edge lies to the right of the spot
            if dx <= -r:
                # box edge is entirely left of the circle => draw entire semicircle facing centerline
                arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
                painter.drawArc(arc_rect, int(90 * 16), int(180 * 16))
                return
            if dx >= r:
                # circle entirely inside penalty area: nothing to draw
                return
            # compute clipping angles
            cos_angle = dx / r
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            # Qt angles: 0 deg at 3-o'clock, positive CCW. We want arc centered at 0 degrees (to the right).
            start_deg = -angle_deg
            span_deg = 2 * angle_deg
            arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            painter.drawArc(arc_rect, int(start_deg * 16), int(span_deg * 16))

        else:  # right side
            dx = cx - penalty_box_edge_x  # positive if box edge lies to left of the spot
            if dx <= -r:
                # box edge entirely right of circle => draw entire semicircle
                arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
                painter.drawArc(arc_rect, int(270 * 16), int(180 * 16))
                return
            if dx >= r:
                # circle entirely inside penalty area: nothing to draw
                return
            cos_angle = dx / r
            cos_angle = max(-1.0, min(1.0, cos_angle))
            angle_rad = math.acos(cos_angle)
            angle_deg = math.degrees(angle_rad)
            # For the right side the arc is centered around 180 degrees (to the left)
            start_deg = 180 - angle_deg
            span_deg = 2 * angle_deg
            arc_rect = QRectF(cx - r, cy - r, 2 * r, 2 * r)
            painter.drawArc(arc_rect, int(start_deg * 16), int(span_deg * 16))

    def draw_goal(self, painter, half_outer: QRectF, side):
        """
        Draw the goal aligned to the provided half_outer QRectF (the outer boundary rectangle
        of the half pitch). This ensures the goal lines up exactly with the goal line used
        when drawing the penalty/6-yard boxes.
        """
        goal_width = (8.0 / self.pitch_width) * half_outer.height()  # 8m goal width
        goal_y = half_outer.top() + (half_outer.height() - goal_width) / 2  # Center vertically
        goal_depth = 25  # visual depth for net

        if side == 'left':
            goal_x = half_outer.left() - goal_depth
            goal_line_x = half_outer.left()
        else:
            goal_x = half_outer.right()
            goal_line_x = half_outer.right()

        net_color = QColor(60, 120, 60, 200)
        net_rect = QRectF(goal_x, goal_y - 5, goal_depth, goal_width + 10)
        painter.fillRect(net_rect, net_color)

        post_width = 6
        post_color = QColor(250, 250, 250)
        shadow_color = QColor(180, 180, 180)

        crossbar_height = 6
        crossbar_color = QColor(240, 240, 240)

        if side == 'left':
            # crossbar shadows and crossbars
            painter.fillRect(QRectF(goal_x + 2, goal_y - crossbar_height // 2 + 2, goal_depth, crossbar_height),
                             shadow_color)
            painter.fillRect(QRectF(goal_x, goal_y - crossbar_height // 2, goal_depth, crossbar_height), crossbar_color)

            painter.fillRect(
                QRectF(goal_x + 2, goal_y + goal_width - crossbar_height // 2 + 2, goal_depth, crossbar_height),
                shadow_color)
            painter.fillRect(QRectF(goal_x, goal_y + goal_width - crossbar_height // 2, goal_depth, crossbar_height),
                             crossbar_color)

            painter.fillRect(QRectF(goal_x + 2, goal_y + 2, post_width, goal_width), shadow_color)
            painter.fillRect(QRectF(goal_x, goal_y, post_width, goal_width), post_color)
        else:
            painter.fillRect(QRectF(goal_x - 2, goal_y - crossbar_height // 2 + 2, goal_depth, crossbar_height),
                             shadow_color)
            painter.fillRect(QRectF(goal_x, goal_y - crossbar_height // 2, goal_depth, crossbar_height), crossbar_color)

            painter.fillRect(
                QRectF(goal_x - 2, goal_y + goal_width - crossbar_height // 2 + 2, goal_depth, crossbar_height),
                shadow_color)
            painter.fillRect(QRectF(goal_x, goal_y + goal_width - crossbar_height // 2, goal_depth, crossbar_height),
                             crossbar_color)

            painter.fillRect(QRectF(goal_x + goal_depth - post_width + 2, goal_y + 2, post_width, goal_width),
                             shadow_color)
            painter.fillRect(QRectF(goal_x + goal_depth - post_width, goal_y, post_width, goal_width), post_color)

        goal_line_thickness = 4
        painter.fillRect(QRectF(goal_line_x - goal_line_thickness // 2, goal_y - post_width,
                                goal_line_thickness, goal_width + 2 * post_width), Qt.white)

        # net pattern
        painter.setPen(QPen(QColor(200, 200, 200, 150), 1))
        net_spacing = goal_width / 12
        for i in range(1, 12):
            y = goal_y + i * net_spacing
            painter.drawLine(QPointF(goal_x, y), QPointF(goal_x + goal_depth, y))

        net_h_spacing = goal_depth / 6
        for i in range(1, 6):
            x = goal_x + i * net_h_spacing
            painter.drawLine(QPointF(x, goal_y), QPointF(x, goal_y + goal_width))

        painter.setPen(QPen(QColor(200, 200, 200, 80), 1))
        for i in range(0, int(goal_width), 15):
            if side == 'left':
                painter.drawLine(QPointF(goal_x, goal_y + i),
                                 QPointF(goal_x + goal_depth, goal_y + i + goal_depth // 3))
            else:
                painter.drawLine(QPointF(goal_x + goal_depth, goal_y + i),
                                 QPointF(goal_x, goal_y + i + goal_depth // 3))

        painter.setPen(QPen(Qt.white, 2))

    def get_absolute_pitch_coordinates(self):
        """Convert normalized position to absolute pitch coordinates in meters"""
        if not self.marker_pos:
            return None

        norm_x, norm_y = self.marker_pos

        if self.half == 'left':
            # Left half: x goes from 0 to 60m
            pitch_x = norm_x * 60.0
        else:
            # Right half: x goes from 60 to 120m
            pitch_x = 60.0 + (norm_x * 60.0)

        # Y always goes from 0 to 80m
        pitch_y = norm_y * 80.0

        return (pitch_x, pitch_y)

    def _get_special_normalized_points(self):
        """
        Return a list of normalized positions (nx, ny) within the selected half
        that are meaningful snap targets. Uses the landmark dictionary in meters.

        Normalization:
            - For 'left' half: x ∈ [0..60] → nx = x/60
            - For 'right' half: x ∈ [60..120] → nx = (x-60)/60
            - y ∈ [0..80] always → ny = y/80
        """
        if not hasattr(self, "pitch_width"):
            self.pitch_width = 80.0
        half_length_m = 60.0
        full_width_m = self.pitch_width

        # landmarks (in meters) you shared
        landmarks = {
            "Top left corner": (0.0, 0.0),
            "Top right corner": (120.0, 0.0),
            "Bottom left corner": (0.0, 80.0),
            "Bottom right corner": (120.0, 80.0),
            "Center line top": (60.0, 0.0),
            "Center line bottom": (60.0, 80.0),
            "Center circle left": (50.0, 40.0),
            "Center circle top": (60.0, 30.0),
            "Center circle right": (70.0, 40.0),
            "Center circle bottom": (60.0, 50.0),
            "Left penalty spot": (12.0, 40.0),
            "Right penalty spot": (108.0, 40.0),
            "Left 6yd box top": (6.0, 30.0),
            "Left 6yd box bottom": (6.0, 50.0),
            "Right 6yd box top": (114.0, 30.0),
            "Right 6yd box bottom": (114.0, 50.0),
            "Left 18yd box top": (18.0, 18.0),
            "Left 18yd box bottom": (18.0, 62.0),
            "Right 18yd box top": (102.0, 18.0),
            "Right 18yd box bottom": (102.0, 62.0),
            "Left goal box top": (0.0, 30.0),
            "Left goal box bottom": (0.0, 50.0),
            "Right goal box top": (120.0, 30.0),
            "Right goal box bottom": (120.0, 50.0),
        }

        candidates = []

        for name, (xm, ym) in landmarks.items():
            # Only keep landmarks inside the currently active half
            if self.half == "left" and xm <= 60.0:
                nx = xm / half_length_m
                ny = ym / full_width_m
                candidates.append((nx, ny))
            elif self.half == "right" and xm >= 60.0:
                nx = (xm - 60.0) / half_length_m
                ny = ym / full_width_m
                candidates.append((nx, ny))

        return candidates

    def _maybe_snap(self, norm_pos, only_hint=False):
        """
        Given a normalized pos (nx, ny), check special points and return the closest
        (nx, ny) if within snap_threshold_px (in widget coordinates). If only_hint=True,
        return candidate for visual hint but don't commit any state.
        """
        try:
            px, py = None, None
            w_candidates = []
            for cand in self._get_special_normalized_points():
                # compute widget coords for candidate
                wp = self._to_widget(*cand)
                w_candidates.append((cand, wp))
            # convert input norm_pos to widget coords
            inp_widget = self._to_widget(*norm_pos)
            best = None
            best_dist = float('inf')
            for cand_norm, wp in w_candidates:
                d = math.hypot(wp.x() - inp_widget.x(), wp.y() - inp_widget.y())
                if d < best_dist:
                    best_dist = d
                    best = (cand_norm, wp, d)
            if best and best_dist <= self.snap_threshold_px:
                # candidate close enough to snap
                if only_hint:
                    # return normalized candidate for highlighting
                    return best[0]
                else:
                    return best[0]
            else:
                return None
        except Exception:
            return None

    def _update_snap_candidate(self, widget_pos):
        """
        Helper to set _near_candidate while dragging (widget_pos is a QPoint)
        """
        try:
            norm = self._to_norm(widget_pos)
            if not norm:
                self._near_candidate = None
                return
            cand = self._maybe_snap(norm, only_hint=True)
            self._near_candidate = cand
        except Exception:
            self._near_candidate = None


class DirectPitchAnnotationDialog(QDialog):
    """
    Direct pitch annotation dialog that combines your original UX (half selection,
    undo/redo, confirm/cancel, pitch widget) with a scrollable high-resolution
    reference frame preview on the right.

    Usage:
        dlg = DirectPitchAnnotationDialog(frame_pixmap=some_qpixmap, marker_color=Qt.red, marker_label="Shot Start", parent=self)
        if dlg.exec_() == QDialog.Accepted:
            pitch_pos = dlg.get_pitch_position()
            half = dlg.get_selected_half()
            abs_coords = dlg.get_absolute_pitch_coordinates()
    """

    def __init__(self, frame_pixmap=None, marker_color=Qt.red, marker_label="Shot Point", parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Direct Pitch Annotation – {marker_label}")
        self.resize(1000, 650)
        v = QVBoxLayout(self)

        # --- Group: choose left/right half ---
        grp = QGroupBox("Select Half for Annotation")
        grp.setStyleSheet("QGroupBox { font-weight: bold; }")
        hb = QHBoxLayout(grp)
        self.left_rb = QRadioButton("Left Half")
        self.right_rb = QRadioButton("Right Half")
        self.left_rb.setChecked(True)

        # Style the radio buttons
        radio_style = """
            QRadioButton {
                font-size: 11pt;
                padding: 5px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
            }
        """
        self.left_rb.setStyleSheet(radio_style)
        self.right_rb.setStyleSheet(radio_style)

        hb.addWidget(self.left_rb)
        hb.addWidget(self.right_rb)
        hb.addStretch(1)
        v.addWidget(grp)

        # --- Main horizontal area: pitch widget (left) + scrollable frame (right) ---
        main_h = QHBoxLayout()
        # --- Main horizontal area: pitch widget (left) + scrollable frame (right) ---
        main_h = QHBoxLayout()

        # Pitch widget (left) - reuse your DraggableFullPitch widget
        self.widget = None
        try:
            self.widget = DraggableFullPitch(marker_color)
            # ensure min height like original
            self.widget.setMinimumHeight(400)
            # ensure the pitch keeps a visible column width so it cannot be fully squeezed by the frame
            PITCH_COL_WIDTH = 420
            self.widget.setMinimumWidth(PITCH_COL_WIDTH)
            self.widget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            print("[DEBUG] DraggableFullPitch created successfully")
        except Exception as e:
            # fallback placeholder in case widget class is not available here
            print(f"[DEBUG] DraggableFullPitch creation failed: {e}")
            self.widget = QWidget()
            self.widget.setMinimumHeight(400)
            self.widget.setMinimumWidth(420)

        # Add the pitch widget as the first column
        main_h.addWidget(self.widget, 0)

        # --- Right: scrollable frame preview ---
        # --- Right: scrollable frame preview (both axes) ---
        right_v = QVBoxLayout()
        self.frame_scroll = QScrollArea()
        # keep pixmap natural size so scrollbars appear when required
        # and explicitly allow both scrollbars when needed
        self.frame_scroll.setWidgetResizable(False)
        self.frame_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.frame_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.frame_preview_label = QLabel()
        self.frame_preview_label.setAlignment(Qt.AlignCenter)
        # do not scale the pixmap (we want real pixel size so scrollbars appear)
        self.frame_preview_label.setScaledContents(False)
        self.frame_preview_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        if frame_pixmap is not None:
            try:
                if isinstance(frame_pixmap, QPixmap):
                    pm = frame_pixmap
                else:
                    pm = QPixmap.fromImage(frame_pixmap)

                # show full resolution and make label the pixmap size so scrollbars engage
                self.frame_preview_label.setPixmap(pm)
                self.frame_preview_label.setFixedSize(pm.size())

                # cap the scroll area width/height so the pitch column stays visible
                try:
                    screen = QApplication.primaryScreen()
                    screen_w = screen.size().width() if screen else 1600
                    screen_h = screen.size().height() if screen else 900
                except Exception:
                    screen_w, screen_h = 1600, 900

                pitch_col_w = getattr(self.widget, "minimumWidth", lambda: 420)() if callable(
                    getattr(self.widget, "minimumWidth", None)) else 420
                max_frame_w = max(600, int(screen_w - pitch_col_w - 200))
                max_frame_h = max(360, int(screen_h - 200))

                # let scroll area be no wider/taller than these caps (but will still show scrollbars if pixmap larger)
                self.frame_scroll.setMaximumWidth(min(pm.width(), max_frame_w))
                self.frame_scroll.setMaximumHeight(min(pm.height(), max_frame_h))

            except Exception as e:
                print(f"[DEBUG] Failed to set frame pixmap: {e}")
                self.frame_preview_label.setText("Reference frame preview unavailable.")
        else:
            self.frame_preview_label.setText("Reference frame preview unavailable.")

        self.frame_scroll.setWidget(self.frame_preview_label)
        right_v.addWidget(self.frame_scroll, 1)

        # Instruction text (same as your earlier instruction)
        instr = QLabel(
            "Click on the pitch (left) to place the shot marker. The frame is on the right is for reference.")
        instr.setWordWrap(True)
        right_v.addWidget(instr)

        main_h.addLayout(right_v, 1)

        # ensure left column (pitch) keeps its column and right column expands when available
        main_h.setStretch(0, 0)  # pitch: fixed-ish
        main_h.setStretch(1, 1)  # frame: takes the leftover space

        v.addLayout(main_h, 1)

        # --- Control buttons (undo/redo on left, confirm/cancel on right) ---
        ctrl = QHBoxLayout()

        # Undo / Redo buttons (styled similarly to your original)
        self.undo_btn = QPushButton("Undo")
        self.redo_btn = QPushButton("Redo")

        btn_style = """
            QPushButton {
                font-size: 10pt;
                padding: 6px 12px;
                border: 2px solid #ccc;
                border-radius: 4px;
                background: white;
            }
            QPushButton:hover {
                background: #f0f0f0;
                border-color: #999;
            }
            QPushButton:disabled {
                background: #f5f5f5;
                color: #999;
                border-color: #ddd;
            }
        """
        self.undo_btn.setStyleSheet(btn_style)
        self.redo_btn.setStyleSheet(btn_style)

        ctrl.addWidget(self.undo_btn)
        ctrl.addWidget(self.redo_btn)
        ctrl.addStretch()

        # Confirm / Cancel buttons (styled)
        self.confirm_btn = QPushButton("✓ Confirm Position")
        self.cancel_btn = QPushButton("✗ Cancel")

        confirm_style = """
            QPushButton {
                font-size: 11pt;
                font-weight: bold;
                padding: 8px 16px;
                border-radius: 4px;
                background: #28a745;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background: #218838;
            }
            QPushButton:disabled {
                background: #6c757d;
            }
        """

        cancel_style = """
            QPushButton {
                font-size: 11pt;
                padding: 8px 16px;
                border-radius: 4px;
                background: #dc3545;
                color: white;
                border: none;
            }
            QPushButton:hover {
                background: #c82333;
            }
        """

        self.confirm_btn.setStyleSheet(confirm_style)
        self.cancel_btn.setStyleSheet(cancel_style)

        ctrl.addWidget(self.confirm_btn)
        ctrl.addWidget(self.cancel_btn)
        v.addLayout(ctrl)

        # --- Connections & signals ---
        # Left/right radio toggles -> set half on widget if widget supports it
        def on_half_toggled(val):
            half = 'left' if self.left_rb.isChecked() else 'right'
            if hasattr(self.widget, "set_half"):
                try:
                    self.widget.set_half(half)
                except Exception:
                    pass

        self.left_rb.toggled.connect(on_half_toggled)
        self.right_rb.toggled.connect(on_half_toggled)

        # Undo / Redo -> call widget methods if present
        if hasattr(self.widget, "undo"):
            self.undo_btn.clicked.connect(self.widget.undo)
        else:
            self.undo_btn.setEnabled(False)
        if hasattr(self.widget, "redo"):
            self.redo_btn.clicked.connect(self.widget.redo)
        else:
            self.redo_btn.setEnabled(False)

        # Confirm / Cancel
        self.confirm_btn.clicked.connect(self.on_accept)
        self.cancel_btn.clicked.connect(self.reject)

        # When widget state changes, update UI state (enable/disable confirm)
        if hasattr(self.widget, "markerChanged"):
            try:
                self.widget.markerChanged.connect(self.update_ui_state)
            except Exception:
                # signal exist check fallback
                pass

        # If your widget exposes a property `marker_pos` and undo/redo stacks, use those
        # Initialize UI state
        self.update_ui_state()

    def update_ui_state(self):
        """Update UI state based on current marker position and undo/redo availability."""
        has_marker = False
        try:
            has_marker = bool(getattr(self.widget, "marker_pos", None))
        except Exception:
            has_marker = False

        self.confirm_btn.setEnabled(has_marker)

        # Update undo/redo button states based on widget stacks if present
        try:
            self.undo_btn.setEnabled(bool(getattr(self.widget, "undo_stack", None)))
        except Exception:
            self.undo_btn.setEnabled(False)
        try:
            self.redo_btn.setEnabled(bool(getattr(self.widget, "redo_stack", None)))
        except Exception:
            self.redo_btn.setEnabled(False)

    def get_pitch_position(self):
        """Return the marker position in normalized coordinates (0-1, 0-1) within the selected half."""
        try:
            return getattr(self.widget, "marker_pos", None)
        except Exception:
            return None

    def get_selected_half(self):
        """Return which half is currently selected ('left'|'right')."""
        return 'left' if self.left_rb.isChecked() else 'right'

    def get_absolute_pitch_coordinates(self):
        """Convert normalized position to absolute pitch coordinates in meters, via widget helper."""
        if hasattr(self.widget, "get_absolute_pitch_coordinates"):
            try:
                return self.widget.get_absolute_pitch_coordinates()
            except Exception:
                return None
        return None

    def on_accept(self):
        """Called when user clicks Confirm. ensures a marker exists before accept."""
        if self.get_pitch_position() is None:
            resp = self._show_confirm_dialog("No marker placed",
                                             "No pitch marker detected. Do you want to accept without placing a pitch marker?")
            if resp != QMessageBox.Yes:
                return
        self.accept()

    def _show_confirm_dialog(self, title, text):
        resp = QMessageBox.question(self, title, text, QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        return resp

class LandmarkMappingDialog(QDialog):
    def __init__(self, frame_pixmaps, segment, landmarks, edges, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Pitch Landmark Selection")
        self.resize(1200, 800)

        # Main horizontal layout with better stretch factors
        layout = QHBoxLayout(self)
        layout.setSpacing(10)  # Add some spacing between sections
        self.frames = frame_pixmaps
        self.segment = segment
        self.current_frame = self.frames[0]

        # — Left: Frame + scrolling —
        left_container = QWidget()
        left = QVBoxLayout(left_container)
        left.setContentsMargins(0, 0, 0, 0)

        self.frame_label = DraggableFrameLabel()
        pix = self._to_pixmap(self.current_frame)
        self.frame_label.setPixmap(pix)
        self.frame_label.setAlignment(Qt.AlignCenter)
        self.frame_label.setMinimumSize(pix.width(), pix.height())

        frame_scroll = QScrollArea()
        frame_scroll.setWidgetResizable(False)
        frame_scroll.setWidget(self.frame_label)
        frame_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        frame_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        left.addWidget(frame_scroll, 1)

        BUTTON_CSS = """
                    QPushButton {
                        background-color: #444444;
                        color: white;
                        border: 1px solid #666666;
                        border-radius: 4px;
                        padding: 5px 15px;
                        margin: 8px;
                        font-size: 14px;
                    }
                    QPushButton:hover {
                        background-color: #666666;
                    }
                """

        nav = QHBoxLayout()
        browse_btn = QPushButton("Browse Frames")
        browse_btn.clicked.connect(self.browse_frames)
        browse_btn.setStyleSheet(BUTTON_CSS)
        nav.addWidget(browse_btn)
        left.addLayout(nav)

        inst = QLabel("<b>Click & drag points from the pitch grid (right) onto video frame on left.</b><br>"
                      "Map at least 4 points. If not possible, click browse frames to look for more in other frames")
        inst.setStyleSheet(
            "color: #444; font-size: 15px; background: #f9f9f9; "
            "padding: 8px; border-radius: 8px;"
        )
        inst.setWordWrap(True)
        left.addWidget(inst)

        # — Right: Grid + scrolling —
        right_container = QWidget()
        right = QVBoxLayout(right_container)
        right.setContentsMargins(0, 0, 0, 0)

        self.grid_widget = SchematicPitchGridWidget(landmarks, edges)

        # Calculate appropriate size for the grid based on the pitch dimensions
        ppm = 7  # pixels per meter
        w_m = self.grid_widget.pitch_config.length
        h_m = self.grid_widget.pitch_config.width
        m = self.grid_widget.margin

        grid_scroll = QScrollArea()
        grid_scroll.setWidgetResizable(False)  # Important: don't resize the widget to fit
        grid_scroll.setWidget(self.grid_widget)

        # Set a fixed viewport size for the scroll area to ensure scrollbars appear
        grid_scroll_viewport_width = min(500, int(w_m * ppm + 2 * m))  # Limit to 500px or calculated size
        grid_scroll_viewport_height = min(500, int(h_m * ppm + 2 * m))  # Limit to 500px or calculated size
        grid_scroll.setMinimumSize(grid_scroll_viewport_width, grid_scroll_viewport_height)

        grid_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        grid_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        grid_scroll.setStyleSheet("""
            QScrollArea {
                background-color: #1A1A1A;
                border: 1px solid #444;
            }
            QScrollBar:vertical, QScrollBar:horizontal {
                background: #2D2D2D; width:12px; height:12px; margin:0px;
            }
            QScrollBar::handle {
                background: #444; border-radius:6px;
            }
            QScrollBar::handle:hover {
                background: #666;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background:none; height:0px; width:0px;
            }
        """)
        right.addWidget(grid_scroll, 1)

        # — Buttons —
        buttons_layout = QVBoxLayout()
        self.confirm_btn = QPushButton("Confirm Mapping")
        self.confirm_btn.clicked.connect(self.on_confirm)
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.clicked.connect(self.frame_label.undo)
        self.redo_btn = QPushButton("Redo")
        self.redo_btn.clicked.connect(self.frame_label.redo)

        for b in (self.confirm_btn, self.undo_btn, self.redo_btn):
            b.setStyleSheet(BUTTON_CSS)
            buttons_layout.addWidget(b)

        right.addLayout(buttons_layout)

        # Add the left and right containers to the main layout with equal stretching
        layout.addWidget(left_container, 1)  # 50% width
        layout.addWidget(right_container, 1)  # 50% width

        # — Connect drag/drop —
        self.grid_widget.landmarkDragged.connect(self.frame_label.startDrag)
        self.grid_widget.landmarkDropped.connect(self.frame_label.endDrag)
        self.frame_label.set_grid_widget(self.grid_widget)

        self.setLayout(layout)

    def get_mapped_points(self):
        return self.frame_label.mapped_points

    @property
    def selected_ids(self):
        return list(self.get_mapped_points().keys())

    def _to_pixmap(self, frame_np):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb_frame.shape
        bytes_per_line = 3 * w
        img = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def browse_frames(self):
        dlg = FrameSelectorDialog(self.frames, parent=self, segment=self.segment)
        if dlg.exec_():
            self.current_frame = dlg.selected_frame
            pix = self._to_pixmap(self.current_frame)
            self.frame_label.setPixmap(pix)
            self.frame_label.setMinimumSize(pix.width(), pix.height())
            self.frame_label.update()

    def on_confirm(self):
        if len(self.frame_label.mapped_points) < 4:
            # prompt the user to browse more frames
            dlg = FrameSelectorDialog(self.frames, parent=self, segment=self.segment)
            if dlg.exec_():
                # user selected a new frame: replace and continue
                self.current_frame = dlg.selected_frame
                pix = self._to_pixmap(self.current_frame)
                self.frame_label.setPixmap(pix)
                self.frame_label.setMinimumSize(pix.width(), pix.height())
                self.frame_label.update()
            else:
                # user cancelled browsing: stay here
                return
        else:
            # enough points: accept and close
            self.accept()

class XGShotPlotDialog(QDialog):
    def __init__(self, start_pitch, end_pitch, xg_val, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Expected Goals (xG) Analysis")
        self.setFixedSize(1200, 700)

        if parent and hasattr(parent, 'windowIcon'):
            self.setWindowIcon(parent.windowIcon())

        # Store original coordinates for display
        self.original_start = start_pitch
        self.original_end = end_pitch

        # Determine which half-pitch to show and transform coordinates
        self.determine_pitch_side_and_transform(start_pitch, end_pitch)

        self.shot_distance, self.shot_angle = self.calculate_shot_metrics(start_pitch, end_pitch)

        self.shot_annotations = [{
            'start_pitch': {'x': self.transformed_start[0], 'y': self.transformed_start[1]},
            'end_pitch': {'x': self.transformed_end[0], 'y': self.transformed_end[1]},
            'original_start': {'x': start_pitch[0], 'y': start_pitch[1]},
            'original_end': {'x': end_pitch[0], 'y': end_pitch[1]},
            'xG': xg_val
        }]

        # Half-pitch dimensions (cm) - converted for calculations
        self.pitch_x0 = 6000  # 60m (halfway line)
        self.pitch_x1 = 12000  # 120m (goal line)
        self.pitch_wid = 8000  # 80m
        self.pitch_len = self.pitch_x1 - self.pitch_x0  # 60m (6000cm)
        self.goal_depth = 200  # 2m

        # Feature distances (cm)
        self.center_circle_r = 915  # 9.15 m
        self.goal_wid = 732  # 7.32 m
        self.goal_dep = 200  # 2 m
        self.six_w = 1832  # 18.32 m
        self.six_d = 550  # 5.5 m
        self.eighteen_w = 4032  # 40.32 m
        self.eighteen_d = 1650  # 16.5 m
        self.penalty_dist = 1200  # 11 m

        self.margin = 20
        self.gap = 15
        self.legend_vals = [0.05, 0.20, 0.35, 0.75]
        self.legend_labels = ["Low xG", "Medium", "High", "Very High"]

        # Hover tracking
        self.hover_pos = None
        self.hover_text = ""

        self.setup_ui()
        self.compute_layout()
        self.setMouseTracking(True)

    def determine_pitch_side_and_transform(self, start_pitch, end_pitch):
        """Determine which half-pitch to show and transform coordinates for consistent upward attack."""
        sx, sy = start_pitch[0], start_pitch[1]
        ex, ey = end_pitch[0], end_pitch[1]

        # Determine which goal is being targeted
        x_direction = ex - sx

        # Determine target goal based on trajectory and position
        if x_direction < 0 or (x_direction == 0 and sx < 60):  # Targeting left goal
            target_goal_x = 0
            self.target_side = "left"
        else:  # Targeting right goal
            target_goal_x = 120
            self.target_side = "right"

        print(f"[DEBUG] Target goal at x={target_goal_x}, side: {self.target_side}")

        # Always transform so that the attack moves toward x=0 (top of display)
        if target_goal_x == 0:  # Left goal - no transformation needed
            self.transformed_start = (sx, sy)
            self.transformed_end = (ex, ey)
            print(f"[DEBUG] Left goal attack - no transformation")
        else:  # Right goal - flip coordinates
            self.transformed_start = (120 - sx, sy)
            self.transformed_end = (120 - ex, ey)
            print(f"[DEBUG] Right goal attack - flipping coordinates")

        # Ensure the transformed shot moves toward lower x values (toward goal)
        transform_x_direction = self.transformed_end[0] - self.transformed_start[0]
        if transform_x_direction > 0:  # If still moving away from goal, something's wrong
            print(f"[WARNING] Shot appears to move away from goal after transformation!")
            print(f"[WARNING] Consider checking coordinate interpretation")

        # print(f"[DEBUG] Original start: ({sx:.1f}, {sy:.1f}), end: ({ex:.1f}, {ey:.1f})")
        # print(f"[DEBUG] Transformed start: ({self.transformed_start[0]:.1f}, {self.transformed_start[1]:.1f})")
        # print(f"[DEBUG] Transformed end: ({self.transformed_end[0]:.1f}, {self.transformed_end[1]:.1f})")
        # print(f"[DEBUG] Transform direction: {transform_x_direction:.1f} (should be negative)")

    def calculate_shot_metrics(self, start_pitch, end_pitch):
        """Calculate shot distance and angle using the same logic as XGModel."""
        sx, sy = start_pitch[0], start_pitch[1]
        ex, ey = end_pitch[0], end_pitch[1]

        # Determine target goal using same logic as XGModel
        goal_center, left_post, right_post = self.determine_target_goal((sx, sy), (ex, ey))

        # Calculate distance to goal center
        start = np.array([sx, sy], dtype=np.float32)
        distance = np.linalg.norm(goal_center - start)

        # Calculate shot angle
        ld = np.linalg.norm(left_post - start)
        rd = np.linalg.norm(right_post - start)
        gw = np.linalg.norm(left_post - right_post)

        if ld == 0 or rd == 0:
            angle = 0.0
        else:
            cos_a = (ld ** 2 + rd ** 2 - gw ** 2) / (2 * ld * rd)
            cos_a = max(min(cos_a, 1.0), -1.0)
            angle = math.degrees(math.acos(cos_a))

        print(f"[DEBUG] Shot metrics - Distance: {distance:.2f}m, Angle: {angle:.2f}°")
        print(f"[DEBUG] Target goal center: ({goal_center[0]}, {goal_center[1]})")

        return distance, angle

    def determine_target_goal(self, start_pos, end_pos=None):
        """Determine which goal the shot is targeting."""
        sx, sy = start_pos

        # Define both goals based on coordinate system
        left_goal_center = np.array([0.0, 40.0], dtype=np.float32)
        left_goal_left_post = np.array([0.0, 30.0], dtype=np.float32)
        left_goal_right_post = np.array([0.0, 50.0], dtype=np.float32)

        right_goal_center = np.array([120.0, 40.0], dtype=np.float32)
        right_goal_left_post = np.array([120.0, 30.0], dtype=np.float32)
        right_goal_right_post = np.array([120.0, 50.0], dtype=np.float32)

        # If we have end position, use trajectory to determine target
        if end_pos is not None:
            ex, ey = end_pos
            x_direction = ex - sx

            if x_direction < 0:  # Moving towards left goal
                target_goal = "left"
                goal_center = left_goal_center
                left_post = left_goal_left_post
                right_post = left_goal_right_post
            else:  # Moving towards right goal
                target_goal = "right"
                goal_center = right_goal_center
                left_post = right_goal_left_post
                right_post = right_goal_right_post
        else:
            # No end position, use nearest goal
            dist_to_left = np.linalg.norm(left_goal_center - np.array([sx, sy]))
            dist_to_right = np.linalg.norm(right_goal_center - np.array([sx, sy]))

            if dist_to_left < dist_to_right:
                target_goal = "left"
                goal_center = left_goal_center
                left_post = left_goal_left_post
                right_post = left_goal_right_post
            else:
                target_goal = "right"
                goal_center = right_goal_center
                left_post = right_goal_left_post
                right_post = right_goal_right_post

        print(f"[DEBUG] Dialog - Target goal: {target_goal}")
        return goal_center, left_post, right_post

    def setup_ui(self):
        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(15, 15, 15, 15)

        # Left side - pitch area (this widget will handle drawing)
        self.pitch_area = QWidget()
        self.pitch_area.setMinimumSize(750, 650)
        self.pitch_area.paintEvent = self.paintEvent
        self.pitch_area.mouseMoveEvent = self.mouseMoveEvent
        self.pitch_area.setMouseTracking(True)
        main_layout.addWidget(self.pitch_area, stretch=3)

        # Right side - info panel
        info_panel = self.create_info_panel()
        main_layout.addWidget(info_panel, stretch=1)

    def create_info_panel(self):
        panel = QWidget()
        panel.setFixedWidth(280)
        panel.setStyleSheet("""
            QWidget {
                background: white;
                border: 2px solid #dee2e6;
                border-radius: 12px;
            }
        """)

        layout = QVBoxLayout(panel)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Shot Analysis")
        title_label.setStyleSheet("""
            font-size: 18pt;
            font-weight: bold;
            color: #343a40;
            border: none;
            padding: 0px;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        # xG Value
        if self.shot_annotations[0]['xG'] is None:
            xg_text = "xG: Pending..."
            xg_color = "#6c757d"
            bg_color = "#f8f9fa"
        else:
            xg_val = self.shot_annotations[0]['xG']
            xg_text = f"xG: {xg_val:.3f}\n({xg_val * 100:.1f}%)"
            # Color code based on xG value
            if xg_val < 0.1:
                xg_color = "#dc3545"
                bg_color = "#f8d7da"
            elif xg_val < 0.3:
                xg_color = "#fd7e14"
                bg_color = "#ffeaa7"
            elif xg_val < 0.6:
                xg_color = "#ffc107"
                bg_color = "#fff3cd"
            else:
                xg_color = "#28a745"
                bg_color = "#d4edda"

        self.xg_label = QLabel(xg_text)
        self.xg_label.setStyleSheet(f"""
            font-size: 16pt;
            font-weight: bold;
            color: {xg_color};
            background: {bg_color};
            padding: 15px;
            border-radius: 8px;
            border: 2px solid {xg_color};
        """)
        self.xg_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.xg_label)

        # Shot statistics
        stats_container = QWidget()
        stats_layout = QVBoxLayout(stats_container)
        stats_layout.setSpacing(10)

        distance_label = QLabel(f"Distance: {self.shot_distance:.1f}m")
        distance_label.setStyleSheet("""
            font-size: 12pt;
            color: #495057;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        """)
        distance_label.setAlignment(Qt.AlignCenter)

        angle_label = QLabel(f"Angle: {self.shot_angle:.1f}°")
        angle_label.setStyleSheet("""
            font-size: 12pt;
            color: #495057;
            background: #f8f9fa;
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #dee2e6;
        """)
        angle_label.setAlignment(Qt.AlignCenter)

        stats_layout.addWidget(distance_label)
        stats_layout.addWidget(angle_label)
        layout.addWidget(stats_container)

        # Add stretch to push legend and button to bottom
        layout.addStretch()

        # Legend (compact version for side panel)
        legend_container = self.create_compact_legend()
        layout.addWidget(legend_container)

        # Close button
        self.ok_btn = QPushButton("Close Analysis")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: bold;
                background: #007bff;
                color: white;
                border-radius: 8px;
                padding: 12px 20px;
                border: none;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton:pressed {
                background: #004085;
            }
        """)
        self.ok_btn.clicked.connect(self.accept)
        layout.addWidget(self.ok_btn)

        return panel

    def create_compact_legend(self):
        container = QWidget()
        container.setStyleSheet("""
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 12px;
        """)

        layout = QVBoxLayout(container)
        layout.setSpacing(8)

        # Title
        title = QLabel("xG Scale")
        title.setStyleSheet("font-weight: bold; color: #495057; font-size: 11pt;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # Legend items in grid
        grid_layout = QGridLayout()
        grid_layout.setSpacing(8)

        colors = [QColor(220, 53, 69), QColor(253, 126, 20), QColor(255, 193, 7), QColor(40, 167, 69)]

        for i, (val, label, color) in enumerate(zip(self.legend_vals, self.legend_labels, colors)):
            row = i // 2
            col = i % 2

            item_layout = QHBoxLayout()
            item_layout.setSpacing(5)

            # Color circle
            circle_label = QLabel()
            circle_label.setFixedSize(12, 12)
            circle_label.setStyleSheet(f"""
                background: {color.name()};
                border: 1px solid #333;
                border-radius: 6px;
            """)

            # Text
            text_label = QLabel(f"{label}\n≤{val:.2f}")
            text_label.setStyleSheet("font-size: 9pt; color: #495057;")

            item_layout.addWidget(circle_label)
            item_layout.addWidget(text_label)
            item_layout.addStretch()

            item_widget = QWidget()
            item_widget.setLayout(item_layout)
            grid_layout.addWidget(item_widget, row, col)

        grid_widget = QWidget()
        grid_widget.setLayout(grid_layout)
        layout.addWidget(grid_widget)

        return container

    def compute_layout(self):
        # Calculate available space for pitch (excluding the right panel)
        panel_width = 280 + 30  # Info panel + margins
        available_width = self.width() - panel_width - 40  # Extra margin for clipping
        available_height = self.height() - 60  # Top/bottom margins

        # Pitch dimensions with proper aspect ratio
        target_ratio = self.pitch_wid / self.pitch_len  # 80/60 = 1.33

        ph = available_height - 40  # Leave some margin
        pw = ph * target_ratio

        if pw > available_width - 40:
            pw = available_width - 40
            ph = pw / target_ratio

        x0 = self.margin + (available_width - pw) / 2
        y0 = self.margin + (available_height - ph) / 2

        self.pitch_rect = QRectF(x0, y0, pw, ph)

    def get_xg_color(self, xg_val):
        """Get color based on xG value."""
        if xg_val < 0.1:
            return QColor(220, 53, 69)  # Red
        elif xg_val < 0.3:
            return QColor(253, 126, 20)  # Orange
        elif xg_val < 0.6:
            return QColor(255, 193, 7)  # Yellow
        else:
            return QColor(40, 167, 69)  # Green

    def get_circle_size(self, xg_val):
        """Get circle size based on xG value."""
        base_size = 2
        return base_size + (xg_val * 12)  # Scale from 8 to 28 pixels

    def mouseMoveEvent(self, event):
        # Check if mouse is over shot positions
        shot = self.shot_annotations[0]
        start_pos = self.field_to_widget(shot['start_pitch']['x'], shot['start_pitch']['y'])
        end_pos = self.field_to_widget(shot['end_pitch']['x'], shot['end_pitch']['y'])

        mouse_pos = event.pos()

        # Check hover over start position - show original coordinates
        if self.distance_to_point(mouse_pos, start_pos) < 15:
            self.hover_pos = mouse_pos
            self.hover_text = f"Start: ({shot['original_start']['x']:.1f}m, {shot['original_start']['y']:.1f}m)"
            self.update()
        # Check hover over end position - show original coordinates
        elif self.distance_to_point(mouse_pos, end_pos) < 15:
            self.hover_pos = mouse_pos
            self.hover_text = f"End: ({shot['original_end']['x']:.1f}m, {shot['original_end']['y']:.1f}m)"
            self.update()
        else:
            if self.hover_pos is not None:
                self.hover_pos = None
                self.hover_text = ""
                self.update()

    def distance_to_point(self, p1, p2):
        return math.sqrt((p1.x() - p2.x()) ** 2 + (p1.y() - p2.y()) ** 2)

    def field_to_widget(self, field_x, field_y):
        """Convert field coordinates (meters) to widget coordinates (pixels)."""
        rect = self.pitch_rect

        # Map x coordinate (0-60m maps to full height of pitch rect)
        # field_x=0 should be at goal (top), field_x=60 should be at halfway line (bottom)
        widget_y = rect.y() + (field_x / 60.0) * rect.height()

        # Try flipping the y-coordinate mapping:
        widget_x = rect.x() + ((80.0 - field_y) / 80.0) * rect.width()

        # # Debug output to verify mapping
        # print(f"[DEBUG] Field coords ({field_x:.1f}, {field_y:.1f}) -> Widget coords ({widget_x:.1f}, {widget_y:.1f})")
        # print(f"[DEBUG] Rect: x={rect.x():.1f}, y={rect.y():.1f}, w={rect.width():.1f}, h={rect.height():.1f}")

        return QPointF(widget_x, widget_y)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(34, 139, 34))  # Forest green

        self.draw_pitch(painter)
        self.draw_shot(painter)
        self.draw_hover_tooltip(painter)

    def draw_pitch(self, painter):
        """Draw the half pitch with all markings - improved penalty arc and goal."""
        rect = self.pitch_rect

        # Set up pen for field lines
        pen = QPen(Qt.white, 2)
        painter.setPen(pen)

        # Draw pitch boundary
        painter.drawRect(rect)

        # Goal area (6-yard box)
        goal_width = (18.32 / 80.0) * rect.width()  # 18.32m in 80m width
        goal_depth = (5.5 / 60.0) * rect.height()  # 5.5m in 60m length
        goal_x = rect.x() + (rect.width() - goal_width) / 2
        goal_y = rect.y()
        painter.drawRect(QRectF(goal_x, goal_y, goal_width, goal_depth))

        # Penalty area (18-yard box)
        penalty_width = (40.32 / 80.0) * rect.width()  # 40.32m in 80m width
        penalty_depth = (16.5 / 60.0) * rect.height()  # 16.5m in 60m length
        penalty_x = rect.x() + (rect.width() - penalty_width) / 2
        penalty_y = rect.y()
        penalty_box = QRectF(penalty_x, penalty_y, penalty_width, penalty_depth)
        painter.drawRect(penalty_box)

        # Penalty spot
        penalty_spot_y = rect.y() + (11.0 / 60.0) * rect.height()  # 11m from goal
        penalty_spot_x = rect.x() + rect.width() / 2
        painter.drawEllipse(QPointF(penalty_spot_x, penalty_spot_y), 3, 3)

        # Improved Penalty arc - cleaner "D" shape
        arc_radius = (9.15 / 60.0) * rect.height()  # 9.15m radius in widget coordinates

        # Center the arc on the penalty spot
        arc_center_x = penalty_spot_x
        arc_center_y = penalty_spot_y

        # Calculate where the arc intersects with the penalty box edge
        penalty_box_edge = penalty_y + penalty_depth

        # Only draw the arc portion that extends beyond the penalty box
        if arc_center_y + arc_radius > penalty_box_edge:
            # Calculate the angle where arc intersects penalty box
            dy = penalty_box_edge - arc_center_y
            if abs(dy) < arc_radius:
                # Calculate intersection angle more precisely
                cos_angle = dy / arc_radius
                cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp to valid range
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)

                # Create arc rectangle
                arc_rect = QRectF(arc_center_x - arc_radius, arc_center_y - arc_radius,
                                  arc_radius * 2, arc_radius * 2)

                # Draw the arc from left intersection to right intersection
                # Qt angles: 0° = 3 o'clock, 90° = 6 o'clock, measured in 1/16 degrees
                start_angle = int((270 - angle_deg) * 16)  # Start from left intersection
                span_angle = int(2 * angle_deg * 16)  # Span to right intersection

                painter.drawArc(arc_rect, start_angle, span_angle)

        # Center circle (half) - at bottom edge
        center_y = rect.y() + rect.height()  # Bottom edge of pitch
        center_x = rect.x() + rect.width() / 2
        circle_radius = (9.15 / 60.0) * rect.height()  # 9.15m radius

        # Draw half circle at halfway line
        circle_rect = QRectF(center_x - circle_radius, center_y - circle_radius,
                             circle_radius * 2, circle_radius * 2)
        painter.drawArc(circle_rect, 0, 180 * 16)  # Top half of circle

        # Center spot at halfway line
        painter.drawEllipse(QPointF(center_x, center_y), 3, 3)

        # GOAL STRUCTURE
        goal_post_width = (8.00 / 80.0) * rect.width()  # 8.00m goal width
        goal_post_x = rect.x() + (rect.width() - goal_post_width) / 2
        goal_post_y = rect.y() - 25  # Moved further up for bigger appearance
        goal_depth = 20  # Much deeper goal for 3D effect

        # Goal net area (darker green background for depth)
        net_color = QColor(60, 120, 60, 200)  # Darker, more opaque
        painter.fillRect(QRectF(goal_post_x - 5, goal_post_y, goal_post_width + 10, goal_depth), net_color)

        # Goal crossbar (thicker and more prominent)
        crossbar_height = 6  # Increased thickness
        crossbar_color = QColor(240, 240, 240)  # Slightly off-white
        painter.fillRect(QRectF(goal_post_x - 3, goal_post_y + goal_depth - crossbar_height / 2,
                                goal_post_width + 6, crossbar_height), crossbar_color)

        # Goal posts (much thicker with better 3D effect)
        post_width = 6  # Increased from 4
        post_height = goal_depth + crossbar_height / 2

        # Left post with enhanced shadow effect
        shadow_color = QColor(180, 180, 180)
        post_color = QColor(250, 250, 250)

        # Left post shadow
        painter.fillRect(QRectF(goal_post_x - post_width / 2 - 2, goal_post_y + 2,
                                post_width, post_height), shadow_color)
        # Left post main
        painter.fillRect(QRectF(goal_post_x - post_width / 2, goal_post_y,
                                post_width, post_height), post_color)

        # Right post shadow
        painter.fillRect(QRectF(goal_post_x + goal_post_width - post_width / 2 + 2, goal_post_y + 2,
                                post_width, post_height), shadow_color)
        # Right post main
        painter.fillRect(QRectF(goal_post_x + goal_post_width - post_width / 2, goal_post_y,
                                post_width, post_height), post_color)
        # Enhanced goal line (thicker and more visible)
        goal_line_thickness = 4
        painter.fillRect(QRectF(goal_post_x - post_width, rect.y() - goal_line_thickness / 2,
                                goal_post_width + 2 * post_width, goal_line_thickness), Qt.white)

        # More realistic net pattern
        painter.setPen(QPen(QColor(200, 200, 200, 150), 1))  # Lighter, more visible net

        # Vertical net lines (more of them)
        net_spacing = goal_post_width / 12  # More vertical lines
        for i in range(1, 12):
            x = goal_post_x + i * net_spacing
            painter.drawLine(QPointF(x, goal_post_y), QPointF(x, goal_post_y + goal_depth))

        # Horizontal net lines (more of them)
        net_v_spacing = goal_depth / 6  # More horizontal lines
        for i in range(1, 6):
            y = goal_post_y + i * net_v_spacing
            painter.drawLine(QPointF(goal_post_x, y), QPointF(goal_post_x + goal_post_width, y))

        # Add diagonal net pattern for more realism
        painter.setPen(QPen(QColor(200, 200, 200, 80), 1))  # Even lighter for diagonals
        for i in range(0, int(goal_post_width), 15):
            painter.drawLine(QPointF(goal_post_x + i, goal_post_y),
                             QPointF(goal_post_x + i + goal_depth, goal_post_y + goal_depth))

        # Reset pen for other drawings
        painter.setPen(QPen(Qt.white, 2))

    def draw_shot(self, painter):
        """Draw the shot with start (dashed circle) and end (solid circle) positions."""
        shot = self.shot_annotations[0]
        xg_val = shot['xG']

        # Get positions in widget coordinates (using transformed coordinates)
        start_pos = self.field_to_widget(shot['start_pitch']['x'], shot['start_pitch']['y'])
        end_pos = self.field_to_widget(shot['end_pitch']['x'], shot['end_pitch']['y'])

        # Get color and size based on xG
        color = self.get_xg_color(xg_val)
        size = self.get_circle_size(xg_val)

        # Draw arrow from start to end
        painter.setPen(QPen(color, 3))
        painter.drawLine(start_pos, end_pos)

        # Draw arrowhead at end position
        self.draw_arrowhead(painter, start_pos, end_pos, color)

        # Draw start position (dashed circle)
        painter.setPen(QPen(color, 2, Qt.DashLine))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(start_pos, size, size)

        # Draw end position (solid circle)
        painter.setPen(QPen(color, 2, Qt.SolidLine))
        painter.setBrush(color)
        painter.drawEllipse(end_pos, size, size)

    def draw_arrowhead(self, painter, start, end, color):
        """Draw arrowhead at the end of the shot line."""
        # Calculate angle of the line
        dx = end.x() - start.x()
        dy = end.y() - start.y()
        angle = math.atan2(dy, dx)

        # Arrowhead size
        head_length = 10
        head_angle = math.pi / 6  # 30 degrees

        # Calculate arrowhead points
        x1 = end.x() - head_length * math.cos(angle - head_angle)
        y1 = end.y() - head_length * math.sin(angle - head_angle)
        x2 = end.x() - head_length * math.cos(angle + head_angle)
        y2 = end.y() - head_length * math.sin(angle + head_angle)

        # Draw arrowhead
        painter.setPen(QPen(color, 1))
        painter.setBrush(color)
        arrowhead = QPolygonF([end, QPointF(x1, y1), QPointF(x2, y2)])
        painter.drawPolygon(arrowhead)

    def draw_hover_tooltip(self, painter):
        """Draw tooltip when hovering over shot positions."""
        if self.hover_pos and self.hover_text:
            painter.setPen(QPen(Qt.black, 1))
            painter.setBrush(QColor(255, 255, 255, 230))

            # Calculate tooltip size
            fm = painter.fontMetrics()
            text_width = fm.width(self.hover_text)
            text_height = fm.height()

            # Position tooltip
            tooltip_x = self.hover_pos.x() + 10
            tooltip_y = self.hover_pos.y() - 25

            # Draw tooltip background
            tooltip_rect = QRectF(tooltip_x, tooltip_y - text_height,
                                  text_width + 10, text_height + 5)
            painter.drawRoundedRect(tooltip_rect, 2, 3)

            # Draw tooltip text
            painter.setPen(Qt.black)
            painter.drawText(tooltip_x + 5, tooltip_y - 5, self.hover_text)


class ShotTransitionDialog(QDialog):
    def __init__(self, frame_pixmaps, pitch_positions, parent=None, retried=False):
        """
        A dialog showing an animated transition of a shot with synchronized views.

        Parameters:
        -----------
        frame_pixmaps: list of QPixmaps
            Key frames showing the shot progression (start, intermediates, end)
        pitch_positions: list of (x, y) tuples
            Ball positions in meters corresponding to each frame
        parent: QWidget, optional
            Parent widget
        retried: bool, optional
            Whether the segment was retried with a fallback homography (H)
        """
        super().__init__(parent)
        self.setWindowTitle("Shot Attempt Transition")
        self.setFixedSize(1300, 750)
        if parent and hasattr(parent, 'windowIcon'):
            self.setWindowIcon(parent.windowIcon())

        # Initialize data
        self.frame_pixmaps = frame_pixmaps
        self.original_positions = pitch_positions  # Store original positions in meters
        self.num_frames = len(self.frame_pixmaps)
        self.animation_progress = 0.0  # 0.0=start, 1.0=end
        self.animation_duration = 7000  # ms
        self.animation_timer = QTimer(self)
        self.animation_timer.setInterval(20)
        self.animation_timer.timeout.connect(self.update_animation)
        self.user_dragging_slider = False
        self.animation_start_time = None
        self.is_playing = True  # Track if animation is playing

        self.pitch_wid = 8000
        self.pitch_len = 0
        self.determine_pitch_side_and_transform()
        self.goal_depth = 200  # 2m

        self.center_circle_r = 915  # 9.15 m
        self.goal_wid = 732  # 7.32 m
        self.goal_dep = 200  # 2 m
        self.six_w = 1832  # 18.32 m
        self.six_d = 550  # 5.5 m
        self.eighteen_w = 4032  # 40.32 m
        self.eighteen_d = 1650  # 16.5 m
        self.penalty_dist = 1200  # 11 m

        self.margin = 20
        self.gap = 15

        # Setup the UI components
        self.setup_ui()

        # Start the animation
        self.animation_timer.start()
        self.animation_start_time = time.time()

        # Add warning banner if retried
        if getattr(self, 'retried', False):
            warning = QLabel("⚠️ This segment used a fallback homography (H)")
            warning.setStyleSheet("color: red; font-weight: bold; padding: 4px;")
            self.layout().addWidget(warning)

    def setup_ui(self):
        """Create and arrange all UI components"""
        main_layout = QVBoxLayout(self)

        # Top section: Content area with pitch and video
        content_layout = QHBoxLayout()

        # Left: pitch + animation
        self.pitch_widget = QWidget(self)
        self.pitch_widget.setMinimumSize(500, 600)
        self.pitch_widget.setMaximumWidth(600)
        self.pitch_widget.paintEvent = self.paint_pitch_event
        self.pitch_widget.setMouseTracking(True)
        content_layout.addWidget(self.pitch_widget, stretch=1)

        # Right: frame display with larger size and frame counter
        frame_container = QVBoxLayout()

        # Only keep frame counter, remove "Shot Analysis" title
        frame_counter_layout = QHBoxLayout()
        frame_counter_layout.setContentsMargins(0, 0, 0, 1)

        self.frame_counter = QLabel("1/5")
        self.frame_counter.setAlignment(Qt.AlignRight)
        self.frame_counter.setStyleSheet("font-size: 10 pt; color: #6c757d;")
        frame_counter_layout.addStretch()  # Push counter to the right
        frame_counter_layout.addWidget(self.frame_counter)
        frame_container.addLayout(frame_counter_layout)

        self.overlay_label = QLabel(self)
        self.overlay_label.setAlignment(Qt.AlignCenter)
        self.overlay_label.setFixedSize(800, 540)
        self.overlay_label.setStyleSheet("""
            border: 2px solid #dee2e6;
            border-radius: 5px;
            background-color: #000;
        """)
        frame_container.addWidget(self.overlay_label, stretch=2)

        content_layout.addLayout(frame_container, stretch=2)
        main_layout.addLayout(content_layout)

        # Middle section: Controls and timeline
        controls_layout = QHBoxLayout()

        # Play/Pause button
        self.play_button = QPushButton(self)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.play_button.setToolTip("Pause animation")
        self.play_button.setFixedSize(45, 45)
        self.play_button.setStyleSheet("""
            QPushButton {
                background: #007bff;
                border: none;
                border-radius: 22px;
                color: white;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton:pressed {
                background: #004085;
            }
        """)
        self.play_button.clicked.connect(self.toggle_play_pause)
        controls_layout.addWidget(self.play_button)

        # Timeline slider with time markers
        slider_layout = QVBoxLayout()

        # Slider itself
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.setValue(0)
        self.slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: #dee2e6;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #007bff;
                border: 2px solid #0056b3;
                width: 20px;
                margin: -7px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: #007bff;
                border-radius: 4px;
            }
        """)

        # Connect slider signals
        self.slider.sliderPressed.connect(self.on_slider_pressed)
        self.slider.sliderReleased.connect(self.on_slider_released)
        self.slider.valueChanged.connect(self.slider_moved)
        slider_layout.addWidget(self.slider)

        # Time markers
        time_markers = QHBoxLayout()
        start_label = QLabel("Start")
        start_label.setStyleSheet("color: #6c757d; font-size: 10pt;")
        time_markers.addWidget(start_label)
        time_markers.addStretch()
        end_label = QLabel("End")
        end_label.setStyleSheet("color: #6c757d; font-size: 10pt;")
        time_markers.addWidget(end_label)
        slider_layout.addLayout(time_markers)

        controls_layout.addLayout(slider_layout)

        # Reset button
        self.reset_button = QPushButton(self)
        self.reset_button.setIcon(self.style().standardIcon(QStyle.SP_MediaSkipBackward))
        self.reset_button.setToolTip("Reset to start")
        self.reset_button.setFixedSize(45, 45)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background: #6c757d;
                border: none;
                border-radius: 22px;
                color: white;
            }
            QPushButton:hover {
                background: #5a6268;
            }
            QPushButton:pressed {
                background: #495057;
            }
        """)
        self.reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(self.reset_button)

        main_layout.addLayout(controls_layout)

        # Bottom section: Shot info and buttons
        info_layout = QHBoxLayout()

        # Shot information
        shot_info = QLabel("Position coordinates shown in pitch view. Drag the slider to analyze specific moments.")
        shot_info.setStyleSheet("color: #6c757d; font-style: italic; font-size: 11pt;")
        info_layout.addWidget(shot_info)

        info_layout.addStretch()

        # OK button
        self.ok_btn = QPushButton("Close")
        self.ok_btn.setStyleSheet("""
            QPushButton {
                font-size: 12pt;
                font-weight: bold;
                background: #007bff;
                color: white;
                border-radius: 8px;
                padding: 12px 24px;
                border: none;
                min-width: 120px;
            }
            QPushButton:hover {
                background: #0056b3;
            }
            QPushButton:pressed {
                background: #004085;
            }
        """)
        self.ok_btn.clicked.connect(self.accept)
        info_layout.addWidget(self.ok_btn)

        main_layout.addLayout(info_layout)

    def toggle_play_pause(self):
        """Toggle between playing and paused states"""
        if self.is_playing:
            # Pause the animation
            self.animation_timer.stop()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_button.setToolTip("Play animation")
            self.is_playing = False
        else:
            # Resume animation from current position
            if self.animation_progress >= 1.0:
                # If at the end, start over
                self.reset_animation()
            else:
                self.animation_start_time = time.time() - self.animation_progress * self.animation_duration / 1000.0
                self.animation_timer.start()
                self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
                self.play_button.setToolTip("Pause animation")
                self.is_playing = True

    def reset_animation(self):
        """Reset animation to the beginning"""
        self.animation_progress = 0.0
        self.animation_start_time = time.time()
        self.slider.setValue(0)
        self.update_pitch_and_overlay()
        # Start playing if it was stopped
        if not self.is_playing:
            self.toggle_play_pause()

    def update_overlay(self):
        """Update the video overlay based on current progress"""
        idx = self.animation_progress * (self.num_frames - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, self.num_frames - 1)
        frac = idx - i0

        # Use ease-in/ease-out for smoother transition
        frac = 0.5 - 0.5 * math.cos(math.pi * frac) if i0 != i1 else 0

        # Create blended transition between frames
        pix0 = self.frame_pixmaps[i0].scaled(800, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pix1 = self.frame_pixmaps[i1].scaled(800, 540, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        img0 = pix0.toImage().convertToFormat(QImage.Format_ARGB32)
        img1 = pix1.toImage().convertToFormat(QImage.Format_ARGB32)

        blended = QImage(img0.size(), QImage.Format_ARGB32)
        painter = QPainter(blended)

        # Draw base image
        painter.drawImage(0, 0, img0)

        # Cross-fade with second image
        painter.setOpacity(frac)
        painter.drawImage(0, 0, img1)

        # Add subtle overlay to reduce flicker and improve contrast
        painter.setOpacity(0.25)
        painter.fillRect(blended.rect(), Qt.black)

        # Add frame indicator at bottom
        painter.setOpacity(1.0)
        painter.setPen(QPen(QColor(220, 220, 220), 2))
        painter.setBrush(QBrush(QColor(0, 0, 0, 180)))

        # Draw mini progress indicators at bottom
        bar_height = 6
        total_width = blended.width() - 40
        segment_width = total_width / (self.num_frames - 1)

        painter.drawRect(QRectF(20, blended.height() - bar_height - 15, total_width, bar_height))

        # Fill in progress bar
        progress_width = total_width * self.animation_progress
        painter.setBrush(QBrush(QColor(76, 175, 80, 220)))
        painter.drawRect(QRectF(20, blended.height() - bar_height - 15, progress_width, bar_height))

        # Draw markers for each keyframe
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        for i in range(self.num_frames):
            x = 20 + i * segment_width
            painter.drawLine(QPointF(x, blended.height() - bar_height - 18),
                             QPointF(x, blended.height() - bar_height - 12))

        painter.end()

        # Update the display
        self.overlay_label.setPixmap(QPixmap.fromImage(blended))

    def on_slider_pressed(self):
        """Handle slider press event"""
        self.user_dragging_slider = True
        if self.animation_timer.isActive():
            self.animation_timer.stop()

    def on_slider_released(self):
        """Handle slider release event"""
        self.user_dragging_slider = False
        # Resume autoplay from the new position if it was playing before
        t = self.slider.value() / 1000.0
        self.animation_progress = t
        self.animation_start_time = time.time() - t * self.animation_duration / 1000.0

        if self.is_playing and not self.animation_timer.isActive():
            self.animation_timer.start()

    def set_animation_progress(self, progress):
        """Set animation progress and update all related UI elements"""
        self.animation_progress = max(0.0, min(1.0, progress))

        # Update slider without triggering signals
        self.slider.blockSignals(True)
        self.slider.setValue(int(self.animation_progress * 1000))
        self.slider.blockSignals(False)

        # Update displays
        self.update_pitch_and_overlay()

        # Update frame counter
        current_frame = int(self.animation_progress * (self.num_frames - 1)) + 1
        self.frame_counter.setText(f"Frame: {current_frame}/{self.num_frames}")

    # Update the existing slider_moved method to use the new helper
    def slider_moved(self, value):
        """Handle slider movement"""
        if self.user_dragging_slider:  # Only respond when user is actively dragging
            t = value / 1000.0
            self.set_animation_progress(t)

    def update_pitch_and_overlay(self):
        """Update both pitch and overlay based on current progress"""
        self.pitch_widget.update()
        self.update_overlay()

        # Update frame counter
        current_frame = int(self.animation_progress * (self.num_frames - 1)) + 1
        self.frame_counter.setText(f"Frame: {current_frame}/{self.num_frames}")

    def update_animation(self):
        """Update animation frame based on elapsed time"""
        elapsed = (time.time() - self.animation_start_time) * 1000  # ms
        t = min(1.0, elapsed / self.animation_duration)
        self.animation_progress = t

        # Update UI
        self.slider.blockSignals(True)
        self.slider.setValue(int(t * 1000))
        self.slider.blockSignals(False)
        self.update_pitch_and_overlay()

        # Handle animation completion
        if t >= 1.0:
            self.animation_timer.stop()
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
            self.play_button.setToolTip("Play animation")
            self.is_playing = False

    def compute_layout(self):
        """Calculate the half-pitch layout - same approach as XGShotPlotDialog"""
        # Calculate available space for pitch
        available_width = self.pitch_widget.width() - 40  # Margins
        available_height = self.pitch_widget.height() - 80  # Margins + coordinate text space

        # Pitch dimensions with proper aspect ratio
        target_ratio = self.pitch_wid / self.pitch_len  # 80/60 = 1.33

        ph = available_height - 40  # Leave some margin
        pw = ph * target_ratio

        if pw > available_width - 40:
            pw = available_width - 40
            ph = pw / target_ratio

        x0 = self.margin + (available_width - pw) / 2
        y0 = self.margin + (available_height - ph) / 2

        return QRectF(x0, y0, pw, ph)

    def determine_pitch_side_and_transform(self):
        """Determine which half-pitch to show and transform coordinates."""
        if len(self.original_positions) < 2:
            # Default transformation if not enough positions
            self.transformed_positions = [(x * 100, y * 100) for x, y in self.original_positions]
            self.target_side = "left"
            self.pitch_x0 = 0  # 0m (goal line)
            self.pitch_x1 = 6000  # 60m (halfway line)
            self.pitch_len = self.pitch_x1 - self.pitch_x0
            return

        start_pos = self.original_positions[0]
        end_pos = self.original_positions[-1]

        sx, sy = start_pos[0], start_pos[1]
        ex, ey = end_pos[0], end_pos[1]

        # Determine which goal is being targeted
        x_direction = ex - sx

        # Determine target goal based on trajectory and position
        if x_direction < 0 or (x_direction == 0 and sx < 60):  # Targeting left goal
            target_goal_x = 0
            self.target_side = "left"
            # Set pitch bounds to show left half (0-60m)
            self.pitch_x0 = 0  # 0m (goal line)
            self.pitch_x1 = 6000  # 60m (halfway line)
        else:  # Targeting right goal
            target_goal_x = 120
            self.target_side = "right"
            # Set pitch bounds to show right half (60-120m)
            self.pitch_x0 = 6000  # 60m (halfway line)
            self.pitch_x1 = 12000  # 120m (goal line)

        self.pitch_len = self.pitch_x1 - self.pitch_x0
        # print(f"[DEBUG] Target goal at x={target_goal_x}, side: {self.target_side}")
        # print(f"[DEBUG] Pitch bounds: {self.pitch_x0 / 100}m to {self.pitch_x1 / 100}m")

        # Transform all positions consistently
        self.transformed_positions = []

        # # In determine_pitch_side_and_transform():
        for x, y in self.original_positions:
            if target_goal_x == 0:  # Left goal
                transformed_x, transformed_y = x, y
            else:  # Right goal - DON'T flip, just use original coordinates
                transformed_x, transformed_y = x, y  # Keep original coordinates!

            # Convert to cm
            self.transformed_positions.append((transformed_x * 100, transformed_y * 100))

        # print(f"[DEBUG] Transformed {len(self.transformed_positions)} positions")
        # print(f"[DEBUG] First position: {self.original_positions[0]} -> {self.transformed_positions[0]}")
        # print(f"[DEBUG] Last position: {self.original_positions[-1]} -> {self.transformed_positions[-1]}")

    def field_to_widget(self, field_x, field_y, rect):
        """Convert field coordinates (cm) to widget coordinates."""

        # Convert cm back to meters for clearer debugging
        field_x_m = field_x / 100.0
        field_y_m = field_y / 100.0

        # print(f"[DEBUG] field_to_widget input: x={field_x_m:.1f}m, y={field_y_m:.1f}m")
        # print(f"[DEBUG] Pitch bounds: {self.pitch_x0 / 100:.1f}m to {self.pitch_x1 / 100:.1f}m")

        # Clamp field coordinates to pitch bounds
        field_x = max(self.pitch_x0, min(self.pitch_x1, field_x))
        field_y = max(0, min(self.pitch_wid, field_y))

        # For left goal (target_side == "left"):
        # - pitch_x0 = 0 (goal line, should be at TOP of widget)
        # - pitch_x1 = 6000 (halfway line, should be at BOTTOM of widget)
        # - field_x closer to 0 should be at top (smaller widget_y)
        # - field_x closer to 6000 should be at bottom (larger widget_y)

        if self.target_side == "left":
            # Map field_x (0-6000cm) to widget_y (top to bottom)
            # field_x=0 (goal) -> widget_y=rect.top (top of pitch)
            # field_x=6000 (halfway) -> widget_y=rect.bottom (bottom of pitch)
            normalized_x = (field_x - self.pitch_x0) / (self.pitch_x1 - self.pitch_x0)
            widget_y = rect.y() + (normalized_x * rect.height())  # DON'T flip Y
        else:
            # For right goal, the mapping should be similar but we're showing 60-120m
            # field_x=6000 (halfway) -> widget_y=rect.bottom
            # field_x=12000 (goal) -> widget_y=rect.top
            normalized_x = (field_x - self.pitch_x0) / (self.pitch_x1 - self.pitch_x0)
            widget_y = rect.y() + rect.height() - (normalized_x * rect.height())  # Flip for right side

        normalized_y = 1.0 - (field_y / self.pitch_wid)  # Flip the Y coordinate
        widget_x = rect.x() + (normalized_y * rect.width())

        # print(f"[DEBUG] field_to_widget output: widget_x={widget_x:.1f}, widget_y={widget_y:.1f}")
        # print(f"[DEBUG] normalized_x={normalized_x:.3f}, normalized_y={normalized_y:.3f}")
        # print(f"[DEBUG] rect bounds: x={rect.x():.1f}, y={rect.y():.1f}, w={rect.width():.1f}, h={rect.height():.1f}")

        return widget_x, widget_y

    def paint_pitch_event(self, event):
        """Paint the half-pitch with shot path and ball position - using XGShotPlotDialog approach"""
        painter = QPainter(self.pitch_widget)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.pitch_widget.rect(), QColor(34, 139, 34))  # Forest green

        # Get pitch rectangle
        rect = self.compute_layout()

        # Draw pitch with same approach as XGShotPlotDialog
        self.draw_pitch(painter, rect)
        self.draw_shot_trajectory(painter, rect)
        self.draw_current_ball_position(painter, rect)
        self.draw_position_text(painter, rect)

    def draw_pitch(self, painter, rect):
        """Draw the half pitch with all markings - same as XGShotPlotDialog"""
        # Set up pen for field lines
        pen = QPen(Qt.white, 2)
        painter.setPen(pen)

        # Draw pitch boundary
        painter.drawRect(rect)

        # Goal area (6-yard box)
        goal_width = (18.32 / 80.0) * rect.width()  # 18.32m in 80m width
        goal_depth = (5.5 / 60.0) * rect.height()  # 5.5m in 60m length
        goal_x = rect.x() + (rect.width() - goal_width) / 2
        goal_y = rect.y()
        painter.drawRect(QRectF(goal_x, goal_y, goal_width, goal_depth))

        # Penalty area (18-yard box)
        penalty_width = (40.32 / 80.0) * rect.width()  # 40.32m in 80m width
        penalty_depth = (16.5 / 60.0) * rect.height()  # 16.5m in 60m length
        penalty_x = rect.x() + (rect.width() - penalty_width) / 2
        penalty_y = rect.y()
        penalty_box = QRectF(penalty_x, penalty_y, penalty_width, penalty_depth)
        painter.drawRect(penalty_box)

        # Penalty spot
        penalty_spot_y = rect.y() + (11.0 / 60.0) * rect.height()  # 11m from goal
        penalty_spot_x = rect.x() + rect.width() / 2
        painter.drawEllipse(QPointF(penalty_spot_x, penalty_spot_y), 3, 3)

        # Improved Penalty arc - same as XGShotPlotDialog
        arc_radius = (9.15 / 60.0) * rect.height()

        arc_center_x = penalty_spot_x
        arc_center_y = penalty_spot_y

        penalty_box_edge = penalty_y + penalty_depth

        if arc_center_y + arc_radius > penalty_box_edge:
            dy = penalty_box_edge - arc_center_y
            if abs(dy) < arc_radius:
                cos_angle = dy / arc_radius
                cos_angle = max(-1.0, min(1.0, cos_angle))
                angle_rad = math.acos(cos_angle)
                angle_deg = math.degrees(angle_rad)

                arc_rect = QRectF(arc_center_x - arc_radius, arc_center_y - arc_radius,
                                  arc_radius * 2, arc_radius * 2)

                start_angle = int((270 - angle_deg) * 16)
                span_angle = int(2 * angle_deg * 16)

                painter.drawArc(arc_rect, start_angle, span_angle)

        # Center circle (half) - at bottom edge
        center_y = rect.y() + rect.height()
        center_x = rect.x() + rect.width() / 2
        circle_radius = (9.15 / 60.0) * rect.height()

        circle_rect = QRectF(center_x - circle_radius, center_y - circle_radius,
                             circle_radius * 2, circle_radius * 2)
        painter.drawArc(circle_rect, 0, 180 * 16)

        # Center spot at halfway line
        painter.drawEllipse(QPointF(center_x, center_y), 3, 3)

        # Enhanced GOAL STRUCTURE - same as XGShotPlotDialog
        goal_post_width = (8.00 / 80.0) * rect.width()
        goal_post_x = rect.x() + (rect.width() - goal_post_width) / 2
        goal_post_y = rect.y() - 25
        goal_depth = 20

        # Goal net area
        net_color = QColor(60, 120, 60, 200)
        painter.fillRect(QRectF(goal_post_x - 5, goal_post_y, goal_post_width + 10, goal_depth), net_color)

        # Goal crossbar
        crossbar_height = 6
        crossbar_color = QColor(240, 240, 240)
        painter.fillRect(QRectF(goal_post_x - 3, goal_post_y + goal_depth - crossbar_height / 2,
                                goal_post_width + 6, crossbar_height), crossbar_color)

        # Goal posts with enhanced 3D effect
        post_width = 6
        post_height = goal_depth + crossbar_height / 2

        shadow_color = QColor(180, 180, 180)
        post_color = QColor(250, 250, 250)

        # Left post shadow and main
        painter.fillRect(QRectF(goal_post_x - post_width / 2 - 2, goal_post_y + 2,
                                post_width, post_height), shadow_color)
        painter.fillRect(QRectF(goal_post_x - post_width / 2, goal_post_y,
                                post_width, post_height), post_color)

        # Right post shadow and main
        painter.fillRect(QRectF(goal_post_x + goal_post_width - post_width / 2 + 2, goal_post_y + 2,
                                post_width, post_height), shadow_color)
        painter.fillRect(QRectF(goal_post_x + goal_post_width - post_width / 2, goal_post_y,
                                post_width, post_height), post_color)

        # Enhanced goal line
        goal_line_thickness = 4
        painter.fillRect(QRectF(goal_post_x - post_width, rect.y() - goal_line_thickness / 2,
                                goal_post_width + 2 * post_width, goal_line_thickness), Qt.white)

        # Realistic net pattern
        painter.setPen(QPen(QColor(200, 200, 200, 150), 1))

        # Vertical net lines
        net_spacing = goal_post_width / 12
        for i in range(1, 12):
            x = goal_post_x + i * net_spacing
            painter.drawLine(QPointF(x, goal_post_y), QPointF(x, goal_post_y + goal_depth))

        # Horizontal net lines
        net_v_spacing = goal_depth / 6
        for i in range(1, 6):
            y = goal_post_y + i * net_v_spacing
            painter.drawLine(QPointF(goal_post_x, y), QPointF(goal_post_x + goal_post_width, y))

        # Diagonal net pattern
        painter.setPen(QPen(QColor(200, 200, 200, 80), 1))
        for i in range(0, int(goal_post_width), 15):
            painter.drawLine(QPointF(goal_post_x + i, goal_post_y),
                             QPointF(goal_post_x + i + goal_depth, goal_post_y + goal_depth))

        # Reset pen
        painter.setPen(QPen(Qt.white, 2))

    def draw_shot_trajectory(self, painter, rect):
        """Draw the complete shot trajectory with markers"""
        if len(self.transformed_positions) < 2:
            return

        # Draw trajectory path (dotted line)
        painter.setPen(QPen(QColor(255, 255, 255, 180), 3, Qt.DashLine))
        path = QPainterPath()

        # Use all transformed positions - don't filter them out
        valid_positions = self.transformed_positions

        # Debug: Print positions to check if they exist
        print(f"[DEBUG] Drawing trajectory with {len(valid_positions)} positions")
        if valid_positions:
            print(f"[DEBUG] First pos: {valid_positions[0]}, Last pos: {valid_positions[-1]}")

        # Create smooth path
        for i, (x_cm, y_cm) in enumerate(valid_positions):
            px, py = self.field_to_widget(x_cm, y_cm, rect)
            if i == 0:
                path.moveTo(px + 2, py + 2)
            else:
                path.lineTo(px + 2, py + 2)

        painter.drawPath(path)

        # Draw position markers
        for i, (x_cm, y_cm) in enumerate(valid_positions):
            px, py = self.field_to_widget(x_cm, y_cm, rect)

            if i == 0:  # Starting position
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 193, 7)))  # Gold
                painter.drawEllipse(QPointF(px, py), 8, 8)


            elif i == len(valid_positions) - 1:  # End position
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(220, 53, 69)))  # Red
                painter.drawEllipse(QPointF(px, py), 8, 8)


            else:  # Intermediate positions
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(255, 255, 255, 120)))
                painter.drawEllipse(QPointF(px, py), 4, 4)

    def draw_current_ball_position(self, painter, rect):
        """Draw the current animated ball position"""
        if not self.transformed_positions:
            return

        # Interpolate current position with smooth easing
        t = self.animation_progress

        # Apply smooth easing function
        eased_t = 0.5 - 0.5 * math.cos(math.pi * t)

        # Use all transformed positions - same as trajectory
        valid_positions = self.transformed_positions

        if len(valid_positions) < 2:
            return

        # Interpolate position
        idx = eased_t * (len(valid_positions) - 1)
        i0 = int(np.floor(idx))
        i1 = min(i0 + 1, len(valid_positions) - 1)
        frac = idx - i0

        sx_cm, sy_cm = valid_positions[i0]
        ex_cm, ey_cm = valid_positions[i1]
        bx_cm = sx_cm + (ex_cm - sx_cm) * frac
        by_cm = sy_cm + (ey_cm - sy_cm) * frac

        # Convert to widget coordinates
        bx, by = self.field_to_widget(bx_cm, by_cm, rect)

        # Draw ball shadow
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 60)))
        painter.drawEllipse(QPointF(bx + 2, by + 4), 10, 5)

        # Draw ball with gradient
        radial = QRadialGradient(bx - 2, by - 2, 12)
        radial.setColorAt(0, QColor(255, 220, 100))
        radial.setColorAt(0.7, QColor(255, 180, 0))
        radial.setColorAt(1, QColor(180, 120, 0))
        painter.setBrush(QBrush(radial))
        painter.drawEllipse(QPointF(bx, by), 12, 12)

        # Ball highlight
        painter.setBrush(QBrush(QColor(255, 255, 255, 150)))
        painter.drawEllipse(QPointF(bx - 4, by - 4), 4, 4)

        # Store current position for text display
        self.current_ball_pos = (bx_cm / 100, by_cm / 100)  # Convert back to meters

    def draw_position_text(self, painter, rect):
        """Draw current ball position text with enhanced formatting"""
        if not hasattr(self, 'current_ball_pos') or not self.original_positions:
            return

        # Get interpolated original coordinates for display
        progress_idx = self.animation_progress * (len(self.original_positions) - 1)
        i0 = int(np.floor(progress_idx))
        i1 = min(i0 + 1, len(self.original_positions) - 1)
        frac = progress_idx - i0

        if i0 < len(self.original_positions) and i1 < len(self.original_positions):
            orig_x0, orig_y0 = self.original_positions[i0]
            orig_x1, orig_y1 = self.original_positions[i1]

            current_orig_x = orig_x0 + (orig_x1 - orig_x0) * frac
            current_orig_y = orig_y0 + (orig_y1 - orig_y0) * frac

            # Create text background
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(QColor(0, 0, 0, 160)))
            text_rect = QRectF(rect.x() + 10, rect.y() + rect.height() + 5, 250, 45)
            painter.drawRoundedRect(text_rect, 5, 5)

            # Draw position text
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            painter.setFont(QFont("Arial", 10, QFont.Bold))

            position_text = f"Position: ({current_orig_x:.1f}m, {current_orig_y:.1f}m)"
            painter.drawText(text_rect.adjusted(10, 5, -10, -25), Qt.AlignLeft, position_text)

            # Draw progress indicator
            painter.setFont(QFont("Arial", 9))
            progress_text = f"Progress: {self.animation_progress * 100:.1f}% | Frame: {int(self.animation_progress * (self.num_frames - 1)) + 1}/{self.num_frames}"
            painter.drawText(text_rect.adjusted(10, 20, -10, -5), Qt.AlignLeft, progress_text)


# -------------------------------------
# ProcessingThread (action recognition)
# -------------------------------------
class ActionRecognitionWorker(QThread):
    progress_update = pyqtSignal(int, str)
    complete = pyqtSignal(list)
    error = pyqtSignal(str)
    segment_detected = pyqtSignal(tuple)
    paused = pyqtSignal()
    resumed = pyqtSignal()

    def __init__(self, video_path, model_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.model_path = model_path
        self.window_size = 8
        self.stride = 2
        self.confidence_threshold = 0.90
        self.motion_threshold = 0.5
        self.max_frame_skip = 30
        self.min_frame_skip = 15
        self._pause_lock = QMutex()
        self._pause_condition = QWaitCondition()
        self._should_pause = False
        self.current_state = None

    def pause(self):
        with QMutexLocker(self._pause_lock):
            self._should_pause = True

    def resume(self):
        with QMutexLocker(self._pause_lock):
            self._should_pause = False
            self._pause_condition.wakeAll()

    def run(self):
        try:
            result = None
            while True:
                result = inference_on_video(
                    self.video_path,
                    self.model_path,
                    window_size=self.window_size,
                    stride=self.stride,
                    max_frame_skip=self.max_frame_skip,
                    min_frame_skip=self.min_frame_skip,
                    confidence_threshold=self.confidence_threshold,
                    motion_threshold=self.motion_threshold,
                    progress_callback=self.safe_progress_callback,
                    segment_callback=self.handle_segment,
                    abort_callback=self.isInterruptionRequested,
                    pause_callback=lambda: self._should_pause,
                    state=self.current_state
                )

                # Handle pause request
                with QMutexLocker(self._pause_lock):
                    if self._should_pause:
                        self.paused.emit()
                        self._pause_condition.wait(self._pause_lock)
                        self.resumed.emit()

                # Check for completion or errors
                if result.get('completed', False):
                    self.handle_completion(result)
                    break

                if result.get('error'):
                    self.error.emit(result['error'])
                    break

                # Save state for next iteration
                self.current_state = result.get('state')

        except Exception as e:
            self.handle_exception(e)
            self.error.emit(f"Critical error: {traceback.format_exc()}")

    def safe_progress_callback(self, progress, message):
        if self.isInterruptionRequested():
            raise RuntimeError("Processing aborted")
        self.progress_update.emit(progress, message)

    def handle_segment(self, segment):
        if not self.isInterruptionRequested():
            self.segment_detected.emit(segment)

    def handle_completion(self, result):
        ontarget_segments = result['actions'].get('ontarget', [])
        self.progress_update.emit(100, "Processing complete")
        self.complete.emit(ontarget_segments)

    def handle_exception(self, error):
        if self.isInterruptionRequested():
            self.error.emit("Processing aborted by user")
        else:
            self.error.emit(f"Processing error: {str(error)}")




# ---------------------------
# Segment states and VideoSegment data class
# ---------------------------
class SegmentState(Enum):
    PENDING = 'pending'
    CONFIRMED = 'confirmed'
    REJECTED = 'rejected'
    BEING_ADJUSTED = 'adjusting'
    PREVIEW = 'preview'


class VideoSegment:
    def __init__(self, start, end, state=SegmentState.PENDING):
        self.start = start
        self.end = end
        self.state = state
        self.original_start = start
        self.original_end = end
        self.start_marker = None
        self.end_marker = None
        self.start_annotation = None
        self.end_annotation = None

    @property
    def duration(self):
        return self.end - self.start

# ---------------------------
# VideoPlayer with VLC
# ---------------------------
class VideoPlayer(QWidget):
    timeChanged = pyqtSignal(float)  # Current time in seconds

    def __init__(self, container):
        super().__init__(container)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(1, 1)
        self.instance = None
        self.player = None
        self.media = None
        self.timeline = None
        self.duration = 0
        self.current_segment = None
        self.is_previewing = False
        self.current_rate = 1.0
        self.muted = False
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.playbackTimer = QTimer(self)
        # self.playbackTimer.setInterval(200)  # check every 200ms
        # self.playbackTimer.timeout.connect(self.check_playback_time)

    def set_timeline(self, timeline):
        self.timeline = timeline

    def _initialize_vlc(self):
        try:
            print("[DEBUG] _initialize_vlc called.")
            # Clean up existing instances if any
            if hasattr(self, 'player') and self.player:
                try:
                    print("[DEBUG] _initialize_vlc: Stopping and releasing existing player.")
                    self.player.stop()
                    self.player.release()
                except Exception as e:
                    print(f"[DEBUG] _initialize_vlc: Error releasing player: {e}")
                self.player = None

            if hasattr(self, 'instance') and self.instance:
                try:
                    print("[DEBUG] _initialize_vlc: Releasing existing instance.")
                    self.instance.release()
                except Exception as e:
                    print(f"[DEBUG] _initialize_vlc: Error releasing instance: {e}")
                self.instance = None

            # Add more compatibility flags for better stability
            self.instance = vlc.Instance('--no-xlib --no-audio --avcodec-hw=none --no-disable-screensaver --quiet')
            self.player = self.instance.media_player_new()

            # Add error callback
            self.player.event_manager().event_attach(
                vlc.EventType.MediaPlayerEncounteredError,
                self._vlc_error_callback
            )

            if sys.platform == "win32":
                self.player.set_hwnd(int(self.winId()))
            else:
                self.player.set_xwindow(int(self.winId()))

            print("[DEBUG] _initialize_vlc: Initialization complete.")
            return True
        except Exception as e:
            print(f"VLC initialization error: {e}")
            return False

    def _vlc_error_callback(self, event):
        print(f"VLC Error: {str(event.u.new_status)}")

    def close_video(self):
        """Clean up VLC resources while preserving instance"""
        try:
            if self.player:
                self.player.stop()
                self.player.release()
                self.player = None
        except Exception as e:
            print(f"VLC cleanup error: {e}")
        finally:
            self.media = None
            self.is_previewing = False

    def load_video(self, file_path):
        try:
            if not self.instance or not self.player:
                self._initialize_vlc()
            self.media = self.instance.media_new(file_path)
            self.player.set_media(self.media)
            self.media.parse()
            self.duration = self.media.get_duration() / 1000
            # self.player.play()
            time.sleep(0.1)
            self.player.pause()
            self.update_video_geometry()
            if hasattr(self.parent(), 'timeline'):
                self.parent().timeline.set_duration(self.duration)
            return True, self.duration
        except Exception as e:
            return False, str(e)

    def update_video_geometry(self):
        if not self.player:
            return
        widget_width = self.width()
        widget_height = self.height()
        self.player.video_set_scale(0.0)
        self.player.video_set_aspect_ratio(None)
        if sys.platform == "win32":
            self.player.set_hwnd(int(self.winId()))
        else:
            self.player.set_xwindow(int(self.winId()))
        self.setGeometry(0, 0, widget_width, widget_height)

    def set_time(self, ms):
        if self.player:
            self.player.set_time(int(ms))
            if self.timeline:
                self.timeline.set_current_time(ms / 1000)
            self.timeChanged.emit(ms / 1000)

    def show_short_segment_warning(self, min_duration):
        dialog = QDialog(self)
        dialog.setWindowTitle("Segment too short")
        dialog.setStyleSheet("""
            QDialog { background-color: blanchedalmond; }
            QLabel { font: 12pt 'Segoe Print'; color: white; }
            QPushButton { background-color: #333; color: white; border-radius: 4px; padding: 5px 15px; }
            QPushButton:hover { background-color: #666; }
        """)
        layout = QVBoxLayout(dialog)
        label = QLabel(
            f"This segment is too short for preview controls to work reliably.<br>"
            f"(Minimum recommended duration: {min_duration} seconds)<br><br>"
            f"<b>Controls (play, pause, rewind, stop, forward) may not be available.</b>"
        )
        label.setWordWrap(True)
        layout.addWidget(label)
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        layout.addWidget(ok_btn, alignment=Qt.AlignCenter)
        dialog.exec_()

    def play_segment(self, segment):
        """Play from segment.start to segment.end using precise time boundaries."""
        MIN_PREVIEW_DURATION = 2.0  # seconds, adjust as needed
        if (segment.end - segment.start) < MIN_PREVIEW_DURATION:
            self.show_short_segment_warning(MIN_PREVIEW_DURATION)

        try:
            self._stop_subclip()  # This destroys the player

            # Re-initialize VLC properly
            if not hasattr(self, 'instance') or not self.instance:
                self._initialize_vlc()
            elif not hasattr(self, 'player') or not self.player:
                self.player = self.instance.media_player_new()
                self.player.event_manager().event_attach(
                    vlc.EventType.MediaPlayerEncounteredError,
                    self._vlc_error_callback
                )
                if sys.platform == "win32":
                    self.player.set_hwnd(int(self.winId()))
                else:
                    self.player.set_xwindow(int(self.winId()))

            # Load media
            self.media = self.instance.media_new(self.window().fileName)
            self.player.set_media(self.media)
            self.player.set_rate(self.current_rate)

            start_ms = int(segment.start * 1000)
            end_ms = int(segment.end * 1000)

            def poll_for_seek():
                current = self.player.get_time()
                print(f"[DEBUG] Polling for seek: current={current}, target={start_ms}")
                if current >= 0 and abs(current - start_ms) < 150:
                    print("[DEBUG] Seek complete, starting playback.")
                    self._start_actual_playback(segment)
                else:
                    QTimer.singleShot(50, poll_for_seek)

            def do_seek():
                print("[DEBUG] Pausing and seeking to start_ms")
                self.player.pause()
                self.player.set_time(start_ms)
                QTimer.singleShot(100, poll_for_seek)

            def start_and_pause():
                print("[DEBUG] Starting playback to initialize VLC internals")
                self.player.play()
                QTimer.singleShot(200, do_seek)

            # Start the sequence
            QTimer.singleShot(200, start_and_pause)

        except Exception as e:
            print(f"Error starting playback: {e}")
            self._stop_subclip()

    def _start_actual_playback(self, segment):
        try:
            if self.timeline:
                self.timeline.start_preview(segment)
            if self.player:
                self.player.play()
                self.is_previewing = True
                self.current_segment = segment
                print(
                    f"[DEBUG] Playback started at {self.player.get_time()} ms (should be close to {int(segment.start * 1000)} ms)")

                # --- Start polling timer for precise stop ---
                self._end_poll_timer = QTimer(self)
                self._end_poll_timer.setInterval(50)  # check every 50ms
                self._end_poll_timer.timeout.connect(lambda: self._check_end_time(segment))
                self._end_poll_timer.start()

        except Exception as e:
            print(f"Error during actual playback start: {e}")
            self._stop_subclip()

    def _check_end_time(self, segment):
        if not self.player or not self.is_previewing:
            return
        current_time = self.player.get_time()
        end_ms = int(segment.end * 1000)
        # Debug print
        print(f"[DEBUG] Polling for end: current={current_time}, end={end_ms}")
        if current_time >= end_ms or current_time == -1:
            print("[DEBUG] End of segment reached, stopping playback.")
            if hasattr(self, '_end_poll_timer') and self._end_poll_timer:
                self._end_poll_timer.stop()
                self._end_poll_timer = None
            self._stop_subclip()

    def _stop_subclip(self):
        try:
            if not getattr(self, 'is_previewing', False):
                print("[DEBUG] _stop_subclip called but not in preview mode.")
                return

            print("[DEBUG] _stop_subclip: Stopping player.")
            if hasattr(self, 'playbackTimer') and self.playbackTimer:
                self.playbackTimer.stop()
            if hasattr(self, '_end_poll_timer') and self._end_poll_timer:
                self._end_poll_timer.stop()
                self._end_poll_timer = None

            # Stop playback if needed
            if hasattr(self, 'player') and self.player:
                if self.player.is_playing():
                    self.player.stop()
                    print("[DEBUG] _stop_subclip: Player stopped.")
                else:
                    print("[DEBUG] _stop_subclip: Player already stopped.")

                # Set to segment end and pause (if current_segment is still set)
                if self.current_segment:
                    self.player.set_time(int(self.current_segment.end * 1000))
                    self.player.pause()

            # Disable controls if present
            if hasattr(self, 'media_panel') and self.media_panel:
                for btn in self.media_panel.findChildren(QPushButton):
                    btn.setEnabled(False)
            if hasattr(self, 'volume_slider') and self.volume_slider:
                self.volume_slider.setEnabled(False)

            self.is_previewing = False
            self.current_segment = None

        except Exception as e:
            print(f"Stop error: {e}")

        finally:
            if hasattr(self, 'timeline') and self.timeline:
                QTimer.singleShot(0, self.timeline.end_preview)

    def stop(self):
        if self.player:
            if self.player.is_playing():
                print("[DEBUG] stop(): Stopping player.")
                self.player.stop()
            else:
                print("[DEBUG] stop(): Player already stopped.")
        if hasattr(self, 'playbackTimer') and self.playbackTimer:
            self.playbackTimer.stop()
        self.is_previewing = False
        self.current_segment = None

    def check_position(self, end_ms):
        """Modified position checker"""
        if not self.player or not self.is_previewing:
            return

        current = self.player.get_time()
        if current == -1:  # VLC returns -1 if position not available
            return

        print(f"Current: {current}ms, End: {end_ms}ms")
        if current >= end_ms:
            self._stop_subclip()
            if self.timeline:
                self.timeline.previewEnded.emit()

    def slow_down(self, rate=0.5):
        if self.player:
            self.set_rate(rate)

    def restore_speed(self):
        if self.player:
            self.set_rate(1.0)

    def pause(self):
        if self.player:
            self.player.pause()
            if self.is_previewing and hasattr(self, '_end_poll_timer') and self._end_poll_timer:
                self._end_poll_timer.stop()
            # Set segment state to PENDING to allow adjustment
            if self.current_segment and self.timeline:
                self.current_segment.state = SegmentState.PENDING
                self.timeline.update_segment_state(self.current_segment, SegmentState.PENDING)

    def play(self):
        if self.player:
            self.player.play()
            if self.is_previewing and hasattr(self, '_end_poll_timer') and self._end_poll_timer:
                self._end_poll_timer.start()

    def check_playback_time(self, end_ms):
        try:
            if not self.player:
                return

            current_time = self.player.get_time()
            if current_time >= end_ms or current_time == -1:
                self._stop_subclip()

        except Exception as e:
            print(f"Playback check error: {e}")
            self._stop_subclip()

    def play_pause_toggle(self):
        if self.player:
            if self.player.is_playing():
                self.player.pause()
            else:
                self.player.play()

    def rewind(self, small_step=True):
        if not self.player:
            return
        current_time = self.player.get_time()
        step = 5000 if small_step else 30000
        new_time = max(0, current_time - step)
        if self.is_previewing and self.current_segment:
            seg_start_ms = self.current_segment.start * 1000
            new_time = max(new_time, seg_start_ms)
        self.player.set_time(new_time)

    def forward(self, small_step=True):
        if not self.player:
            return
        current_time = self.player.get_time()
        step = 5000 if small_step else 30000
        new_time = current_time + step
        if self.is_previewing and self.current_segment:
            seg_end_ms = self.current_segment.end * 1000
            new_time = min(new_time, seg_end_ms)
        self.player.set_time(new_time)

    def get_time(self):
        return self.player.get_time() if self.player else 0

    def set_rate(self, rate):
        if self.player and 0.25 <= rate <= 4.0:
            self.current_rate = rate
            self.player.set_rate(rate)

    def set_volume(self, volume):
        if self.player:
            self.player.audio_set_volume(max(0, min(100, volume)))

    def toggle_mute(self):
        if self.player:
            self.muted = not self.muted
            self.player.audio_toggle_mute()

    def extract_thumbnail(self, time_pos):
        if not self.player:
            return None
        current_pos = self.get_time()
        try:
            self.set_time(int(time_pos * 1000))
            time.sleep(0.2)
            if sys.platform == "win32":
                screen = QApplication.primaryScreen()
                geometry = self.geometry()
                frame = screen.grabWindow(
                    self.winId(),
                    geometry.x(), geometry.y(),
                    geometry.width(), geometry.height()
                )
            else:
                frame = self.grab()
            if not frame or frame.isNull():
                return None
            return frame.scaled(160, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        finally:
            self.set_time(current_pos)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(0, self.update_video_geometry)

    def showEvent(self, event):
        super().showEvent(event)
        self.update_video_geometry()


# ---------------------------
# VideoValidator
# ---------------------------
class VideoValidator:
    @staticmethod
    def compute_laplacian_variance(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def compute_tenengrad(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        return np.mean(magnitude)

    @staticmethod
    def validate(file_path, progress_callback=None):
        try:
            if progress_callback:
                progress_callback(5, "Checking file existence...")
            if not os.path.exists(file_path):
                return False, "File does not exist"
            if progress_callback:
                progress_callback(10, "Validating format...")
            valid_exts = {'.mp4', '.avi', '.mov', '.webm'}
            ext = os.path.splitext(file_path)[1].lower()
            if ext not in valid_exts:
                return False, f"Invalid format. Supported: {', '.join(valid_exts)}"
            if progress_callback:
                progress_callback(15, "Checking file size...")
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            if file_size < 1:
                return False, "File too small (min 1MB)"
            if file_size > 2000:
                return False, "File too large (max 2GB)"
            if progress_callback:
                progress_callback(20, "Opening video file...")
            cap = cv2.VideoCapture(file_path)
            if not cap.isOpened():
                return False, "Could not open video"
            if progress_callback:
                progress_callback(30, "Reading video properties...")
            props = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': int(cap.get(cv2.CAP_PROP_FPS)),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'codec': int(cap.get(cv2.CAP_PROP_FOURCC)),
            }
            if progress_callback:
                progress_callback(50, "Analyzing video content...")
            validations = []

            # Sample 5 frames uniformly across the video
            sample_indices = [int(props['frame_count'] * x) for x in [0.2, 0.4, 0.6, 0.8, 1.0]]
            lap_variances = []
            tenengrad_scores = []
            for idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    lap_var = VideoValidator.compute_laplacian_variance(frame)
                    tenengrad = VideoValidator.compute_tenengrad(frame)
                    lap_variances.append(lap_var)
                    tenengrad_scores.append(tenengrad)
            # Define thresholds (adjust as needed)
            lap_threshold = 100
            tenengrad_threshold = 10  # Arbitrary unit; tune as needed
            low_lap = sum(1 for v in lap_variances if v < lap_threshold)
            low_tenengrad = sum(1 for v in tenengrad_scores if v < tenengrad_threshold)
            # Reject if at least 3 out of 5 frames are below both thresholds
            if low_lap >= 3 and low_tenengrad >= 3:
                validations.append("Video contains blurry sections")

            duration = 0
            if props['fps'] > 0:
                duration = props['frame_count'] / props['fps']
            aspect_ratio = props['width'] / props['height'] if props['height'] > 0 else 0
            if props['width'] < 480 or props['height'] < 360:
                validations.append("Resolution too low (min 480x360)")
            if not (1.3 <= aspect_ratio <= 2.1):
                validations.append("Invalid aspect ratio (should be 1.33-2.1)")
            if duration < 30:
                validations.append("Video too short (min 30s)")
            if duration > 7200:
                validations.append("Video too long (max 2h)")
            if props['fps'] < 24:
                validations.append("Frame rate too low (min 24fps)")
            if props['fps'] > 120:
                validations.append("Frame rate too high (max 120fps)")
            cap.release()
            if validations:
                return False, "\n".join(validations)
            if progress_callback:
                progress_callback(100, "Validation complete!")
            return True, {
                'resolution': f"{props['width']}x{props['height']}",
                'duration': f"{int(duration)} seconds",
                'fps': f"{props['fps']} fps",
                'frame_count': props['frame_count'],
                'file_size': f"{file_size:.1f} MB",
                'aspect_ratio': f"{aspect_ratio:.2f}",
                'codec': props['codec'],
            }
        except Exception as e:
            return False, f"Validation error: {str(e)}"


# ---------------------------
# TimelineMarker widget
# ---------------------------
class TimelineMarker(QWidget):
    """
    A draggable marker representing either the 'start' or 'end' of a segment.
    The vertical line is centered by drawing at self.width()//2.
    """
    markerMoved = pyqtSignal(float)

    def __init__(self, segment, is_start=True, parent=None):
        super().__init__(parent)
        self.segment = segment
        self.is_start = is_start
        self.dragging = False

        # Marker size: increased for better visibility
        self.setFixedSize(20, 60)
        self.setCursor(Qt.SizeHorCursor)
        self.setMouseTracking(True)
        self.setStyleSheet("background-color: transparent;")

        self._old_pos_opacity = 1.0
        self._old_pos = None  # For showing a faded line while dragging
        self.dragThrottleTimer = QTimer(self)
        self.dragThrottleTimer.setSingleShot(True)
        self.dragThrottleTimer.timeout.connect(self._updateDragThumbnail)
        self.pendingTime = None

        timeline = self.parent()
        if timeline:
            marker_y = timeline.height() - timeline._segment_height - 20
            self.move(0, marker_y)
        self.raise_()

    @pyqtProperty(float)
    def oldPosOpacity(self):
        return self._old_pos_opacity

    @oldPosOpacity.setter
    def oldPosOpacity(self, val):
        self._old_pos_opacity = val
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.segment.state == SegmentState.CONFIRMED:
                return
            self.dragging = True
            self.drag_start_pos = event.pos()
            timeline = self.parent()
            if timeline:
                marker_center_x = self.x() + (self.width() // 2)
                self.drag_start_time = timeline.pixelToTime(marker_center_x)
            # Pause video for accurate adjustment
            if hasattr(self.parent(), 'video_player'):
                self.parent().video_player.pause()

    def mouseMoveEvent(self, event):
        if not self.dragging:
            return
        timeline = self.parent()
        if not timeline:
            return
        if self.segment.state == SegmentState.CONFIRMED:
            return
        new_left_x = self.mapToParent(event.pos()).x() - self.drag_start_pos.x()
        new_left_x = max(0, min(new_left_x, timeline.width() - self.width()))
        marker_center_x = new_left_x + (self.width() // 2)
        new_time = timeline.pixelToTime(marker_center_x)
        self._old_pos = self.pos()
        self.move(new_left_x, self.y())
        if self.is_start:
            potential_start = new_time
            potential_end = self.segment.end
        else:
            potential_start = self.segment.start
            potential_end = new_time
        if self._would_overlap(timeline, potential_start, potential_end):
            self.revert_to_original()
            return
        else:
            if self.is_start:
                self.segment.start = new_time
            else:
                self.segment.end = new_time
        self.pendingTime = new_time
        if not self.dragThrottleTimer.isActive():
            self.dragThrottleTimer.start(150)
        # timeline.update()
        self.markerMoved.emit(new_time)

    def mouseReleaseEvent(self, event):
        if not self.dragging:
            return

        self.dragging = False
        timeline = self.parent()

        # 1. Stop the drag timer immediately
        if self.dragThrottleTimer.isActive():
            self.dragThrottleTimer.stop()

        # 2. Disable hover previews during this process
        if timeline:
            timeline.disable_hover_preview = True

        # 3. Hide any preview popup
        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()

        # 4. If the segment is not confirmed, ask for confirmation
        if self.segment.state != SegmentState.CONFIRMED:
            reply = self._show_confirm_dialog(
                "Confirm Boundary Change",
                "Keep this new boundary position?"
            )
            if reply == QMessageBox.Yes:
                # Keep the new boundary: store it as original for now
                if self.is_start:
                    self.segment.original_start = self.segment.start
                else:
                    self.segment.original_end = self.segment.end

                # 5. Launch the fine-tuning dialog so the user can make minute adjustments.
                #    This dialog updates the timeline markers and preview in real time.
                boundary = "start" if self.is_start else "end"
                fine_tune_result = self.window().show_fine_tuning_dialog(self.segment, boundary)
                # If the user accepted the fine tuning, the segment boundary remains as adjusted.
                # Otherwise, if the user canceled, the FineTuneDialog's reject() will have reverted it.

                # 6. Animate the boundary change for visual feedback.
                timeline.animate_boundary_change(self.segment)
            else:
                # User did not confirm; revert to the original boundary values.
                if self.is_start:
                    self.segment.start = self.segment.original_start
                else:
                    self.segment.end = self.segment.original_end
                if self.segment.state == SegmentState.BEING_ADJUSTED:
                    self.segment.state = SegmentState.PENDING
                # timeline.update()
                self.revert_to_original()

        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()
        if timeline:
            timeline.update_marker_positions()
            timeline.update()

    def _show_confirm_dialog(self, title: str, text: str) -> int:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #FFFACD;
            }
            QPushButton {
                background-color: #333333;
                color: white;
                border: 2px solid #000000;
                border-radius: 4px;
                padding: 5px 15px;
                margin: 8px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #666666;
            }
        """)

        layout = QVBoxLayout(dialog)
        label = QLabel(text, dialog)
        layout.addWidget(label)

        btn_box = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
        btn_box.button(QDialogButtonBox.Yes).setText("Yes")
        btn_box.button(QDialogButtonBox.No).setText("No")
        layout.addWidget(btn_box)

        def on_accept():
            dialog.done(QMessageBox.Yes)

        def on_reject():
            dialog.done(QMessageBox.No)

        btn_box.accepted.connect(on_accept)
        btn_box.rejected.connect(on_reject)

        result = dialog.exec_()
        return result

    def revert_to_original(self):
        timeline = self.parent()
        if not timeline or not timeline.duration:
            return
        original_time = self.segment.original_start if self.is_start else self.segment.original_end
        original_x = timeline.timeToPixel(original_time) - (self.width() // 2)

        # If you prefer an immediate revert with no animation:
        self.move(original_x, self.y())
        timeline.update()

        return

    def updatePosition(self):
        timeline = self.parent()
        if not timeline:
            return
        timeline_height = timeline.height()
        seg_height = int(timeline_height * 0.8)
        y_offset = int((timeline_height - seg_height) / 2)
        marker_y = y_offset + (seg_height - self.height()) // 2

        boundary_time = self.segment.start if self.is_start else self.segment.end
        x = timeline.timeToPixel(boundary_time) - (self.width() // 2)
        self.move(x, marker_y)
        label = "Start" if self.is_start else "End"
        self.setToolTip(f"{label}: {boundary_time:.2f}s")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.transparent)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.segment.state == SegmentState.CONFIRMED:
            color = QColor(40, 167, 69)
        elif self.dragging:
            color = QColor(255, 140, 0)
        else:
            color = QColor(13, 110, 253)
        pen = QPen(color, 2)
        painter.setPen(pen)
        center_x = self.width() // 2
        painter.drawLine(center_x, 0, center_x, self.height())
        if self.dragging and self._old_pos is not None:
            old_x = self._old_pos.x() + (self.width() // 2)
            fade_color = QColor(128, 128, 128, int(255 * self._old_pos_opacity))
            painter.setPen(QPen(fade_color))
            painter.drawLine(old_x, 0, old_x, self.height())
        if self.segment.state != SegmentState.CONFIRMED:
            handle_rect = QRect(center_x - 4, 15, 8, 10)
            painter.fillRect(handle_rect, color)

    def _updateDragThumbnail(self):
        # If the segment is confirmed, hide the popup and do nothing
        if self.segment.state == SegmentState.CONFIRMED:
            if hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.hide_preview()
            return

        if self.pendingTime is not None:
            timeline = self.parent()
            pix = timeline.get_frame_at_time(self.pendingTime)
            if pix and hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.show_preview(pix)
            else:
                if hasattr(self.window(), 'preview_popup'):
                    self.window().preview_popup.hide_preview()

    def _would_overlap(self, timeline, new_start, new_end):
        for other in timeline.segments:
            if other is self.segment:
                continue
            if new_start < other.end and new_end > other.start:
                return True
        return False


# ---------------------------
# VideoTimeline widget
# ---------------------------
class VideoTimeline(QWidget):
    segmentClicked = pyqtSignal(object)
    timelineClicked = pyqtSignal(float)
    previewEnded = pyqtSignal()
    zoom_level_changed = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(300)
        # self.setAttribute(Qt.WA_OpaquePaintEvent)
        # self.setAutoFillBackground(True)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAutoFillBackground(False)
        palette = self.palette()
        palette.setColor(self.backgroundRole(), Qt.white)
        self.setPalette(palette)
        self.duration = 0
        # Only one active segment at a time
        self.segments = []
        self.current_time = 0
        self._zoom_level = 1.0
        self.zoom_offset = 0
        self._segment_height = 80
        self.preview_label = QLabel(self)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setVisible(False)  # Hide by default

        # Create and set the layout if not already present
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.preview_label)

        self.cap = None
        self.hovering_segment = None
        self.previewing_segment = None
        self.thumbnails = {}
        self.disable_hover_preview = False
        self.video_file_path = None
        self.cap = None

    def timeToPixel(self, t: float) -> int:
        if self.duration <= 0:
            return 0
        visible_width = self.width()
        visible_duration = self.duration / self._zoom_level
        time_offset = (self.zoom_offset / visible_width) * self.duration
        shifted_time = t - time_offset
        x = (shifted_time / visible_duration) * visible_width
        return int(round(x))

    def pixelToTime(self, x: int) -> float:
        if self.duration <= 0 or self.width() <= 0:
            return 0.0
        visible_width = self.width()
        visible_duration = self.duration / self._zoom_level
        time_offset = (self.zoom_offset / visible_width) * self.duration
        time_in_view = (x / visible_width) * visible_duration
        return time_in_view + time_offset

    def open_capture_once(self):
        """Safely manage single OpenCV capture"""
        if self.cap and self.cap.isOpened():
            return

        if self.video_file_path:
            try:
                self.cap = cv2.VideoCapture(self.video_file_path)
                if not self.cap.isOpened():
                    print("Failed to open OpenCV capture")
            except Exception as e:
                print(f"OpenCV init error: {e}")

    def clear_capture(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception as e:
                print(f"OpenCV release error: {e}")
            self.cap = None

    def get_frame_at_time(self, time_sec: float) -> QPixmap:
        if not self.cap or not self.cap.isOpened():
            self.open_capture_once()
            if not self.cap.isOpened():
                return None

        try:
            self.cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)
            ret, frame = self.cap.read()
            self.cap.release()
            if not ret or frame is None:
                return None

            h, w, ch = frame.shape
            bytes_per_line = ch * w
            q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            return pixmap

        except Exception as e:
            print(f"Frame extraction error: {e}")
            return None

    def set_duration(self, duration):
        self.duration = duration
        self.clear_thumbnail_cache()
        self.update()

    def start_preview(self, segment):
        self.previewing_segment = segment
        segment.original_state = segment.state
        segment.state = SegmentState.PREVIEW
        self.update_segment_state(segment, SegmentState.PREVIEW)
        self.update()

    def end_preview(self):
        print("Timeline: Ending preview mode")
        if self.previewing_segment:
            # Restore to PENDING (or original) state
            self.previewing_segment.state = getattr(
                self.previewing_segment, 'original_state', SegmentState.PENDING
            )
            self.update_segment_state(self.previewing_segment, self.previewing_segment.state)
            self.previewing_segment = None
        self.disable_hover_preview = False
        self.clear_thumbnail_cache()
        self.update_marker_positions()
        self.update()
        QApplication.processEvents()

    def set_current_time(self, time_seconds):
        """
        Update the playhead position without auto-ending any preview.
        """
        self.current_time = float(time_seconds)
        if self.duration > 0:
            self.current_time = max(0, min(self.current_time, self.duration))
        # We intentionally do NOT emit previewEnded here.
        self.update()

    def add_segment(self, segment: VideoSegment):
        """Clear previous segments before adding new one"""
        print(f"[DEBUG] add_segment: segments before = {len(self.segments)}")
        # Remove existing segments
        for seg in self.segments[:]:
            self._remove_segment(seg)

        # Create new markers
        segment.start_marker = TimelineMarker(segment, True, self)
        segment.end_marker = TimelineMarker(segment, False, self)

        # Position markers
        start_x = self.timeToPixel(segment.start) - (segment.start_marker.width() // 2)
        end_x = self.timeToPixel(segment.end) - (segment.end_marker.width() // 2)

        segment.start_marker.move(start_x, segment.start_marker.y())
        segment.end_marker.move(end_x, segment.end_marker.y())
        segment.start_marker.show()
        segment.end_marker.show()
        segment.start_marker.raise_()
        segment.end_marker.raise_()

        # Add to tracking
        self.segments.append(segment)
        self.update()
        QApplication.processEvents()
        print(f"[DEBUG] add_segment: segments after = {len(self.segments)}")

    def update_segment_state(self, segment: VideoSegment, new_state: SegmentState):
        segment.state = new_state
        if new_state == SegmentState.PREVIEW:
            if segment.start_marker:
                segment.start_marker.setEnabled(False)
            if segment.end_marker:
                segment.end_marker.setEnabled(False)
        if new_state == SegmentState.REJECTED:
            if segment.start_marker:
                segment.start_marker.hide()
            if segment.end_marker:
                segment.end_marker.hide()
            self.update()
            if hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.hide_preview()
            return
        if new_state == SegmentState.CONFIRMED:
            if segment.start_marker:
                segment.start_marker.setEnabled(False)
                segment.start_marker.show()
            if segment.end_marker:
                segment.end_marker.setEnabled(False)
                segment.end_marker.show()
            self.update()
            if hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.hide_preview()
            return
        if new_state == SegmentState.PENDING:
            if segment.start_marker:
                segment.start_marker.setEnabled(True)
                segment.start_marker.show()
            if segment.end_marker:
                segment.end_marker.setEnabled(True)
                segment.end_marker.show()
            self.update()
            return
        if new_state == SegmentState.BEING_ADJUSTED:
            if segment.start_marker:
                segment.start_marker.setEnabled(True)
                segment.start_marker.show()
            if segment.end_marker:
                segment.end_marker.setEnabled(True)
                segment.end_marker.show()
            self.update()
            return

    def animate_boundary_change(self, segment: VideoSegment):
        group = QParallelAnimationGroup()
        fade_start = QPropertyAnimation(segment.start_marker, b"oldPosOpacity")
        fade_start.setDuration(300)
        fade_start.setStartValue(1.0)
        fade_start.setEndValue(0.0)
        fade_end = QPropertyAnimation(segment.end_marker, b"oldPosOpacity")
        fade_end.setDuration(300)
        fade_end.setStartValue(1.0)
        fade_end.setEndValue(0.0)
        group.addAnimation(fade_start)
        group.addAnimation(fade_end)

        def on_finished():
            if hasattr(segment.start_marker, '_old_pos'):
                segment.start_marker._old_pos = None
            if hasattr(segment.end_marker, '_old_pos'):
                segment.end_marker._old_pos = None
            self.update()

        group.finished.connect(on_finished)
        group.start(QAbstractAnimation.DeleteWhenStopped)

    def _remove_segment(self, segment: VideoSegment):
        if segment.start_marker:
            segment.start_marker.deleteLater()
            segment.start_marker = None
        if segment.end_marker:
            segment.end_marker.deleteLater()
            segment.end_marker = None
        self.segments.remove(segment)
        self.update()
        QApplication.processEvents()

    def clear_all_segments(self):
        for seg in self.segments[:]:
            self._remove_segment(seg)
        QApplication.processEvents()  # Force deletion and repaint
        print("[DEBUG] Timeline children after clear:", self.findChildren(QWidget))
        self.update_marker_positions()
        self.update()

    def rebuild_segments(self):
        self.clear_thumbnail_cache()
        self.update()
        QApplication.processEvents()

    def update_marker_positions(self):
        if self.duration <= 0 or getattr(self.window(), 'annotating', False):
            return
        visible_width = self.width()
        if visible_width == 0:
            return
        visible_duration = self.duration / self._zoom_level if self._zoom_level != 0 else self.duration
        time_offset = (self.zoom_offset / visible_width) * self.duration

        # Update positions for all segments
        for seg in self.segments:
            if seg.start_marker and seg.start_marker.isVisible():
                start_time = seg.start - time_offset
                start_x = int((start_time / visible_duration) * visible_width)
                try:
                    anim = QPropertyAnimation(seg.start_marker, b"pos")
                    anim.setDuration(100)
                    anim.setEndValue(QPoint(start_x, seg.start_marker.y()))
                    anim.start(QAbstractAnimation.DeleteWhenStopped)
                except Exception as e:
                    print("Exception updating start_marker:", e)
            if seg.end_marker and seg.end_marker.isVisible():
                end_time = seg.end - time_offset
                end_x = int((end_time / visible_duration) * visible_width)
                try:
                    anim = QPropertyAnimation(seg.end_marker, b"pos")
                    anim.setDuration(100)
                    anim.setEndValue(QPoint(end_x, seg.end_marker.y()))
                    anim.start(QAbstractAnimation.DeleteWhenStopped)
                except Exception as e:
                    print("Exception updating end_marker:", e)

    def show_preview(self, pixmap: QPixmap):
        if pixmap and not pixmap.isNull():
            self.preview_label.setPixmap(pixmap)
            self.preview_label.setVisible(True)
        else:
            self.preview_label.clear()
            self.preview_label.setVisible(False)

    @pyqtSlot(QPixmap)
    def show_thumbnail(self, pixmap):
        if pixmap and not pixmap.isNull():
            if hasattr(self, 'preview_label'):
                self.preview_label.setPixmap(pixmap)
                self.show()
        else:
            self.hide()

    def clear_thumbnail_cache(self):
        self.thumbnails.clear()

    def request_thumbnail(self, time_pos):
        cache_key = int(time_pos)
        if cache_key in self.thumbnails:
            # Use QMetaObject to invoke on the main thread
            QMetaObject.invokeMethod(self, "show_thumbnail",
                                     Qt.QueuedConnection,
                                     Q_ARG(QPixmap, self.thumbnails[cache_key]))
            return

        # Use a thread-safe way to generate and cache thumbnails
        def generate_thumbnail():
            thumbnail = self.get_frame_at_time(time_pos)
            if thumbnail:
                self.thumbnails[cache_key] = thumbnail
                QMetaObject.invokeMethod(self, "show_thumbnail",
                                         Qt.QueuedConnection,
                                         Q_ARG(QPixmap, thumbnail))

        # Run in a separate thread to avoid blocking UI
        QThreadPool.globalInstance().start(generate_thumbnail)

    def leaveEvent(self, event):
        self.disable_hover_preview = False
        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            click_time = self.pixelToTime(event.pos().x())
            for seg in self.segments:
                if seg.state != SegmentState.REJECTED:
                    if seg.start <= click_time <= seg.end:
                        self.segmentClicked.emit(seg)
                        return
            self.timelineClicked.emit(click_time)

    def mouseMoveEvent(self, event):
        if self.duration <= 0 or self.disable_hover_preview:
            if hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.hide_preview()
            return
        hover_time = self.pixelToTime(event.pos().x())
        hover_seg = next((s for s in self.segments
                          if s.start <= hover_time <= s.end
                          and s.state != SegmentState.REJECTED), None)
        if hover_seg and hover_seg.state != SegmentState.CONFIRMED:
            self.request_thumbnail(hover_time)
        else:
            if hasattr(self.window(), 'preview_popup'):
                self.window().preview_popup.hide_preview()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        # painter.begin(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.eraseRect(self.rect())  # Critical fix for ghosting
        painter.fillRect(self.rect(), Qt.white)
        painter.setFont(QFont("Arial", 8))
        timeline_height = self.height()
        seg_height = int(timeline_height * 0.8)
        y_offset = int((timeline_height - seg_height) / 2)
        self._draw_time_grid(painter, y_offset, seg_height)
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(0, int(y_offset + seg_height / 2), self.width(), int(y_offset + seg_height / 2))
        if self.duration <= 0:
            return
        if self.segments:
            seg = self.segments[0]
            if seg.state != SegmentState.REJECTED:
                start_x = self.timeToPixel(seg.start)
                end_x = self.timeToPixel(seg.end)
                if seg.state == SegmentState.CONFIRMED:
                    color = QColor(40, 167, 69, 180)
                    border_color = QColor(25, 135, 84)
                else:
                    color = QColor(13, 110, 253, 180)
                    border_color = QColor(0, 80, 255)
                rect = QRect(start_x, y_offset, end_x - start_x, seg_height)
                painter.fillRect(rect, color)
                painter.setPen(QPen(border_color, 2))
                painter.drawRect(rect)
                if seg == self.hovering_segment:
                    dur = seg.end - seg.start
                    painter.setPen(Qt.white)
                    painter.drawText(rect, Qt.AlignCenter, f"{dur:.1f}s")
        # painter.end()

    def _draw_time_grid(self, painter, y_offset, height):
        if self.duration <= 0:
            return
        visible_width = self.width()
        # Divide the duration into a fixed number of intervals (e.g., 10 intervals).
        interval = max(1, self.duration / 10)
        t = 0.0
        painter.setPen(QPen(QColor(200, 200, 200)))
        while t <= self.duration:
            # Compute the x position corresponding to time t.
            x = self.timeToPixel(t)
            if 0 <= x <= visible_width:
                # Draw a vertical grid line.
                painter.drawLine(x, y_offset, x, y_offset + height)
                painter.setPen(Qt.black)
                # Draw time text centered at x with left/right margins.
                # Adjust the x offset and width so that text doesn't get cut off.
                painter.drawText(x - 10, y_offset - 10, 30, 20, Qt.AlignCenter, f"{int(t)}s")
                painter.setPen(QPen(QColor(200, 200, 200)))
            t += interval

    def closeEvent(self, event):
        self.clear_capture()
        super().closeEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for seg in self.segments:
            if seg.start_marker:
                seg.start_marker.updatePosition()
            if seg.end_marker:
                seg.end_marker.updatePosition()
        self.update_marker_positions()
        zoom_container = self.findChild(QWidget, "zoom_container")
        if zoom_container:
            zoom_container.move(self.width() - 150, 10)


# ---------------------------
# WorkflowButton
# ---------------------------
class WorkflowButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.button_style = """
            QPushButton {
                background-color: whitesmoke;
                font: bold 15.75pt 'Segoe Print';
                color: peru;
                border-radius: 15px;
                padding: 10px;
                min-height: 50px;
                width: 100%;
            }
            QPushButton:hover {
                color: green;
                background-color: #f0f0f0;
            }
            QPushButton:disabled {
                color: gray;
                background-color: #e0e0e0;
            }
        """
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)
        self.status = QLabel()
        self.status.setVisible(False)
        self.status.setStyleSheet("color: #4CAF50;")
        self.layout.addWidget(self.progress)
        self.layout.addWidget(self.status)

    def show_progress(self, show=True):
        self.progress.setVisible(show)
        if show:
            self.setEnabled(False)

    def update_progress(self, value, status=None):
        self.progress.setValue(value)
        if status:
            self.status.setText(status)
            self.status.setVisible(True)

    def complete(self, success=True):
        self.progress.setVisible(False)
        self.setEnabled(False)
        self.status.setStyleSheet(
            f"color: {'#4CAF50' if success else '#f44336'};"
        )

# ---------------------------
# Main Window (xG)
# ---------------------------
class xG(QMainWindow):
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.settings = QSettings('YourCompany', 'xG')
        self.current_window_state = WindowState.NORMAL
        self.processing_thread = None
        self.video_player = None
        self.current_segment = None
        self.timeline = None
        self.current_state = 'initial'
        self.shot_timestamps = []
        self.skipped_segments = []
        self.reviewed_shots = 0
        self.total_shots = 0
        self.confirmed_segments = []  # to store confirmed segments
        self.initUI()
        self.restore_window_state()
        self.setup_shortcuts()
        self.error_occurred.connect(self.show_error_message)
        self.preview_popup = PreviewPopup(self)
        self.processingComplete = False
        self.pendingQueue = []  # Queue for incoming pending segments
        self.homography_history = []
        self.segments_pending_homography = []  # Segments with some landmarks, H failed
        self.segments_pending_direct_pitch = []  # Segments with no/insufficient landmarks
        self.all_landmarks = {
            # Corners (in meters)
            "Top left corner": (1, 0.0, 0.0),
            "Top right corner": (27, 120.0, 0.0),  # 12000 StatsBomb units → 120m
            "Bottom left corner": (6, 0.0, 80.0),  # 8000 → 80m
            "Bottom right corner": (32, 120.0, 80.0),

            # Center line
            "Center line top": (15, 60.0, 0.0),  # 6000 → 60m
            "Center line bottom": (18, 60.0, 80.0),

            # Circle
            "Center circle left": (14, 50.0, 40.0),  # 5000 → 50m
            "Center circle top": (16, 60.0, 30.0),
            "Center circle right": (19, 70.0, 40.0),
            "Center circle bottom": (17, 60.0, 50.0),

            # Penalty spots
            "Left penalty spot": (9, 12.0, 40.0),  # 1200 → 12m
            "Right penalty spot": (24, 108.0, 40.0),  # 10800 → 108m

            # 6-yd boxes (5.5m boxes in reality)
            "Left 6yd box top": (7, 6.0, 30.0),  # 600 → 6m
            "Left 6yd box bottom": (8, 6.0, 50.0),
            "Right 6yd box top": (25, 114.0, 30.0),  # 11400 → 114m
            "Right 6yd box bottom": (26, 114.0, 50.0),

            # 18-yd boxes (16.5m boxes)
            "Left 18yd box top": (10, 18.0, 18.0),  # 1800 → 18m
            "Left 18yd box bottom": (13, 18.0, 62.0),  # 6200 → 62m
            "Left arc upper": (11, 18.0, 30.85),
            "Left arc lower": (12, 18.0, 49.15),
            "Right arc upper": (21, 102.0, 30.85),
            "Right arc lower": (22, 102.0, 49.15),
            "Left 18yb box meet with pitch boundary top": (2, 0.0, 18.0),
            "Left 18yb box meet with pitch boundary bottom": (5, 0.0, 62.0),
            "Right 18yb box meet with pitch boundary top": (28, 120.0, 18.0),
            "Right 18yb box meet with pitch boundary bottom": (31, 120.0, 62.0),
            "Right 18yd box top": (20, 102.0, 18.0),  # 10200 → 102m
            "Right 18yd box bottom": (23, 102.0, 62.0),

            # Goal boxes
            "Left goal box top": (3, 0.0, 30.0),
            "Left goal box bottom": (4, 0.0, 50.0),
            "Right goal box top": (29, 120.0, 30.0),
            "Right goal box bottom": (30, 120.0, 50.0),
        }

        self.right_side_ids = [19, 20, 21, 22, 23, 24, 25, 26, 28, 29, 30, 31, 27, 32]
        self.left_side_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

        self.id_to_world = {v[0]: (v[1], v[2]) for v in self.all_landmarks.values()}
        self.id_to_landmark = {v[0]: k for k, v in self.all_landmarks.items()}

        self.scaler_path = "xgboost&lightgbm_feature_scaler_with_zone14.pkl"
        self.xg_model_path = "XGBoost_with_zone14_acc_0.7003_err_0.0820.pkl"
        self.label_encoder_path = 'label_encoders_with_zone14.pkl'
        self.xg_model = XGModel(self.scaler_path, self.xg_model_path, self.label_encoder_path)

    def create_workflow_buttons(self, layout):
        self.upload_btn = WorkflowButton("1. Upload Video")
        self.process_btn = WorkflowButton("2. Process Video")
        self.review_btn = WorkflowButton("3. Review Shots")
        for btn in [self.upload_btn, self.process_btn, self.review_btn]:
            btn.setStyleSheet(btn.button_style)
            layout.addWidget(btn, stretch=20)
        self.upload_btn.clicked.connect(self.videobutton_Click)
        self.process_btn.clicked.connect(self.start_processing)
        self.review_btn.clicked.connect(self.start_review)
        self.process_btn.setEnabled(False)
        self.review_btn.setEnabled(False)

    def save_confirmed_actions(self):
        confirmed_segments_data = []
        for seg in self.confirmed_segments:
            confirmed_segments_data.append(seg)
        if not confirmed_segments_data:
            print("No confirmed segments to save.")
            return
        base_name = os.path.splitext(os.path.basename(self.fileName))[0]
        json_filename = f"actions_found_from_{base_name}.json"
        data = {
            "video_file": self.fileName,
            "confirmed_segments": confirmed_segments_data
        }
        try:
            with open(json_filename, "w") as f:
                json.dump(data, f, indent=4)
            print(f"Saved actions to {json_filename}")
        except Exception as e:
            print(f"Error saving actions to JSON: {e}")

    def check_review_complete(self):
        try:
            # Get total segments that need review
            total_segments = len(self.shot_timestamps)

            # Count processed segments (confirmed + skipped)
            processed_segments = len(self.confirmed_segments) + len(self.skipped_segments)

            print(f"[DEBUG] Review progress: {processed_segments}/{total_segments} segments processed")

            # Only proceed if all segments are processed and processing is complete
            if processed_segments >= total_segments:
                if not self.processingComplete:
                    print("[DEBUG] Processing still in progress; waiting before saving actions.")
                    return

                # First ask about saving
                reply = QMessageBox.question(
                    self,
                    "Review Complete",
                    "All segments have been reviewed. Do you want to save the actions?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    self.save_confirmed_actions()
                    QMessageBox.information(self, "Save Successful", "Actions saved successfully.")

                # Disable review button
                self.review_btn.setEnabled(False)

                # Ask about loading new video
                next_action = QMessageBox.question(
                    self,
                    "Next Action",
                    "Would you like to load a new video?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if next_action == QMessageBox.Yes:
                    self.resetUI()
                    self.videobutton_Click()
                else:
                    QApplication.quit()

        except Exception as e:
            print(f"[ERROR] Error in check_review_complete: {e}")
            traceback.print_exc()

    def start_review(self):
        """
        When the user clicks "Review Shots":
        Load exactly one pending segment onto the timeline.
        """
        # If nothing left to review, disable the button
        if not self.pendingQueue:
            print("[DEBUG] No pending segments to review.")
            self.review_btn.setEnabled(False)
            return

        # Pull one segment out of the queue
        next_seg = self.pendingQueue.pop(0)

        # Clear any leftovers (shouldn't be any)
        self.timeline.clear_all_segments()

        # Add the next segment to the timeline
        self.timeline.add_segment(next_seg)
        self.current_segment = next_seg

        # Update our status display
        self.update_status_text()

        # Make sure the timeline is visible
        self.timeline_container.show()

        # Keep the review button enabled until queue is empty
        self.review_btn.setEnabled(bool(self.pendingQueue))

    def initUI(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.setWindowTitle('Expected Goals')
        self.setStyleSheet("""
            QMainWindow {
                background-color: white;
                color: green;
            }
        """)
        self.setWindowIcon(QIcon('black.png'))
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 10, 20, 10)
        header_layout.addStretch(1)
        self.header = QLabel('Goal Probability Of Shot Attempts')
        self.header.setStyleSheet("""
            QLabel {
                color: peru;
                font-weight: bold;
                padding: 10px;
            }
        """)
        header_layout.addWidget(self.header)
        self.main_layout.addWidget(header_container)
        self.content_layout = QHBoxLayout()
        self.left_panel = self.create_left_panel()
        left_layout = self.left_panel.layout()
        self.create_workflow_buttons(left_layout)
        self.content_layout.addWidget(self.left_panel, 1)
        self.video_panel = self.create_video_panel()
        self.setup_shortcuts()
        self.content_layout.addWidget(self.video_panel, 4)
        self.main_layout.addLayout(self.content_layout)
        self.media_panel = self.create_media_controls()
        self.main_layout.addWidget(self.media_panel)
        self.media_panel.setVisible(False)
        self.resize_timer = QTimer()
        self.resize_timer.setSingleShot(True)
        self.resize_timer.timeout.connect(self.delayed_resize)

    def update_header_font(self):
        window_width = self.width()
        base_size = window_width / 40
        size = max(16, min(32, base_size))
        font = QFont('Segoe Script', size, QFont.Bold)
        self.header.setFont(font)
        self.header.parentWidget().layout().update()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'resize_timer'):
            self.resize_timer.stop()
        self.resize_timer.start(150)

    def delayed_resize(self):
        self.update_header_font()

    def create_left_panel(self):
        panel = QWidget()
        panel.setStyleSheet("background-color: moccasin;")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)
        self.textedit = QTextEdit()
        self.textedit.setFont(QFont('Segoe Print', 16, QFont.Bold))
        self.textedit.setStyleSheet("""
            QTextEdit {
                color: green;
                background-color: white;
                border: 1px solid green;
                border-radius: 5px;
            }
        """)
        self.textedit.setReadOnly(True)
        layout.addWidget(self.textedit, stretch=70)
        return panel

    def handle_segment_click(self, seg: VideoSegment):
        # First check if preview is already active and exit early if so
        if self.video_player and self.video_player.is_previewing:
            print("Preview already active - ignoring click")
            return

        # Stop any active marker dragging and hide preview popup
        for marker in (seg.start_marker, seg.end_marker):
            if marker and marker.dragging:
                marker.dragging = False
                marker.pendingTime = None
                if marker.dragThrottleTimer.isActive():
                    marker.dragThrottleTimer.stop()
        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()

        # Create a context menu with four options.
        menu = QMenu(self)
        preview_action = menu.addAction("Preview Segment")
        preview_action.triggered.connect(lambda: self.preview_segment(seg))
        skip_action = menu.addAction("Skip Region")
        skip_action.triggered.connect(lambda: self.skip_segment(seg))
        confirm_action = menu.addAction("Confirm Segment")
        confirm_action.triggered.connect(lambda: self.confirm_segment(seg))
        menu.exec_(QCursor.pos())

    def preview_segment(self, segment):
        # 1) If there's already a running preview, tear it down.
        if getattr(self, "_preview_active", False):
            self._stop_preview()
        self._preview_active = True

        # 2) Disable hover-on-timeline previews while this forced preview runs.
        self.timeline.disable_hover_preview = True

        # 3) Hook up a one-time cleanup when the preview actually ends,
        #    whether the user clicks "stop" or it naturally finishes.
        try:
            self.timeline.previewEnded.disconnect(self._stop_preview)
        except TypeError:
            pass
        self.timeline.previewEnded.connect(self._stop_preview)

        # 4) If the recognition thread is already running, pause it first.
        if self.recognition_worker.isRunning():
            # stash a lambda that captures our segment
            self._pause_handler = lambda: self._on_worker_paused_for_preview(segment)
            # connect our handler before we request the pause
            self.recognition_worker.paused.connect(self._pause_handler)
            # now actually ask the worker to pause mid-inference
            self.recognition_worker.pause()
        else:
            # nothing to pause — start immediately
            self._start_segment_preview(segment)

    def _on_worker_paused_for_preview(self, segment):
        # 1) disconnect the same lambda instance we connected above
        try:
            self.recognition_worker.paused.disconnect(self._pause_handler)
        except TypeError:
            pass

        # 2) Kick off the actual timeline preview
        self._start_segment_preview(segment)

    def _start_segment_preview(self, seg: VideoSegment):
        """
        1. Ask for speed
        2. Mark UI state
        3. Show & raise controls
        4. Schedule our own stop timer
        5. Give VLC time to seek, then play
        """
        # ——— 1. Speed selector (unchanged) ———
        speed_dialog = QDialog(self)
        speed_dialog.setWindowTitle("Select Play Speed")
        speed_dialog.setStyleSheet("""
            QDialog { background-color: blanchedalmond; }
            QRadioButton { font: 12pt 'Segoe Print'; color: saddlebrown; }
        """)
        layout = QVBoxLayout(speed_dialog)
        speed_group = QButtonGroup(speed_dialog)
        for sp in (0.5, 1.0, 1.5, 2.0):
            rb = QRadioButton(f"{sp}x")
            if sp == 1.0:
                rb.setChecked(True)
            speed_group.addButton(rb)
            layout.addWidget(rb)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(speed_dialog.accept)
        buttons.rejected.connect(speed_dialog.reject)
        layout.addWidget(buttons)
        if speed_dialog.exec_() != QDialog.Accepted:
            self._preview_active = False
            self.timeline.disable_hover_preview = False
            return
        rate = float(speed_group.checkedButton().text().rstrip('x'))

        # --- 2. Validate segment boundaries ---
        video_duration = self.video_player.duration
        seg.start = max(0, min(seg.start, video_duration - 0.1))
        seg.end = max(seg.start + 0.1, min(seg.end, video_duration))

        if seg.start >= seg.end:  # Handle invalid segments
            QMessageBox.warning(self, "Invalid Segment",
                                "Selected segment has invalid time boundaries")
            return

        # --- 3. Atomic UI state update ---
        self.current_segment = seg
        seg.original_state = seg.state
        seg.state = SegmentState.PREVIEW
        self.timeline.update_segment_state(seg, SegmentState.PREVIEW)
        self.timeline.previewing_segment = seg
        QApplication.processEvents()  # Force UI update

        # --- 4. Prepare playback with VLC async handling ---
        try:
            # Show controls first
            self.media_panel.setVisible(True)
            self.media_panel.raise_()
            self._setup_preview_controls()

            # Set rate before any playback operations
            self.video_player.set_rate(rate)

            # Calculate duration with rate adjustment + buffer
            clip_duration = (seg.end - seg.start) / rate
            buffer_ms = 500  # Extra time to handle VLC imprecision

            # --- 5. Start playback with safe seek ---
            def start_playback():
                try:
                    # Ensure player is properly reset
                    if self.video_player.player:
                        self.video_player.player.stop()

                    # Init playback with clean state
                    self.video_player.play_segment(seg)

                    # Set stop timer
                    QTimer.singleShot(int(clip_duration * 1000) + buffer_ms,
                                      self._stop_preview)

                except Exception as e:
                    print(f"Playback error: {e}")
                    self._stop_preview()

            # Give VLC time to initialize before seeking
            QTimer.singleShot(300, start_playback)

        except Exception as e:
            print(f"Preview setup error: {e}")
            self._stop_preview()

    def _setup_preview_controls(self):
        """
        Enable and hook every button in self.media_buttons and the volume slider.
        """
        for name, btn in self.media_buttons.items():
            btn.setEnabled(True)
            try:
                btn.clicked.disconnect()
            except Exception:
                pass
            if name == "Play":
                btn.clicked.connect(self.video_player.play)
            elif name == "Pause":
                btn.clicked.connect(self.video_player.pause)
            elif name == "Stop":
                btn.clicked.connect(self._stop_preview)
            elif name == "Rewind":
                btn.clicked.connect(lambda: self.video_player.rewind(True))
            elif name == "Forward":
                btn.clicked.connect(lambda: self.video_player.forward(True))

        # Volume control
        self.volume_slider.setEnabled(True)
        self.volume_slider.valueChanged.disconnect()
        self.volume_slider.valueChanged.connect(self.update_volume_label)

    def _stop_preview(self):
        """
        Restores UI, timeline, and (if needed) resumes the inference worker.
        """
        if not getattr(self, "_preview_active", False):
            return
        self._preview_active = False

        # 1) Stop VLC preview and its timer
        try:
            self.video_player._stop_subclip()
        except Exception:
            pass

        # 2) Restore the segment's state visually
        if self.current_segment and self.current_segment in self.timeline.segments:
            orig = getattr(self.current_segment, "original_state", SegmentState.PENDING)
            self.current_segment.state = orig
            self.timeline.update_segment_state(self.current_segment, orig)
        self.current_segment = None

        # 3) Hide media controls
        self.media_panel.setVisible(False)
        for btn in self.media_panel.findChildren(QPushButton):
            btn.setEnabled(False)
        self.volume_slider.setEnabled(False)

        # 4) Reset timeline hover/preview flags
        self.timeline.previewing_segment = None
        self.timeline.disable_hover_preview = False
        self.timeline.clear_thumbnail_cache()
        self.timeline.clear_capture()

        # 5) Reset playback rate
        self.video_player.set_rate(1.0)

        # 6) **Only** resume the worker if it was actually paused
        if (self.recognition_worker
                and self.recognition_worker.isRunning()
                and getattr(self.recognition_worker, "_should_pause", False)
        ):
            self.recognition_worker.resume()

    def skip_segment(self, seg: VideoSegment):
        """Rejects (skips) the current segment."""
        # If we are previewing, stop first
        if self.video_player.is_previewing:
            self._stop_preview()

        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()

        # Mark it REJECTED on the timeline
        self.timeline.update_segment_state(seg, SegmentState.REJECTED)
        # Also mark it in shot_timestamps
        for t in self.shot_timestamps:
            if abs(t['start'] - seg.start) < 0.001 and abs(t['end'] - seg.end) < 0.001:
                t['state'] = 'rejected'
                break

        # Update stats
        self.update_status_text()
        # Remove from timeline entirely
        self.skipped_segments.append(seg)
        self.timeline._remove_segment(seg)
        self.check_review_complete()

    @pyqtSlot(object)
    def remove_segment_slot(self, seg):
        # Safely remove the segment's markers and remove it from the timeline.
        if seg.start_marker:
            try:
                seg.start_marker.markerMoved.disconnect()
            except Exception:
                pass  # Already disconnected or never connected
            seg.start_marker.hide()
            seg.start_marker.setParent(None)
            seg.start_marker.deleteLater()
        if seg.end_marker:
            try:
                seg.end_marker.markerMoved.disconnect()
            except Exception:
                pass
            seg.end_marker.hide()
            seg.end_marker.setParent(None)
            seg.end_marker.deleteLater()
        if seg in self.timeline.segments:
            self.timeline.segments.remove(seg)
        self.timeline.update()

    def remove_and_rebuild(self, seg: VideoSegment):
        self.remove_segment_slot(seg)
        # Rebuild the timeline so that the remaining segments are updated correctly.
        self.timeline.rebuild_segments()

    def interpolate_positions(self, start, end, num=5):
        """Interpolate between start and end positions."""
        if num < 2:
            return [start, end]

        positions = []
        for i in range(num):
            t = i / (num - 1)
            x = start[0] + t * (end[0] - start[0])
            y = start[1] + t * (end[1] - start[1])
            positions.append((x, y))

        return positions

    def sample_segment_frames(self, video_path, start_time, end_time, num_frames=5):
        cap = cv2.VideoCapture(video_path)
        duration = end_time - start_time
        times = [start_time + i * duration / (num_frames - 1) for i in range(num_frames)]
        frames = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret and frame is not None:
                rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_img.shape
                bytes_per_line = ch * w
                qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                frames.append(pixmap)
        cap.release()
        return frames

    def show_shot_transition(self, seg):
        if not hasattr(self, "segment_analysis_results") or not self.segment_analysis_results:
            print("[DEBUG] No segment analysis results for shot transition dialog.")
            return
        # Find the analysis result for this segment
        seg_data = next(
            (d for d in self.segment_analysis_results
             if abs(d["start_time"] - seg.start) < 1e-3 and abs(d["end_time"] - seg.end) < 1e-3),
            None
        )
        if not seg_data:
            print("[DEBUG] No analysis result found for this segment.")
            return
        start_pitch = self.project_point(seg_data["start_ball_pos"], np.array(seg_data["homography"]))
        end_pitch = self.project_point(seg_data["end_ball_pos"], np.array(seg_data["homography"]))
        if start_pitch is None or end_pitch is None:
            print("[DEBUG] Cannot show shot transition: missing pitch positions.")
            return
        pitch_positions = self.interpolate_positions(start_pitch, end_pitch, num=9)
        frames = self.sample_segment_frames(self.fileName, seg.start, seg.end, num_frames=9)
        frame_pixmaps = [self.bgr_to_pixmap(f) for f in frames]
        if len(frame_pixmaps) < 2:
            print("[DEBUG] Not enough frames for shot transition dialog.")
            return
        dlg = ShotTransitionDialog(frame_pixmaps, pitch_positions, parent=self)
        dlg.exec_()

    def show_shot_transition_for_retried_segment(self, seg, start_pitch, end_pitch):
        """Show shot transition visualization for a segment that was retried"""

        print("[DEBUG] Showing shot transition after direct mapping.")
        frames = self.sample_segment_frames(
            self.fileName,
            seg.start,
            seg.end,
            num_frames=5
        )
        frame_pixmaps = [self.bgr_to_pixmap(frame) for frame in frames]
        pitch_positions = self.interpolate_positions(start_pitch, end_pitch, num=5)
        dlg = ShotTransitionDialog(
            frame_pixmaps,
            pitch_positions,
            parent=self,
            retried=True
        )
        dlg.exec_()

    def _show_confirm_dialog(self,
                             title: str,
                             message: str,
                             yes_text: str = "Yes",
                             no_text: str = "No",
                             default: int = QMessageBox.Yes):
        """
        Show a modal Yes/No dialog and return a QMessageBox.StandardButton-like value.
        Return values: QMessageBox.Yes or QMessageBox.No

        Usage:
            resp = self._show_confirm_dialog("Reopen?", "Would you like to reopen mapping?")
            if self._confirm_yes(resp):
                ...
        """
        parent = self if hasattr(self, "parent") or True else None
        msg = QMessageBox(parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(title)
        msg.setText(message)

        # Add custom-labeled buttons and keep references
        yes_btn = msg.addButton(yes_text, QMessageBox.YesRole)
        no_btn = msg.addButton(no_text, QMessageBox.NoRole)

        # Set default button
        if int(default) == int(QMessageBox.Yes):
            msg.setDefaultButton(yes_btn)
        else:
            msg.setDefaultButton(no_btn)

        # Execute and return a standardized value
        msg.exec_()
        clicked = msg.clickedButton()
        if clicked is yes_btn:
            return QMessageBox.Yes
        else:
            return QMessageBox.No

    def _confirm_yes(self, resp) -> bool:
        """
        Robust check that returns True if `resp` means 'Yes/Confirm'.
        Accepts either:
          - a bool (True/False),
          - a QMessageBox.StandardButton (e.g. QMessageBox.Yes),
          - or an int (compat).
        """
        try:
            if isinstance(resp, bool):
                return resp
            return int(resp) == int(QMessageBox.Yes)
        except Exception:
            return False

    def detect_pitch_keypoints_with_fallback(
            self,
            frames: List[np.ndarray],
            id_to_world: Dict[int, Tuple[float, float]],
            id_to_landmark: Dict[int, str],
            left_side_ids: List[int],
            right_side_ids: List[int],
            segment=None
    ) -> Tuple[Optional[Dict[int, Tuple[float, float, float]]], Optional[Set[int]]]:
        """
        Opens LandmarkMappingDialog over all frames in `frames`.  If the user
        maps fewer than 4 points and clicks Confirm, forces a FrameSelectorDialog
        to pick a new frame and try again.
        Returns (mapping, ids) or (None, None).
        """
        try:
            # Build the {id: (name,(x_m,y_m))} dict
            landmarks = {
                lid: (id_to_landmark[lid], id_to_world[lid])
                for lid in id_to_world
            }

            # edge connections
            edges = [
                (1, 27), (1, 6), (6, 32), (27, 32),  # pitch boundary
                (2, 10),  # left 18 yard box meets pitch boundary at the top to left 18 yard box top
                (3, 7),  # left 6 yard box meets pitch boundary at the top to left 6 yard box top
                (4, 8),  # left 6 yard box meets pitch boundary at the bottom to left 6 yard box bottom
                (7, 8),  # left 6 yard box top to bottom
                (3, 4),  # left 6 yard box meets pitch boundary at the top to bottom
                (10, 11),
                (11, 12),
                (12, 13),
                # (10, 13), #left 18 yd box top to left 18 yd box bottom
                (5, 13),  # left 18 yard box meets pitch boundary at the bottom to left 18 yard box bottom
                (15, 18),  # top center of pitch to bottom center
                (28, 20),  # right 18 yard box meets pitch boundary at the top to right 18 yard box top
                (20, 21),
                (21, 22),
                (22, 23),
                (27, 28),
                (28, 29),
                (29, 25),  # right 6 yard box meets pitch boundary at the top to right 6 yard box top
                (30, 26),  # right 6 yard box meets pitch boundary at the bottom to right 6 yard box bottom
                (25, 26),  # right 6 yard box top to bottom
                (29, 30),  # right 6 yard box meets pitch boundary at the top to bottom
                (30, 31),
                (31, 32),
                (31, 23),  # right 18 yard box meets pitch boundary at the bottom to right 18 yard box bottom

            ]

            # Launch the mapping dialog over the full frame list:
            dlg = LandmarkMappingDialog(
                frame_pixmaps=frames,
                segment=segment,
                landmarks=landmarks,
                edges=edges,
                parent=self
            )
            if dlg.exec_() != QDialog.Accepted:
                return None, None

            # Check mapped points
            mapped = dlg.get_mapped_points()
            if len(mapped) >= 4:
                return {lid: (x, y, 1.0) for lid, (x, y) in mapped.items()}, set(mapped)

            # If <4, on_confirm() inside the dialog already forced browsing,
            # so check again:
            mapped = dlg.get_mapped_points()
            if len(mapped) >= 4:
                return {lid: (x, y, 1.0) for lid, (x, y) in mapped.items()}, set(mapped)

            # Still not enough
            return None, None

        except Exception:
            traceback.print_exc()
            return {}, set()

    def compute_homography_from_keypoints(self, filtered_kps, id_to_world):
        if filtered_kps is None or len(filtered_kps) < 4:
            print("[HOMO] Not enough keypoints for homography. Got:",
                  None if filtered_kps is None else list(filtered_kps.keys()))
            return None

        img_pts = []
        world_pts = []
        seen = set()
        for pid, (x, y, conf) in filtered_kps.items():
            if pid not in id_to_world:
                print(f"[HOMO] Keypoint ID {pid} not in id_to_world mapping, skipping.")
                continue
            # Round to 1 decimal to avoid floating point noise
            key = (round(x, 1), round(y, 1))
            if key in seen:
                continue  # Skip duplicate image points
            seen.add(key)
            img_pts.append([x, y])
            world_pts.append(list(id_to_world[pid]))

        print(f"[HOMO] Using {len(img_pts)} keypoints for homography.")
        print(f"[HOMO] img_pts: {img_pts}")
        print(f"[HOMO] world_pts: {world_pts}")

        if len(img_pts) < 4 or len(world_pts) < 4:
            print("[HOMO] Still not enough valid keypoints after filtering.")
            return None

        img_pts = np.array(img_pts, dtype=np.float32)
        world_pts = np.array(world_pts, dtype=np.float32)

        try:
            H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 3.0)
            if H is None:
                print("[HOMO] cv2.findHomography failed to compute a matrix.")
            else:
                print(f"[HOMO] Homography computed successfully. H;{H}")
            return H
        except Exception as e:
            print(f"[HOMO] Exception during cv2.findHomography: {e}")
            return None

    def sample_segment_frames(self, video_path, start_time, end_time, num_frames=5):
        """Sample num_frames evenly spaced frames between start_time and end_time."""
        cap = cv2.VideoCapture(video_path)
        duration = end_time - start_time
        times = [start_time + i * duration / (num_frames - 1) for i in range(num_frames)]
        frames = []
        for t in times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames.append(frame)
        cap.release()
        return frames

    def extract_frame(self, video_path, timestamp):
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ret, frame = cap.read()
        cap.release()
        if not ret or frame is None:
            return None
        return frame  # BGR numpy array


    def bgr_to_pixmap(self, bgr_img):
        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_img.shape
        bytes_per_line = ch * w
        qimg = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def project_point(self, pt, H):
        """
        Project a point from image coordinates to pitch coordinates using homography.

        Args:
            pt: Dictionary with 'x' and 'y' keys representing image coordinates
            H: 3x3 homography matrix (numpy array)

        Returns:
            Tuple (x, y) of pitch coordinates in meters, or None if projection fails
        """
        if pt is None or H is None:
            return None

        # Ensure pt has the required keys
        if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
            print(f"[PROJECT] Invalid point format: {pt}")
            return None

        try:
            # Create homogeneous coordinates
            pt_arr = np.array([pt['x'], pt['y'], 1.0], dtype=np.float32)

            # Apply homography transformation
            pitch_pt = H @ pt_arr
            print(f"[DEBUG] Pixel pt={pt_arr.tolist()} → raw_homog={pitch_pt.tolist()}")

            # Check for division by zero (points at infinity)
            if abs(pitch_pt[2]) < 1e-8:
                print(f"[PROJECT] Point at infinity: {pitch_pt}")
                return None

            # Convert from homogeneous to Cartesian coordinates
            x = pitch_pt[0] / pitch_pt[2]
            y = pitch_pt[1] / pitch_pt[2]

            print(f"[DEBUG] World coords: ({x:.3f}, {y:.3f}) from pixel ({pt['x']:.1f}, {pt['y']:.1f})")

            # Define pitch boundaries (in meters)
            # Standard soccer pitch: 100-130m length, 50-100m width
            # Using 120m x 80m as standard with 2m goal depth extension
            MIN_X, MAX_X = -2.0, 122.0  # Allow 2m behind goal line
            MIN_Y, MAX_Y = -2.0, 82.0  # Allow 2m outside sidelines

            # Tolerance for clamping near-boundary points
            TOLERANCE = 1.0  # 1 meter tolerance

            # Check if point is within extended boundaries
            if MIN_X - TOLERANCE <= x <= MAX_X + TOLERANCE and MIN_Y - TOLERANCE <= y <= MAX_Y + TOLERANCE:
                # Clamp to valid pitch boundaries if slightly outside
                x = max(MIN_X, min(x, MAX_X))
                y = max(MIN_Y, min(y, MAX_Y))

                return (float(x), float(y))
            else:
                print(f"[PROJECT] Point outside valid pitch area: ({x:.2f}, {y:.2f})")
                return None

        except Exception as e:
            print(f"[PROJECT] Error projecting point {pt}: {e}")
            return None

    def show_processing_progress(self, message="Processing...", max_value=100):
        dlg = QProgressDialog(message, None, 0, max_value, self)
        dlg.setWindowTitle("Please Wait")
        dlg.setWindowModality(Qt.ApplicationModal)
        dlg.setCancelButton(None)
        dlg.setMinimumDuration(0)
        dlg.setValue(0)
        dlg.show()
        QApplication.processEvents()
        return dlg

    def retry_pending_segments_with_homography(self, fallback_H):
        """Try to process all segments that previously failed homography using the new fallback_H."""
        if fallback_H is None or not isinstance(fallback_H, np.ndarray):
            return
        still_pending = []
        for entry in self.segments_pending_homography:
            seg = entry["segment"]
            frames = entry["frames"]
            try:
                start_frame = frames[0]
                end_frame = frames[-1]
                start_pixmap = self.bgr_to_pixmap(start_frame)
                end_pixmap = self.bgr_to_pixmap(end_frame)
                dialog = PointAnnotationDialog(start_pixmap, "Mark ball position at Shot Start:", seg.start, self)
                if dialog.exec_() == QDialog.Accepted and dialog.annotation:
                    start_ball_pos = dialog.annotation
                else:
                    still_pending.append(entry)
                    continue
                dialog = PointAnnotationDialog(end_pixmap, "Mark final ball position after shot:", seg.end, self)
                if dialog.exec_() == QDialog.Accepted and dialog.annotation:
                    end_ball_pos = dialog.annotation
                else:
                    still_pending.append(entry)
                    continue
                start_pitch = self.project_point(start_ball_pos, fallback_H)
                end_pitch = self.project_point(end_ball_pos, fallback_H)
                if start_pitch and end_pitch:
                    xg_val = self.xg_model.predict(start_pitch, end_pitch)
                    if xg_val is not None:
                        self.show_xg_result(xg_val, start_pitch, end_pitch)
                        self.show_shot_transition_for_retried_segment(seg, start_pitch, end_pitch)
                        seg_data = {
                            "start_time": seg.start,
                            "end_time": seg.end,
                            "homography": fallback_H.tolist(),
                            "start_ball_pos": start_ball_pos,
                            "end_ball_pos": end_ball_pos,
                            "start_pitch_pos": start_pitch,
                            "end_pitch_pos": end_pitch,
                            "retried": True
                        }
                        if not hasattr(self, "segment_analysis_results"):
                            self.segment_analysis_results = []
                        self.segment_analysis_results.append(seg_data)
                    else:
                        self.show_error_message("Could not compute expected goals for a previously failed segment.")
                        still_pending.append(entry)
                else:
                    self.show_error_message("Could not project ball positions for a previously failed segment.")
                    still_pending.append(entry)
            except Exception as e:
                print(f"Error retrying segment: {e}")
                still_pending.append(entry)

        old_len = len(self.segments_pending_homography)
        self.segments_pending_homography = still_pending
        if len(self.segments_pending_homography) < old_len:
            self.update_status_text()

    def retry_pending_segments_with_direct_pitch(self):
        still_pending = []
        for entry in self.segments_pending_direct_pitch:
            seg = entry["segment"]
            frames = entry["frames"]
            if not frames or frames[0] is None or frames[-1] is None:
                print("[ERROR] No valid frames for direct pitch annotation.")
                still_pending.append(entry)
                continue
            start_frame_pixmap = self.bgr_to_pixmap(frames[0])
            end_frame_pixmap = self.bgr_to_pixmap(frames[-1])
            if start_frame_pixmap is None or start_frame_pixmap.isNull():
                print("[ERROR] Invalid start frame pixmap.")
                still_pending.append(entry)
                continue
            if end_frame_pixmap is None or end_frame_pixmap.isNull():
                print("[ERROR] Invalid end frame pixmap.")
                still_pending.append(entry)
                continue
            try:
                start_dlg = DirectPitchAnnotationDialog(
                    start_frame_pixmap, marker_color=Qt.blue, marker_label="Shot Start", parent=self
                )
            except Exception as e:
                print(f"[ERROR] Failed to create DirectPitchAnnotationDialog: {e}")
                still_pending.append(entry)
                continue
            if start_dlg.exec_() != QDialog.Accepted:
                still_pending.append(entry)
                continue
            start_pitch = start_dlg.get_absolute_pitch_coordinates()
            try:
                end_dlg = DirectPitchAnnotationDialog(
                    end_frame_pixmap, marker_color=Qt.red, marker_label="Shot End", parent=self
                )
            except Exception as e:
                print(f"[ERROR] Failed to create DirectPitchAnnotationDialog: {e}")
                still_pending.append(entry)
                continue
            if end_dlg.exec_() != QDialog.Accepted:
                still_pending.append(entry)
                continue
            end_pitch = end_dlg.get_absolute_pitch_coordinates()
            self.process_segment_with_pitch_positions(seg, start_pitch, end_pitch)
        self.segments_pending_direct_pitch = still_pending

    def attempt_direct_pitch(self, seg, frames):
        """
        Immediately ask user to do direct pitch annotation (start/end).
        Show high-quality RGB reference frames (start frame and end frame) in a scrollable view.
        Compute xG immediately after both pitch points are supplied (no ball annotation step).
        Returns True if processed, False if user cancelled or failed.
        """
        try:
            # frames is expected to be a list-like of QPixmap / QImage / or objects convertible to QPixmap.
            start_frame_pixmap = None
            end_frame_pixmap = None
            if frames:
                # prefer first and last frame provided
                try:
                    start_frame_pixmap = self.bgr_to_pixmap(frames[0])
                    end_frame_pixmap =  self.bgr_to_pixmap(frames[-1])
                except Exception:
                    # defensive fallback if frames is not indexable
                    pass

            # Ask for start pitch using a direct-pitch dialog that shows the start reference frame
            start_dlg = DirectPitchAnnotationDialog(frame_pixmap=start_frame_pixmap,
                                                    marker_color=Qt.green,
                                                    marker_label="Shot Start",
                                                    parent=self)
            if start_dlg.exec_() != QDialog.Accepted:
                # user cancelled start mapping
                return False
            start_pitch = start_dlg.get_absolute_pitch_coordinates()
            if not start_pitch:
                # user accepted dialog but did not place a pitch marker
                self.show_error_message("Start pitch marker not provided.")
                return False

            # Ask for end pitch using a direct-pitch dialog that shows the end reference frame
            end_dlg = DirectPitchAnnotationDialog(frame_pixmap=end_frame_pixmap,
                                                  marker_color=Qt.red,
                                                  marker_label="Shot End",
                                                  parent=self)
            if end_dlg.exec_() != QDialog.Accepted:
                # user cancelled end mapping
                return False
            end_pitch = end_dlg.get_absolute_pitch_coordinates()
            if not end_pitch:
                self.show_error_message("End pitch marker not provided.")
                return False

            self.process_segment_with_pitch_positions(seg, start_pitch, end_pitch)

            return True

        except Exception as e:
            print(f"[ERROR] attempt_direct_pitch_now error: {e}")
            return False


    def process_segment_with_pitch_positions(self, seg, start_pitch, end_pitch):
        """
        Given the segment and start/end pitch coordinates (typically in pitch meters or the
        normalized coordinate system you use), compute xG immediately, show the xG UI
        and the shot transition visualization, and save the result in analysis structures.
        """
        try:

            xg_val = self.xg_model.predict(start_pitch, end_pitch)
            self.show_xg_result(xg_val, start_pitch, end_pitch)
            self.show_shot_transition_for_retried_segment(seg, start_pitch, end_pitch)

            seg_data = {
                "start_time": seg.start,
                "end_time": seg.end,
                "homography": None,
                "start_ball_pos": None,
                "end_ball_pos": None,
                "start_pitch_pos": start_pitch,
                "end_pitch_pos": end_pitch,
                "retried": True,
                "direct_pitch_annotation": True
            }
            if not hasattr(self, "segment_analysis_results"):
                self.segment_analysis_results = []
            self.segment_analysis_results.append(seg_data)

            # Update status UI / counters
            self.update_status_text()

        except Exception as e:
            print(f"[ERROR] process_segment_with_pitch_positions: {e}")
            self.show_error_message("Failed to process segment with direct pitch positions.")

    def process_confirmed_segment(self, seg):
        progress_dlg = self.show_processing_progress("Analyzing shot segment...", max_value=100)
        try:
            # 1. Sample frames
            progress_dlg.setLabelText("Extracting frames...")
            progress_dlg.setValue(20)
            QApplication.processEvents()
            frames = self.sample_segment_frames(self.fileName, seg.start, seg.end, num_frames=9)
            if not frames:
                self.show_error_message("Could not extract frames for this segment.")
                return

            # 2. Manual keypoint detection (user maps landmarks directly)
            progress_dlg.setLabelText("Manual landmark mapping...")
            progress_dlg.setValue(40)
            QApplication.processEvents()

            keypoints, keypoint_ids = self.detect_pitch_keypoints_with_fallback(
                frames=frames,
                id_to_world=self.id_to_world,
                id_to_landmark=self.id_to_landmark,
                left_side_ids=self.left_side_ids,
                right_side_ids=self.right_side_ids,
                segment=seg
            )

            # Normalize just-in-case (detect_pitch_keypoints_with_fallback now returns {} and set())
            if keypoints is None:
                keypoints = {}
            if keypoint_ids is None:
                keypoint_ids = set()

            num_points = len(keypoint_ids or set())

            # --- CASE A: user mapped nothing / closed the dialog (0 points) ---
            if num_points == 0:
                # Let the user retry mapping repeatedly if they wish.
                # Loop until we get >=4 points, partial mapping (2-3), or user elects to go to direct-pitch / defer.
                while True:
                    resp = self._show_confirm_dialog(
                        "Landmark mapping closed",
                        "No landmarks were mapped. Reopen mapping dialog to try again?"
                    )

                    if resp != QMessageBox.Yes:
                        # User explicitly declines to reopen mapping. Offer direct pitch now.
                        proceeded = self.attempt_direct_pitch(seg, frames)
                        if proceeded:
                            return
                        else:
                            # User cancelled direct-pitch -> defer for later
                            self.segments_pending_direct_pitch.append({"segment": seg, "frames": frames})
                            self.show_error_message("Direct pitch annotation deferred to later.")
                            self.update_status_text()
                            print(f"[DEBUG] Pending direct pitch count: {len(self.segments_pending_direct_pitch)}")
                            return

                    # User chose to reopen mapping
                    keypoints2, keypoint_ids2 = self.detect_pitch_keypoints_with_fallback(
                        frames=frames,
                        id_to_world=self.id_to_world,
                        id_to_landmark=self.id_to_landmark,
                        left_side_ids=self.left_side_ids,
                        right_side_ids=self.right_side_ids,
                        segment=seg
                    )

                    keypoints2 = keypoints2 or {}
                    keypoint_ids2 = keypoint_ids2 or set()
                    kcount = len(keypoint_ids2)

                    if kcount >= 4:
                        # success — fall through to homography path below
                        keypoints, keypoint_ids = keypoints2, keypoint_ids2
                        break

                    if kcount == 0:
                        # Either user closed dialog again or detection returned 0.
                        # Ask whether to try again or go to direct pitch.
                        retry_resp = self._show_confirm_dialog(
                            "Still no landmarks",
                            "Still no landmarks were mapped. Try mapping again?"
                        )
                        if retry_resp == QMessageBox.Yes:
                            continue
                        else:
                            # user chose no -> proceed to direct pitch attempt
                            proceeded = self.attempt_direct_pitch_now(seg, frames)
                            if proceeded:
                                return
                            else:
                                self.segments_pending_direct_pitch.append({"segment": seg, "frames": frames})
                                self.show_error_message("Direct pitch annotation deferred to later.")
                                self.update_status_text()
                                print(f"[DEBUG] Pending direct pitch count: {len(self.segments_pending_direct_pitch)}")
                                return

                    # Partial mapping (2-3)
                    if kcount in (2, 3):
                        # Save partial mapping for retry later so the user or an automated process can continue.
                        self.segments_pending_homography.append({
                            "segment": seg,
                            "frames": frames,
                            "keypoints": keypoints2,
                            "keypoint_ids": keypoint_ids2
                        })
                        self.show_error_message("Partial landmark mapping (2-3 points) saved for retry.")
                        print(f"[DEBUG] Pending homography count: {len(self.segments_pending_homography)}")
                        return



            # --- CASE B: 2 or 3 points mapped (insufficient for homography) ---
            if num_points in (2, 3):
                # Save partial mapping for retry later
                self.segments_pending_homography.append({
                    "segment": seg,
                    "frames": frames,
                    "keypoints": keypoints,
                    "keypoint_ids": keypoint_ids
                })
                self.show_error_message(
                    "Partial landmark mapping (2-3 points) saved for retry. xG will be computed when more landmarks are available.")
                return

            # 3. Compute homography directly from manual keypoints
            progress_dlg.setLabelText("Computing pitch mapping...")
            progress_dlg.setValue(60)
            QApplication.processEvents()

            homography = self.compute_homography_from_keypoints(keypoints, self.id_to_world)
            if homography is None:
                self.segments_pending_homography.append({
                    "segment": seg,
                    "frames": frames,
                    "keypoints": keypoints,
                    "keypoint_ids": keypoint_ids
                })
                self.show_error_message("Could not compute pitch mapping. xG will be computed later.")
                return

            # Store homography for future use
            self.homography_history.append({
                "start_time": seg.start,
                "end_time": seg.end,
                "homography": homography
            })

            # 4. Ball position annotation
            progress_dlg.setLabelText("Annotating ball positions...")
            progress_dlg.setValue(75)
            QApplication.processEvents()

            # Use first and last frames for ball annotation
            start_pixmap = self.bgr_to_pixmap(frames[0])
            end_pixmap = self.bgr_to_pixmap(frames[-1])

            # Annotate start ball position
            start_ball_pos = None
            while not start_ball_pos:
                dlg = PointAnnotationDialog(start_pixmap,
                                            "Mark ball position at Shot Start:",
                                            seg.start,
                                            self)
                if dlg.exec_() == QDialog.Accepted and dlg.annotation:
                    start_ball_pos = dlg.annotation
                else:
                    self.show_error_message("You must annotate the ball position at shot start to proceed.")

            # Annotate end ball position
            end_ball_pos = None
            while not end_ball_pos:
                dlg = PointAnnotationDialog(end_pixmap,
                                            "Mark final ball position after shot:",
                                            seg.end,
                                            self)
                if dlg.exec_() == QDialog.Accepted and dlg.annotation:
                    end_ball_pos = dlg.annotation
                else:
                    self.show_error_message("You must annotate the ball position at shot end to proceed.")

            # 5. Project ball positions to pitch coordinates and compute xG
            progress_dlg.setLabelText("Computing xG...")
            progress_dlg.setValue(90)
            QApplication.processEvents()

            start_pitch = self.project_point(start_ball_pos, homography)
            end_pitch = self.project_point(end_ball_pos, homography)

            if start_pitch and end_pitch:
                xg_val = self.xg_model.predict(start_pitch, end_pitch)
                if xg_val is not None:
                    self.show_xg_result(xg_val, start_pitch, end_pitch)
                    # Show shot transition visualization
                    self.show_shot_transition(seg)

                    # Try to process any previously failed segments using this homography
                    self.retry_pending_segments_with_homography(homography)
                    self.retry_pending_segments_with_direct_pitch()
                else:
                    self.show_error_message("Could not compute expected goals for this segment.")
            else:
                self.show_error_message("Ball positions could not be projected to pitch coordinates.")

            # 6. Save results
            seg_data = {
                "start_time": seg.start,
                "end_time": seg.end,
                "keypoints": {k: (v[0], v[1]) for k, v in keypoints.items()},  # Remove confidence values
                "homography": homography.tolist(),
                "start_ball_pos": start_ball_pos,
                "end_ball_pos": end_ball_pos,
                "start_pitch_pos": start_pitch,
                "end_pitch_pos": end_pitch
            }

            # Store results
            if not hasattr(self, "segment_analysis_results"):
                self.segment_analysis_results = []
            self.segment_analysis_results.append(seg_data)

            progress_dlg.setLabelText("Complete!")
            progress_dlg.setValue(100)
            QApplication.processEvents()

        except Exception as e:
            self.show_error_message(f"An error occurred during processing: {str(e)}")
            print(f"Error in process_confirmed_segment: {e}")

            traceback.print_exc()
        finally:
            progress_dlg.close()
        self.update_status_text()

    def show_xg_result(self, xg_val, start_pitch, end_pitch):
        # start_pitch, end_pitch = self.ensure_attacking_direction(start_pitch, end_pitch)
        print(f"[DEBUG] Showing xG plot: xg_val={xg_val}, start_pitch={start_pitch}, end_pitch={end_pitch}")
        dlg = XGShotPlotDialog(start_pitch, end_pitch, xg_val, self)
        dlg.exec_()

    def confirm_segment(self, seg: VideoSegment):
        if QThread.currentThread() != self.thread():
            print("[DEBUG] Attempted to confirm segment from non-UI thread!")
            return

        print("[DEBUG] Starting confirm_segment for segment", seg)

        # Stop any active preview
        if self.video_player.is_previewing:
            print("[DEBUG] Stopping preview")
            self.video_player.stop()
            self.video_player.is_previewing = False
            self.timeline.previewing_segment = None

        self.timeline.update_segment_state(seg, SegmentState.CONFIRMED)
        self.timeline.update()
        # Save the confirmed data
        confirmed_data = {
            'start': seg.start,
            'end': seg.end,
            'duration': seg.duration
        }
        self.confirmed_segments.append(confirmed_data)
        # Hide preview popup if any
        if hasattr(self.window(), 'preview_popup'):
            self.window().preview_popup.hide_preview()

        progress_dlg = self.show_processing_progress("Analyzing shot segment...")
        try:
            self.process_confirmed_segment(seg)
            self.show_shot_transition(seg)
        finally:
            progress_dlg.close()
        self.check_review_complete()

    def update_status_text(self):
        confirmed = len(self.confirmed_segments)
        rejected = sum(1 for t in self.shot_timestamps if t.get('state') == 'rejected')
        pending_timeline = len([s for s in self.timeline.segments if s.state == SegmentState.PENDING])
        pending_queue = len(self.pendingQueue)
        total = self.total_shots

        status_text = (
            f"Shot Review Progress:\n"
            f"Confirmed: {confirmed}\n"
            f"Rejected: {rejected}\n"
            f"Pending (To Be Displayed): {pending_timeline}\n"
            f"Queued: {pending_queue}\n"
            f"Total: {total}"
        )
        self.textedit.setText(status_text)

        # If there are pending segments, show and enable the review button.
        if pending_timeline > 0 or pending_queue > 0:
            self.review_btn.show()
            self.review_btn.setEnabled(True)
        else:
            # Otherwise, keep it hidden and disabled.
            self.review_btn.hide()
            self.review_btn.setEnabled(False)

    def show_fine_tuning_dialog(self, seg: VideoSegment, which_boundary: str):
        dialog = FineTuneDialog(self.window(), seg, which_boundary)
        result = dialog.exec_()
        if result == QDialog.Accepted:
            # Fine tuning confirmed; update segment state accordingly.
            print(f"{which_boundary.capitalize()} boundary fine-tuned to:",
                  seg.start if which_boundary == "start" else seg.end)
            # animate the boundary change.
            self.timeline.animate_boundary_change(seg)
        else:
            # Canceled; the reject() in the dialog will revert the changes.
            print("Fine tuning canceled.")

    def create_video_panel(self):
        panel = QFrame()
        panel.setStyleSheet("""
            QFrame {
                background-color: blanchedalmond;
                border: 1px solid black;
            }
        """)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(5, 5, 5, 5)
        panel_layout.setSpacing(0)

        # --- 1) VIDEO CONTAINER ---
        video_container = QWidget()
        video_container.setStyleSheet("background-color: black;")
        video_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)
        video_layout.setSpacing(0)

        # Create the VideoPlayer
        self.video_player = VideoPlayer(video_container)
        video_layout.addWidget(self.video_player, stretch=1)

        # Add the video container as the TOP row
        panel_layout.addWidget(video_container, stretch=70)

        # --- 2) TIMELINE CONTAINER ---
        # Create the timeline
        self.timeline = VideoTimeline()
        self.video_player.set_timeline(self.timeline)
        self.timeline.segmentClicked.connect(self.handle_segment_click)
        self.timeline.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.timeline.setMinimumHeight(100)

        # Put timeline in its own container
        self.timeline_container = QWidget()
        self.timeline_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

        timeline_layout = QVBoxLayout(self.timeline_container)
        timeline_layout.setContentsMargins(8, 8, 8, 8)
        timeline_layout.addWidget(self.timeline)

        # Optionally hide timeline until "Review Shots" is clicked
        self.timeline_container.hide()

        # Add timeline container as the BOTTOM row
        panel_layout.addWidget(self.timeline_container, stretch=30)

        return panel

    def create_media_controls(self):
        panel = QWidget()
        panel.setMinimumHeight(80)
        layout = QHBoxLayout(panel)
        layout.setSpacing(10)
        layout.setContentsMargins(20, 10, 20, 10)
        control_buttons = ['Play', 'Pause', 'Stop', 'Rewind', 'Forward']
        button_style = """
            QPushButton {
                background-color: blanchedalmond;
                font: bold 15.75pt 'Segoe Print';
                color: saddlebrown;
                border: 1px solid saddlebrown;
                border-radius: 5px;
                padding: 5px 15px;
            }
            QPushButton:hover {
                background-color: #fff5ee;
                color: darkgreen;
            }
        """
        self.media_buttons = {}
        for bt in control_buttons:
            btn = QPushButton(bt)
            btn.setObjectName(bt)
            btn.setStyleSheet(button_style)
            layout.addWidget(btn)
            self.media_buttons[bt] = btn
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)
        self.volume_slider.valueChanged.connect(self.update_volume_label)
        self.volume_label = QLabel("Volume: 100%")
        self.volume_slider.setEnabled(False)
        layout.addWidget(self.volume_slider)
        self.volume_label.setFont(QFont('Segoe Script', 12))
        self.volume_label.setStyleSheet("color: peru;")
        layout.addWidget(self.volume_label)
        return panel

    def update_volume_label(self, val):
        if self.video_player.player:
            self.video_player.player.audio_set_volume(val)
        self.volume_label.setText(f"Volume: {val}%")

    @pyqtSlot(tuple)
    def handle_new_segment(self, seg_tuple):
        """
        Enqueue each 'ontarget' segment as soon as it's detected.
        """
        start, end = seg_tuple
        print(f"[DEBUG] Queuing new segment: {start:.2f}-{end:.2f}")
        vs = VideoSegment(start, end, SegmentState.PENDING)

        # 1) Append to the review queue
        self.pendingQueue.append(vs)

        # 2) Mirror in shot_timestamps for the status text
        self.shot_timestamps.append({'start': start, 'end': end, 'state': 'pending'})
        self.total_shots = len(self.shot_timestamps)

        # 3) Enable the Review button if needed
        if not self.review_btn.isEnabled():
            self.review_btn.setEnabled(True)

        # 4) Refresh the left-pane stats
        self.update_status_text()

    def start_processing(self):
        self.processingComplete = False
        model_path = "movinet_a3train-size_70.0 _test-size_30.0_test accuracy_97.46_batch(1)_frames(8)_ops_(fp16)_bundle_input_init_states(False).tflite"
        self.process_btn.show_progress()
        QApplication.processEvents()
        self.recognition_worker = ActionRecognitionWorker(self.fileName, model_path)
        self.recognition_worker.progress_update.connect(self.update_progress)
        self.recognition_worker.segment_detected.connect(self.handle_new_segment)
        self.recognition_worker.complete.connect(self.handle_recognition_complete)
        self.recognition_worker.error.connect(self.handle_recognition_error)
        print("[DEBUG] Starting recognition worker thread")
        self.recognition_worker.start()

    def add_detected_segment(self, seg_tuple):
        start, end = seg_tuple
        print(f"[DEBUG] New segment: start={start:.2f}, end={end:.2f}, video_duration={self.video_player.duration:.2f}")
        video_segment = VideoSegment(start, end, SegmentState.PENDING)
        # If there is already a pending segment on the timeline, queue this new one.
        if any(s.state == SegmentState.PENDING for s in self.timeline.segments):
            self.pendingQueue.append(video_segment)
            print("[DEBUG] Segment queued; pendingQueue length:", len(self.pendingQueue))
        else:
            self.timeline.add_segment(video_segment)
        self.shot_timestamps.append({'start': start, 'end': end, 'state': 'pending'})
        self.total_shots += 1
        self.update_status_text()

    def process_pending_queue(self):
        # Only load a new segment if the timeline is empty and there is a queued segment.
        if not self.timeline.segments and self.pendingQueue:
            next_seg = self.pendingQueue.pop(0)
            self.timeline.add_segment(next_seg)
            print("[DEBUG] Loaded next pending segment from queue. Remaining:", len(self.pendingQueue))
        # Update the status which will decide whether to show/enable the review button.
        self.update_status_text()

    def update_progress(self, percent, message):
        print(f"Progress: {percent}% - {message}")
        self.process_btn.update_progress(percent, message)
        # Force the UI to refresh
        QApplication.processEvents()

    def handle_recognition_complete(self, ontarget_segments):
        """
        When the worker finishes, just mark "done" and update the button.
        Any segments already enqueued by handle_new_segment stay in place.
        """
        print("[DEBUG] Processing complete, final segments:", ontarget_segments)
        self.processingComplete = True

        # In case the model found any that didn't fire live, enqueue them:
        for start, end in ontarget_segments:
            # skip duplicates
            if not any(abs(s['start'] - start) < 1e-3 and abs(s['end'] - end) < 1e-3
                       for s in self.shot_timestamps):
                self.handle_new_segment((start, end))

        # Update the "Process" button visual
        self.process_btn.complete(True)

        # Ensure "Review Shots" is enabled if there's anything to do
        self.review_btn.setEnabled(bool(self.pendingQueue))
        self.update_status_text()

    def handle_recognition_error(self, error_message):
        print("Error:", error_message)
        QMessageBox.critical(self, "Action Recognition Error", error_message)
        self.process_btn.complete(False)

    def _call_if_video_ready(self, fn, *args, **kwargs):
        try:
            if getattr(self, "video_player", None) is None:
                try:
                    self.statusBar().showMessage("Video not loaded", 1200)
                except Exception:
                    pass
                return
            return fn(*args, **kwargs)
        except Exception as e:
            print(f"Shortcut handler failed: {e}")

    def _shortcut_play_pause(self):
        # This will call self.video_player.play_pause_toggle() only if video_player exists
        self._call_if_video_ready(lambda: self.video_player.play_pause_toggle())

    def _shortcut_seek(self, seconds):
        self._call_if_video_ready(lambda: self.video_player.seek_relative(seconds))

    def _shortcut_confirm_segment(self):
        try:
            self.confirm_segment()
        except Exception:
            self._call_if_video_ready(lambda: self.video_player.confirm_segment())

    def _shortcut_skip_segment(self):
        try:
            self.skip_segment()
        except Exception:
            self._call_if_video_ready(lambda: self.video_player.skip_segment())

    def _shortcut_preview_segment(self):
        try:
            self.preview_segment()
        except Exception:
            self._call_if_video_ready(lambda: self.video_player.preview_segment())

    def setup_shortcuts(self):
        """Create application-wide shortcuts. Uses video_player.rewind/forward directly."""
        # cleanup any previously created shortcuts (safe to call multiple times)
        if hasattr(self, "_shortcuts") and self._shortcuts:
            for sc in self._shortcuts:
                try:
                    sc.setParent(None)
                except Exception:
                    pass
        self._shortcuts = []

        def _wrap(name, fn):
            """Wrap handler to give a console trace and swallow handler errors."""

            def _callable():
                try:
                    print(f"[SHORTCUT] {name} triggered")
                    return fn()
                except Exception as e:
                    print(f"[SHORTCUT ERROR] {name}: {e}")

            return _callable

        # Map of key sequence -> zero-arg callable
        mapping = {
            # fullscreen / window
            'F11': _wrap('F11', self.toggle_fullscreen),
            'F': _wrap('F', self.toggle_fullscreen),
            'Esc': _wrap('Esc', self.exit_fullscreen),

            # playback toggle (assumes _shortcut_play_pause is zero-arg and safe)
            'Space': _wrap('Space', self._shortcut_play_pause),

            # SEEKING — use video_player.rewind/forward directly.
            # Assumes signatures: rewind(small_step=True/False) and forward(small_step=True/False)
            # Left/Right = small step; Ctrl+Left/Ctrl+Right = large step
            'Left': _wrap('Left', partial(self.video_player.rewind, small_step=True)),
            'Right': _wrap('Right', partial(self.video_player.forward, small_step=True)),
            'Ctrl+Left': _wrap('Ctrl+Left', partial(self.video_player.rewind, small_step=False)),
            'Ctrl+Right': _wrap('Ctrl+Right', partial(self.video_player.forward, small_step=False)),

            # audio / misc (change to bound method if you prefer no lambda)
            'M': _wrap('M', lambda: self.video_player.toggle_mute() if getattr(self, 'video_player', None) else None),
            'Up': _wrap('Up', self.volume_up),
            'Down': _wrap('Down', self.volume_down),

            # window snap (Meta is platform-dependent; you can duplicate with Ctrl if needed)
            'Meta+Left': _wrap('Meta+Left', self.snap_left),
            'Meta+Right': _wrap('Meta+Right', self.snap_right),
        }

        # Review controls — connect bound methods directly (zero-arg callables)
        mapping.update({
            'C': _wrap('C (confirm)', self._shortcut_confirm_segment),
            'S': _wrap('S (skip)', self._shortcut_skip_segment),
            'P': _wrap('P (preview)', self._shortcut_preview_segment),
        })

        # create QShortcut objects and attach them
        for seq_text, handler in mapping.items():
            try:
                sc = QShortcut(QKeySequence(seq_text), self)
                sc.setContext(Qt.ApplicationShortcut)
                sc.activated.connect(handler)
                self._shortcuts.append(sc)
            except Exception as e:
                print(f"[SHORTCUT SETUP ERROR] {seq_text}: {e}")

    def volume_up(self):
        current = self.volume_slider.value()
        self._update_volume_label(self.volume_slider.setValue(min(100, current + 5)))

    def volume_down(self):
        current = self.volume_slider.value()
        self._update_volume_label(self.volume_slider.setValue(max(0, current - 5)))

    def snap_left(self):
        screen = QApplication.desktop().screenGeometry()
        self.setGeometry(0, 0, screen.width() // 2, screen.height())

    def snap_right(self):
        screen = QApplication.desktop().screenGeometry()
        self.setGeometry(screen.width() // 2, 0, screen.width() // 2, screen.height())

    def save_window_state(self):
        self.settings.setValue('window_geometry', self.saveGeometry())
        self.settings.setValue('window_state', self.saveState())
        self.settings.setValue('volume', self.volume_slider.value())

    def restore_window_state(self):
        if self.settings.value('window_geometry'):
            self.restoreGeometry(self.settings.value('window_geometry'))
        if self.settings.value('window_state'):
            self.restoreState(self.settings.value('window_state'))
        vol = self.settings.value('volume', 100, type=int)
        self.volume_slider.setValue(vol)

    def closeEvent(self, event):
        self.save_window_state()
        if hasattr(self, 'recognition_worker') and self.recognition_worker.isRunning():
            # ask it nicely to stop…
            self.recognition_worker.requestInterruption()
            # …and block until it really finishes
            self.recognition_worker.wait()
        # now it's safe to tear down VLC, etc.
        if hasattr(self, 'video_player') and self.video_player:
            self.video_player.close_video()
        if hasattr(self, 'timeline') and self.timeline and self.timeline.cap:
            self.timeline.cap.release()
            self.timeline.clear_capture()
        event.accept()

    def delayed_resize(self):
        width = self.width()
        height = self.height()
        base_font_size = min(width, height) // 50
        self.header.setFont(QFont('Segoe Script', max(12, base_font_size)))
        margin = min(width, height) // 100
        self.main_layout.setContentsMargins(margin, margin, margin, margin)
        self.main_layout.setSpacing(margin)

    def resetUI(self):
        # 1) Hide timeline container, clear timeline
        self.timeline_container.hide()
        self.timeline.clear_all_segments()
        self.timeline.clear_thumbnail_cache()
        self.timeline.set_duration(0)
        self.timeline.update()

        self.shot_timestamps = []
        self.total_shots = 0

        self.textedit.clear()

        # 2) Reset button states
        self.upload_btn.setEnabled(True)
        self.process_btn.setEnabled(False)
        self.review_btn.setEnabled(False)

        # 3) Clear progress messages
        self.upload_btn.show_progress(False)
        self.upload_btn.complete(False)
        self.process_btn.show_progress(False)
        self.process_btn.complete(False)
        self.review_btn.show_progress(False)
        self.review_btn.complete(False)

        # 4) Clear the text from each status label
        self.upload_btn.status.setText("")
        self.upload_btn.status.setVisible(False)
        self.process_btn.status.setText("")
        self.process_btn.status.setVisible(False)
        self.review_btn.status.setText("")
        self.review_btn.status.setVisible(False)

        # 5) Hide media panel
        self.media_panel.setVisible(False)

        # Force-hide the preview popup if it's still around
        if hasattr(self, 'preview_popup'):
            self.preview_popup.hide_preview()

    def stop_all_marker_timers(self):
        for seg in self.timeline.segments:
            if seg.start_marker:
                seg.start_marker.dragging = False
                seg.start_marker.pendingTime = None
                if seg.start_marker.dragThrottleTimer.isActive():
                    seg.start_marker.dragThrottleTimer.stop()
            if seg.end_marker:
                seg.end_marker.dragging = False
                seg.end_marker.pendingTime = None
                if seg.end_marker.dragThrottleTimer.isActive():
                    seg.end_marker.dragThrottleTimer.stop()

    def videobutton_Click(self):
        # Clear previous state.
        self.timeline.clear_all_segments()
        self.timeline.clear_thumbnail_cache()
        self.timeline.set_duration(0)
        self.timeline.update()
        self.shot_timestamps = []
        self.textedit.clear()
        self.review_btn.setEnabled(False)
        self.process_btn.setEnabled(False)
        self.stop_all_marker_timers()
        if hasattr(self, 'preview_popup'):
            self.preview_popup.hide_preview()

        # Prompt the user to select a video file.
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov *.webm)"
        )
        if not file_path:
            return  # User cancelled the file dialog.

        self.fileName = file_path

        # Show progress on the upload button.
        self.upload_btn.show_progress(True)
        self.upload_btn.update_progress(0, "Loading video...")
        QApplication.processEvents()  # Force the UI to update the progress bar.

        # Load the video via the VideoPlayer.
        success, duration = self.video_player.load_video(self.fileName)
        if not success:
            QMessageBox.critical(self, "Error", "Could not load video.")
            self.upload_btn.complete(False)
            return

        else:
            # Set the timeline's duration now that we have the video duration.
            self.timeline.set_duration(duration)
            # Set the video file path for thumbnail capture.
            self.timeline.video_file_path = self.fileName
            self.timeline.open_capture_once()

            # Update progress and mark upload as complete.
            self.upload_btn.update_progress(100, "Video Loaded")
            self.upload_btn.complete(True)
            # Disable the upload button to prevent re-upload.
            self.upload_btn.setEnabled(False)

            # Enable the Process Video button.
            self.process_btn.setEnabled(True)

    def toggle_fullscreen(self):
        if self.current_window_state != WindowState.FULLSCREEN:
            self.showFullScreen()
            self.current_window_state = WindowState.FULLSCREEN
        else:
            self.showNormal()
            self.current_window_state = WindowState.NORMAL

    def exit_fullscreen(self):
        if self.current_window_state == WindowState.FULLSCREEN:
            self.showNormal()
            self.current_window_state = WindowState.NORMAL

    def show_error_message(self, msg):
        QMessageBox.critical(self, "Error", msg)


# ---------------------------
# Entry point
# ---------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = xG()
    ex.show()
    sys.exit(app.exec_())
