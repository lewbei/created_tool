import os
import shutil
import json
import logging
import yaml
import threading
import csv
import xml.etree.ElementTree as ET
from datetime import datetime

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk

# Setup logging for global error handling
LOG_FILE = os.path.join(os.getcwd(), "error.log")
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Icon glyphs for toolbar buttons (using simple Unicode emojis)
ICON_UNICODE = {
    'auto_annotate': '⚡',
    'save': '💾',
    'load_model': '📂',
    'export': '📤',
    'mode_box': '⬜',
    'mode_polygon': '🔷',
    'undo': '↶',
    'redo': '↷',
    'zoom_in': '🔍+',
    'zoom_out': '🔍-',
    'shortcuts': '⌨️',
}

# --------------------------------------------------
# Global Constants and Helpers
# --------------------------------------------------

PROJECTS_DIR = os.path.join(os.getcwd(), "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)

def center_window(win, width, height):
    """
    Centers a Tkinter window 'win' given the desired 'width' and 'height'.
    """
    win.update_idletasks()
    screen_width = win.winfo_screenwidth()
    screen_height = win.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    win.geometry(f"{width}x{height}+{x}+{y}")

def write_annotations_to_file(label_path, bboxes, polygons, image_shape):
    """
    Writes bounding boxes (YOLO format) and polygons (normalized points) to a label file.

    :param label_path: Path to the .txt label file.
    :param bboxes: List of bounding boxes [ (x, y, w, h, class_id), ... ] in pixel coords.
    :param polygons: List of polygons [ {'class_id': int, 'points': [(x1, y1), ...]}, ... ] in pixel coords.
    :param image_shape: (height, width) of the image used for normalization.
    """
    img_h, img_w = image_shape[:2]
    with open(label_path, 'w') as label_file:
        # Write bounding boxes
        for x, y, w, h, class_id in bboxes:
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h
            label_file.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")

        # Write polygons in YOLO segmentation format (normalized points)
        for poly_data in polygons:
            class_id = poly_data['class_id']
            points = poly_data['points']
            normalized_points = []
            for px, py in points:
                normalized_points.append(px / img_w)
                normalized_points.append(py / img_h)
            # Format: class_id x1_norm y1_norm x2_norm y2_norm ...
            label_file.write(f"{class_id} {' '.join(map(str, normalized_points))}\n")

def read_annotations_from_file(label_path, image_shape):
    """
    Reads YOLO-format bounding boxes and normalized polygons from a label file
    and converts them to pixel coordinates.

    :param label_path: Path to the .txt label file.
    :param image_shape: (height, width) of the image used for denormalization.
    :return: Tuple (list of bboxes, list of polygons)
             bboxes: [ (x, y, w, h, class_id), ... ] in pixel coords.
             polygons: [ {'class_id': int, 'points': [(x1, y1), ...]}, ... ] in pixel coords.
    """
    bboxes = []
    polygons = []
    if not os.path.exists(label_path):
        return bboxes, polygons

    img_h, img_w = image_shape[:2]
    with open(label_path, 'r') as label_file:
        for line in label_file:
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            coords = parts[1:]

            if len(coords) == 4: # Bounding box (YOLO format: x_center, y_center, width, height)
                x_center, y_center, width, height = coords
                x_center_abs = x_center * img_w
                y_center_abs = y_center * img_h
                width_abs = width * img_w
                height_abs = height * img_h
                x_min = int(x_center_abs - width_abs / 2)
                y_min = int(y_center_abs - height_abs / 2)
                bboxes.append((x_min, y_min, int(width_abs), int(height_abs), class_id))
            elif len(coords) % 2 == 0 and len(coords) >= 6: # Polygon (YOLO segmentation format: x1, y1, x2, y2, ...)
                points = []
                for i in range(0, len(coords), 2):
                    px_norm = coords[i]
                    py_norm = coords[i+1]
                    points.append((int(px_norm * img_w), int(py_norm * img_h)))
                polygons.append({'class_id': class_id, 'points': points})
            # else: ignore malformed lines
    return bboxes, polygons

def copy_files_recursive(file_list_relative_paths, base_images_src_dir, images_dst_base, base_labels_src_dir, labels_dst_base):
    """
    Copies images and their corresponding label files, preserving subdirectory structure.

    :param file_list_relative_paths: List of image file paths relative to base_images_src_dir.
    :param base_images_src_dir: The root directory where source images are located.
    :param images_dst_base: The base destination directory for images.
    :param base_labels_src_dir: The root directory where source labels are located.
    :param labels_dst_base: The base destination directory for labels.
    """
    for relative_path in file_list_relative_paths:
        src_image_path = os.path.join(base_images_src_dir, relative_path)
        
        # Construct label path based on relative image path
        label_relative_path = os.path.splitext(relative_path)[0] + '.txt'
        src_label_path = os.path.join(base_labels_src_dir, label_relative_path)

        # Create destination directories, preserving structure
        dst_image_path = os.path.join(images_dst_base, relative_path)
        dst_label_path = os.path.join(labels_dst_base, label_relative_path)

        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)

        try:
            shutil.copy(src_image_path, dst_image_path)
            if os.path.exists(src_label_path):
                shutil.copy(src_label_path, dst_label_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export {relative_path}:\n{e}")

# --------------------------------------------------
# BoundingBoxEditor Class
# --------------------------------------------------

class BoundingBoxEditor:
    """
    This class provides a Tkinter-based interface for visualizing and editing bounding boxes
    on images. It also integrates a YOLO-based auto-annotation feature and a project-based 
    organizational structure for labeling tasks.
    """

    def __init__(self, root, project):
        self.root = root
        self.project = project
        self.root.title(f"Bounding Box Editor - Project: {project['project_name']}")

        # Model and concurrency handles
        self.model = None
        self.cancel_event = None

        # Folder paths from the project
        self.folder_path = project["dataset_path"]
        self.label_folder = os.path.join(self.folder_path, "labels")
        os.makedirs(self.label_folder, exist_ok=True)

        # YAML file path (dataset config)
        self.yaml_path = os.path.join(self.folder_path, "dataset.yaml")
        self.create_default_yaml_if_missing()

        # Load data from YAML
        with open(self.yaml_path, "r") as f:
            data = yaml.safe_load(f)

        # Class names from YAML
        raw_names = data.get("names", ["person"])
        if isinstance(raw_names, dict):
            # Convert dict to list sorted by integer keys
            self.class_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x: int(x))]
        else:
            self.class_names = raw_names

        # Paths used in the YAML (optional usage)
        self.paths = data.get("paths", {"dataset": self.folder_path, "train": "", "val": ""})
        self.validation = bool(self.paths.get("val"))
        self.auto_save_interval = data.get("auto_save_interval", 0)

        # Color mapping for classes
        self.update_class_colors()        # Dictionary to hold status of each image
        self.image_status = {}

        # -----------------------------
        # Main UI Layout
        # -----------------------------
        # Top Bar: Buttons (Auto Annotate, Save, Load Model, Export)
        self.setup_top_bar()

        # Main content frame (everything except top bar and bottom status bar)
        self.main_content_frame = tk.Frame(root)
        self.main_content_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Pane: Treeview for image list (inside main_content_frame)
        self.image_list_frame = tk.Frame(self.main_content_frame, width=200)
        self.image_list_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.setup_image_list_panel_widgets()

        # Middle Pane: Canvas (image) + Info Panel (inside main_content_frame)
        self.content_frame = tk.Frame(self.main_content_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_canvas()
        self.setup_info_panel()
        
        # Right Pane: Class List + Actions (inside main_content_frame)
        self.class_frame = tk.Frame(self.content_frame, width=200)
        self.class_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 10), pady=10)
        self.setup_class_panel()

        # Status Bar: labeling progress counters (at the very bottom)
        self.setup_status_bar()
        
        # Initialize image-related variables
        self.image = None
        self.image_path = None
        self.original_image = None # Explicitly initialize
        self.bboxes = [] # Stores (x, y, w, h, class_id) for boxes
        self.polygons = [] # Stores {'class_id': int, 'points': [(x1, y1), (x2, y2), ...]} for polygons
        self.current_bbox = None
        self.current_polygon_points = [] # For drawing current polygon
        self.polygon_drawing_active = False # Track if actively drawing a polygon
        self.rect = None
        self.polygon_line_ids = [] # To store canvas line IDs for current polygon
        
        # Polygon point editing state
        self.dragging_point = False # Track if currently dragging a polygon point
        self.drag_polygon_index = -1 # Index of polygon being edited
        self.drag_point_index = -1 # Index of point being dragged
        self.hover_polygon_index = -1 # Index of polygon with hovered point
        self.hover_point_index = -1 # Index of hovered point
        # Polygon movement state
        self.dragging_whole_polygon = False # Track if currently dragging entire polygon
        self.drag_whole_polygon_index = -1  # Index of polygon being moved
        self.polygon_move_start = (0, 0)    # Starting position for polygon move
        self.image_files = []
        self.current_image_index = -1
        self.selected_class_index = None
        self.annotation_mode = 'box' # Default mode
        self.zoom_level = 1.0 # Initialize zoom level

        # History for Undo/Redo
        self.history = []
        self.history_index = -1
        self.max_history_size = 20 # Limit history to prevent excessive memory usage

        # Load dataset images + statuses
        self.load_dataset()

        # Additional Key Bindings
        self.setup_bindings()

        # Initial save to history
        self.save_history()
        if self.auto_save_interval and self.auto_save_interval > 0:
            self.start_auto_save()

        # Attempt to load the last opened image for this project
        if 'last_opened_image_relative' in self.project:
            last_image_relative_path = self.project['last_opened_image_relative']
            if last_image_relative_path: # Ensure it's not empty
                last_image_full_path = os.path.join(self.folder_path, last_image_relative_path)
                if os.path.exists(last_image_full_path) and last_image_relative_path in self.image_files:
                    # Select in tree and load
                    try:
                        self.image_tree.selection_set(last_image_relative_path) # Select in tree
                        self.image_tree.focus(last_image_relative_path)         # Ensure it's visible
                        self.image_tree.see(last_image_relative_path)           # Scroll to make it visible
                        self.load_image(last_image_full_path)                   # Load the image
                    except tk.TclError:
                        # Item might not exist in tree if image_files list changed, or tree not fully populated
                        print(f"Info: Could not auto-select last opened image '{last_image_relative_path}' in tree.")
                        # Fallback to loading the first image if available
                        if self.image_files:
                            self.load_image(os.path.join(self.folder_path, self.image_files[0]))
                            self.image_tree.selection_set(self.image_files[0])
                elif self.image_files: # If last opened doesn't exist, load first image
                    self.load_image(os.path.join(self.folder_path, self.image_files[0]))
                    self.image_tree.selection_set(self.image_files[0])
        elif self.image_files: # If no last opened image, load the first one
            self.load_image(os.path.join(self.folder_path, self.image_files[0]))
            self.image_tree.selection_set(self.image_files[0])


    # --------------------------------------------------
    # Setup / Layout Methods
    # --------------------------------------------------

    def setup_image_list_panel_widgets(self): # Renamed
        """
        Sets up the widgets within the image list panel.
        """
        # self.image_list_frame is now created in __init__

        self.image_tree = ttk.Treeview(
            self.image_list_frame,
            columns=("filename",),
            show="headings",
            selectmode="browse"
        )
        self.image_tree.heading("filename", text="Images")
        self.image_tree.column("filename", anchor=tk.W)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Status-based color tags
        self.image_tree.tag_configure("edited", background="lightgreen")
        self.image_tree.tag_configure("viewed", background="lightblue")
        self.image_tree.tag_configure("not_viewed", background="white")
        self.image_tree.tag_configure("review_needed", background="red")

        scrollbar = tk.Scrollbar(self.image_list_frame, orient="vertical", command=self.image_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=scrollbar.set)

        self.image_tree.bind("<<TreeviewSelect>>", self.on_image_select)

    def setup_canvas(self):
        """
        Creates the main canvas used to display the current image and bounding boxes.
        """
        self.canvas = tk.Canvas(self.content_frame, width=500, height=720)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        # Track canvas size for dynamic resizing
        self.canvas_width = 500
        self.canvas_height = 720
        # Handle resizing event to scale image and annotations
        self.canvas.bind("<Configure>", self.on_canvas_resize)
        # Mouse wheel for switching images
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down
        
        # Drawing bounding boxes
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_motion) # For hover detection over polygon points
        self.canvas.bind("<Double-Button-1>", self.on_double_click) # For completing polygons
        self.canvas.bind("<Button-3>", self.on_right_click) # Right-click to cancel polygon

    def setup_info_panel(self):
        """
        Creates the right-side panel (inside the middle frame) to show bounding box info.
        """
        self.info_frame = tk.Frame(self.content_frame, width=300)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.info_label = tk.Label(self.info_frame, text="Annotations Info", font=("Arial", 14, "bold"))
        self.info_label.pack(pady=10)
 
        self.image_name_label = tk.Label(self.info_frame, text="", font=("Arial", 10))
        self.image_name_label.pack(pady=5)
 
        # Use a Canvas for the info frame to allow scrolling if many annotations
        self.info_canvas = tk.Canvas(self.info_frame)
        self.info_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
 
        self.info_scrollbar = tk.Scrollbar(self.info_frame, orient="vertical", command=self.info_canvas.yview)
        self.info_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.info_canvas.configure(yscrollcommand=self.info_scrollbar.set)
 
        self.bbox_info_frame = tk.Frame(self.info_canvas)
        self.info_canvas.create_window((0, 0), window=self.bbox_info_frame, anchor="nw")
          # Configure scroll region
        self.bbox_info_frame.bind("<Configure>", lambda e: self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")))

    def setup_class_panel(self):
        """
        Creates the panel for class list, class management, and copy/paste features.
        """
        self.class_label = tk.Label(self.class_frame, text="Classes", font=("Arial", 14))
        self.class_label.pack(pady=10)

        self.class_listbox = tk.Listbox(self.class_frame)
        self.class_listbox.pack(pady=10, fill=tk.BOTH, expand=True)

        for cls in self.class_names:
            self.class_listbox.insert(tk.END, cls)

        self.class_entry = tk.Entry(self.class_frame)
        self.class_entry.pack(pady=5, fill=tk.X, padx=5)

        btn_frame = tk.Frame(self.class_frame)
        btn_frame.pack(pady=5)
        
        tk.Button(btn_frame, text="Add", command=self.add_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Update", command=self.update_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Remove", command=self.remove_class).pack(side=tk.LEFT, padx=2)
        
        # Second row for reload button
        btn_frame2 = tk.Frame(self.class_frame)
        btn_frame2.pack(pady=5)
        tk.Button(btn_frame2, text="Reload from YAML", command=self.reload_classes_from_yaml).pack(padx=2)

        self.clear_selection_button = tk.Button(
            self.class_frame,
            text="Clear Selection",
            command=self.clear_class_selection
        )
        self.clear_selection_button.pack(pady=10)

        self.paste_all_button = tk.Button(self.class_frame, text="Paste All", command=self.paste_all_bboxes)
        self.paste_all_button.pack(pady=10)

        self.delete_image_button = tk.Button(self.class_frame, text="Delete Image", command=self.delete_image)
        self.delete_image_button.pack(pady=10)

        # Frame for showing which bounding boxes were copied
        self.copy_frame = tk.Frame(self.class_frame)
        self.copy_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.copy_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-4>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-5>", self.on_mouse_wheel)

        # Store the currently copied bounding boxes
        self.copied_bbox_list = []
        self.update_copied_bbox_display()

        # Confidence threshold scale for auto-annotation
        self.confidence_threshold = tk.DoubleVar(value=0.5)
        self.confidence_scale = tk.Scale(
            self.class_frame,
            from_=0.0,
            to=1.0,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            label="Confidence Threshold",
            variable=self.confidence_threshold
        )
        self.confidence_scale.pack(pady=10)

    def setup_top_bar(self):
        """
        Creates the top bar buttons: Auto Annotate, Save, Load Model, Export YAML.
        """
        top_frame = tk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        buttons_frame = tk.Frame(top_frame)
        buttons_frame.pack(side=tk.LEFT, pady=5)

        self.auto_annotate_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['auto_annotate']} Auto Annotate",
            command=self.auto_annotate_dataset_threaded
        )
        self.auto_annotate_button.pack(side=tk.LEFT, padx=5)
 
        self.save_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['save']} Save Labels",
            command=self.save_labels
        )
        self.save_button.pack(side=tk.LEFT, padx=5)
 
        self.load_model_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['load_model']} Load Model",
            command=self.load_model
        )
        self.load_model_button.pack(side=tk.LEFT, padx=5)
 
        self.export_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['export']} Export Annotations",
            command=self.export_format_selection_window
        )
        self.export_button.pack(side=tk.LEFT, padx=5)
 
        self.mode_toggle_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['mode_box']} Mode: Box",
            command=self.toggle_annotation_mode
        )
        self.mode_toggle_button.pack(side=tk.LEFT, padx=15)  # More space for mode toggle
 
        # Undo/Redo buttons
        self.undo_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['undo']} Undo",
            command=self.undo,
            state=tk.DISABLED
        )
        self.undo_button.pack(side=tk.LEFT, padx=5)
        self.redo_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['redo']} Redo",
            command=self.redo,
            state=tk.DISABLED
        )
        self.redo_button.pack(side=tk.LEFT, padx=5)

        # Zoom controls
        self.zoom_in_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['zoom_in']}",
            command=self.zoom_in
        )
        self.zoom_in_button.pack(side=tk.LEFT, padx=5)
        self.zoom_out_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['zoom_out']}",
            command=self.zoom_out
        )
        self.zoom_out_button.pack(side=tk.LEFT, padx=5)
        # Keyboard shortcuts help
        self.shortcuts_button = tk.Button(
            buttons_frame,
            text=f"{ICON_UNICODE['shortcuts']} Shortcuts",
            command=self.show_shortcuts
        )
        self.shortcuts_button.pack(side=tk.LEFT, padx=5)

    def setup_status_bar(self):
        """
        Creates a simple status bar at the bottom of the root window.
        """
        self.status_frame = tk.Frame(self.root, bd=1, relief=tk.SUNKEN) # Use self.root
        self.status_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(2,0), padx=2)

        self.status_labels = {}
        statuses = [
            ("Viewed", "viewed"),
            ("Labeled", "edited"),
            ("Review Needed", "review_needed"),
            ("Non-viewed", "not_viewed")
        ]
        for display_name, tag in statuses:
            frame = tk.Frame(self.status_frame) # Use self.status_frame
            frame.pack(side=tk.LEFT, padx=10, pady=2) # Added pady
            label = tk.Label(frame, text=f"{display_name}: 0", font=("Arial", 9)) # Smaller font
            label.pack()
            self.status_labels[display_name] = label

    def setup_bindings(self):
        """
        Binds additional keyboard shortcuts for navigation and actions.
        """
        # Ctrl+S to save
        self.root.bind("<Control-s>", lambda event: self.save_labels())

        # Ctrl+Z for Undo, Ctrl+Y for Redo
        self.root.bind("<Control-z>", lambda event: self.undo())
        self.root.bind("<Control-y>", lambda event: self.redo())
        self.root.bind("<Control-Shift-Z>", lambda event: self.redo()) # Common alternative for redo        # ESC to cancel polygon or clear selection
        self.root.bind("<Escape>", self.on_escape_key)

        # Down/Up arrow to navigate images
        self.root.bind("<Down>", lambda event: self.navigate_image(+1))
        self.root.bind("<Up>", lambda event: self.navigate_image(-1))

        # Prevent Listbox from also capturing up/down
        self.class_listbox.bind("<Down>", lambda e: "break")
        self.class_listbox.bind("<Up>", lambda e: "break")        # Digit-based quick class selection
        self.root.bind("<Key>", self.on_key_press)
        # Delete key to remove hovered polygon vertex
        self.root.bind("<Delete>", self.on_delete_vertex)
        # Backspace also removes hovered polygon vertex
        self.root.bind("<BackSpace>", self.on_delete_vertex)

        # Zoom via Ctrl + Mouse Wheel
        self.canvas.bind("<Control-MouseWheel>", self.on_zoom)
        self.canvas.bind("<Control-Button-4>", self.on_zoom)  # Linux scroll up
        self.canvas.bind("<Control-Button-5>", self.on_zoom)  # Linux scroll down
        # Pan via middle-mouse drag
        self.canvas.bind("<ButtonPress-2>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B2-Motion>",      lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))

    def show_shortcuts(self):
        """
        Displays a dialog listing all available keyboard and mouse shortcuts.
        """
        shortcut_list = [
            ("Ctrl+S", "Save labels"),
            ("Ctrl+Z", "Undo"),
            ("Ctrl+Y", "Redo"),
            ("Esc", "Cancel polygon or clear selection"),
            ("Up/Down Arrow", "Navigate images"),
            ("Mouse Wheel", "Navigate images"),
            ("Ctrl + Mouse Wheel", "Zoom in/out"),
            ("Middle Mouse Drag", "Pan"),
            ("Digits 1-9", "Select class"),
        ]
        dlg = tk.Toplevel(self.root)
        dlg.title("Keyboard Shortcuts")
        dlg.transient(self.root)
        dlg.grab_set()
        frame = ttk.Frame(dlg, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        cols = ("Shortcut", "Description")
        tree = ttk.Treeview(frame, columns=cols, show="headings", selectmode="none")
        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, anchor="w")
        for key, desc in shortcut_list:
            tree.insert("", "end", values=(key, desc))
        tree.pack(fill=tk.BOTH, expand=True)
        btn = ttk.Button(frame, text="Close", command=dlg.destroy)
        btn.pack(pady=(10, 0))
        dlg.update_idletasks()
        center_window(dlg, 400, 300)
        self.root.wait_window(dlg)

    def on_zoom(self, event):
        """
        Handles zoom in/out with Ctrl + Mouse Wheel or equivalent.
        """
        if hasattr(event, "delta"):
            delta = event.delta
        elif hasattr(event, "num") and event.num == 4:
            delta = 120
        elif hasattr(event, "num") and event.num == 5:
            delta = -120
        else:
            return
        factor = 1.1 if delta > 0 else 0.9
        new_zoom = self.zoom_level * factor
        new_zoom = max(0.1, min(new_zoom, 10.0))
        scale = new_zoom / self.zoom_level
        self.zoom_level = new_zoom
        # Scale bounding boxes
        self.bboxes = [
            (int(x * scale), int(y * scale), int(w * scale), int(h * scale), class_id)
            for x, y, w, h, class_id in self.bboxes
        ]
        # Scale polygons
        scaled_polygons = []
        for poly in self.polygons:
            scaled_polygons.append({
                "class_id": poly["class_id"],
                "points": [(int(px * scale), int(py * scale)) for px, py in poly["points"]]
            })
        self.polygons = scaled_polygons
        # Trigger resize logic to update image display with new zoom
        class _E: pass
        e = _E()
        e.width = self.canvas_width
        e.height = self.canvas_height
        self.on_canvas_resize(e)

    def zoom_in(self):
        """Zoom in via toolbar button."""
        class _E: pass
        e = _E()
        e.delta = 120
        self.on_zoom(e)

    def zoom_out(self):
        """Zoom out via toolbar button."""
        class _E: pass
        e = _E()
        e.delta = -120
        self.on_zoom(e)

    def on_escape_key(self, event):
        """
        Handles ESC key press - cancels polygon drawing if active, otherwise clears class selection.
        """
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            self.cancel_current_polygon()
        else:
            self.clear_class_selection()

    def on_right_click(self, event):
        """
        Handles right-click on canvas - cancels polygon drawing if in polygon mode.
        """
        if self.annotation_mode == 'polygon' and self.polygon_drawing_active:
            self.cancel_current_polygon()

    def on_delete_vertex(self, event):
        """
        Deletes the hovered vertex of a polygon, or removes polygon if vertices < 3.
        """
        if self.annotation_mode == 'polygon' and not self.polygon_drawing_active:
            idx = self.hover_polygon_index
            vidx = self.hover_point_index
            if 0 <= idx < len(self.polygons) and 0 <= vidx < len(self.polygons[idx]['points']):
                points = self.polygons[idx]['points']
                if len(points) > 3:
                    del points[vidx]
                else:
                    # Removing this point would invalidate polygon; delete entire polygon
                    if messagebox.askyesno(
                            "Delete Polygon",
                            "Deleting this vertex will remove the whole polygon. Proceed?"):
                        del self.polygons[idx]
                        self.hover_polygon_index = -1
                        self.hover_point_index = -1
                self.display_annotations()
                self.save_history()

    def toggle_annotation_mode(self):
        """
        Toggles between 'box' and 'polygon' annotation modes.
        """
        if self.annotation_mode == 'box':
            self.annotation_mode = 'polygon'
            self.mode_toggle_button.config(text="Mode: Polygon")
            messagebox.showinfo("Mode Switched", "Switched to Polygon Annotation Mode.\nClick to add points, double-click to complete polygon.\nPress ESC or right-click to cancel current polygon.")
            # Clear any partial box drawing
            self.canvas.delete(self.rect)
            self.current_bbox = None
        else:
            self.annotation_mode = 'box'
            self.mode_toggle_button.config(text="Mode: Box")
            messagebox.showinfo("Mode Switched", "Switched to Box Annotation Mode.")
            # Clear any partial polygon drawing
            self.clear_current_polygon_drawing()
            self.current_polygon_points = []
            self.polygon_drawing_active = False  # Reset polygon drawing state

    # --------------------------------------------------
    # YAML / Dataset / Project Setup
    # --------------------------------------------------

    def create_default_yaml_if_missing(self):
        """
        Creates a default dataset YAML file if none exists in the dataset folder.
        """
        if not os.path.exists(self.yaml_path):
            default_yaml = {
                "train": os.path.join(self.folder_path, 'train'),
                "val": os.path.join(self.folder_path, 'val'),
                "nc": 1,   # Number of classes
                "names": ["person"],
                "auto_save_interval": 120
            }
            with open(self.yaml_path, "w") as f:
                yaml.dump(default_yaml, f, sort_keys=False)

    def load_dataset(self):
        """
        Loads the image filenames from the dataset folder, their statuses, 
        and populates the Treeview with color-coded status tags.
        """
        if not self.folder_path:
            messagebox.showerror("Error", "Dataset folder not set.")
            return

        # Clear any existing items in the tree
        for item in self.image_tree.get_children():
            self.image_tree.delete(item)

        # Gather image files from dataset folder and its subfolders
        self.image_files = []
        for root_dir, _, files in os.walk(self.folder_path):
            for file in files:
                if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    # Store the path relative to the dataset folder
                    relative_path = os.path.relpath(os.path.join(root_dir, file), self.folder_path)
                    self.image_files.append(relative_path)
        self.image_files.sort() # Ensure consistent order
        if not self.image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return

        # Load statuses from JSON if available
        self.load_statuses()

        # Insert items into Treeview
        for relative_image_path in self.image_files:
            status = self.image_status.get(relative_image_path, "not_viewed")
            self.image_tree.insert("", tk.END, iid=relative_image_path, values=(relative_image_path,), tags=(status,))

        # Make sure we save and refresh labels
        self.save_statuses()
        self.update_status_labels()

    # --------------------------------------------------
    # Status Persistence
    # --------------------------------------------------

    def save_statuses(self):
        """
        Persists the image statuses (viewed, edited, etc.) to a JSON file in the dataset folder.
        """
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            with open(status_file, "w") as f:
                json.dump(self.image_status, f)

    def load_statuses(self):
        """
        Loads the image statuses from a JSON file in the dataset folder if it exists.
        """
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    self.image_status = json.load(f)
            else:
                self.image_status = {}

    def update_status_labels(self):
        """
        Updates the status bar labels with the latest counts of each status category.
        """
        counts = {
            "Viewed": 0,
            "Labeled": 0,
            "Review Needed": 0,
            "Non-viewed": 0
        }
        for relative_image_path in self.image_files:
            status = self.image_status.get(relative_image_path, "not_viewed")
            if status == "edited":
                counts["Labeled"] += 1
                counts["Viewed"] += 1  # "edited" implies it was also viewed
            elif status == "viewed":
                counts["Viewed"] += 1
            elif status == "review_needed":
                counts["Review Needed"] += 1
            elif status == "not_viewed":
                counts["Non-viewed"] += 1

        for display_name in counts:
            self.status_labels[display_name].config(text=f"{display_name}: {counts[display_name]}")

    # --------------------------------------------------
    # Auto-Save Mechanism
    # --------------------------------------------------

    def start_auto_save(self):
        """
        Starts a periodic auto-save timer based on the configured interval.
        """
        self.auto_save_id = self.root.after(self.auto_save_interval * 1000, self._auto_save_callback)

    def _auto_save_callback(self):
        """
        Auto-save callback: saves labels and statuses and reschedules the timer.
        """
        self.save_labels()
        self.start_auto_save()

    # --------------------------------------------------
    # Class Management
    # --------------------------------------------------

    def update_class_colors(self):
        """
        Initializes a color map for each class index. 
        You can expand the color list or generate them programmatically if needed.
        """
        predefined_colors = [
            "red", "green", "blue", "yellow",
            "cyan", "magenta", "orange", "purple",
            "brown", "pink"
        ]
        self.class_colors = {
            i: predefined_colors[i % len(predefined_colors)]
            for i in range(len(self.class_names))
        }

    def add_class(self):
        """
        Adds a new class name from the class_entry input to the class list and updates YAML.
        """
        new_class = self.class_entry.get().strip()
        if new_class:
            self.class_listbox.insert(tk.END, new_class)
            self.class_names.append(new_class)
            self.update_class_colors()
            self.update_yaml_classes()
            self.class_entry.delete(0, tk.END)

    def update_class(self):
        """
        Updates the currently selected class name to the new text in class_entry, then updates YAML.
        """
        selection = self.class_listbox.curselection()
        if selection:
            index = selection[0]
            new_val = self.class_entry.get().strip()
            if new_val:
                self.class_listbox.delete(index)
                self.class_listbox.insert(index, new_val)
                self.class_names[index] = new_val
                self.update_class_colors()
                self.update_yaml_classes()
                self.class_entry.delete(0, tk.END)

    def remove_class(self):
        """
        Removes the selected class from the class list, but ensures at least one class remains.
        Re-labels bounding boxes that used the removed class to 0 if out-of-range.
        """
        selection = self.class_listbox.curselection()
        if selection:
            if len(self.class_names) == 1:
                messagebox.showwarning("Warning", "You must have at least one class.")
                return
            index = selection[0]
            self.class_listbox.delete(index)
            self.class_names.pop(index)
            self.update_class_colors()
            self.update_yaml_classes()

            # Adjust existing bboxes that used that class index
            updated_bboxes = []
            max_idx = len(self.class_names) - 1
            for x, y, w, h, class_id in self.bboxes:
                if class_id > max_idx:
                    class_id = 0
                updated_bboxes.append((x, y, w, h, class_id))
            self.bboxes = updated_bboxes
            self.display_annotations()

    def update_yaml_classes(self):
        """
        Updates the 'nc' and 'names' fields in the YAML file with the current class list.
        Also ensures 'train' and 'val' paths are updated if needed.
        """
        try:
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f)
        except Exception:
            data = {}

        data["nc"] = len(self.class_names)
        data["names"] = self.class_names
        data["train"] = os.path.join(self.folder_path, 'train')
        data["val"] = os.path.join(self.folder_path, 'val')        
        with open(self.yaml_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    # --------------------------------------------------
    # History Management (Undo/Redo)
    # --------------------------------------------------

    def save_history(self):
        """
        Saves the current state (bboxes and polygons) to the history stack for undo/redo functionality.
        """
        if self.current_image_index == -1:
            return

        # Create a deep copy of current state
        current_state = {
            'bboxes': [bbox[:] for bbox in self.bboxes],  # Deep copy of bboxes
            'polygons': [{'class_id': p['class_id'], 'points': p['points'][:]} for p in self.polygons],  # Deep copy of polygons
            'image_index': self.current_image_index
        }

        # Remove any future history if we're not at the end
        if self.history_index < len(self.history) - 1:
            self.history = self.history[:self.history_index + 1]

        # Add new state
        self.history.append(current_state)
        self.history_index = len(self.history) - 1

        # Limit history size
        if len(self.history) > self.max_history_size:
            self.history.pop(0)
            self.history_index -= 1

        self.update_undo_redo_buttons()

    def undo(self):
        """
        Reverts to the previous state in the history stack.
        """
        if self.history_index > 0:
            self.history_index -= 1
            self.restore_from_history()

    def redo(self):
        """
        Moves forward to the next state in the history stack.
        """
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.restore_from_history()

    def restore_from_history(self):
        """
        Restores the current state from the history at history_index.
        """
        if 0 <= self.history_index < len(self.history):
            state = self.history[self.history_index]
            
            # Restore bboxes and polygons
            self.bboxes = [bbox[:] for bbox in state['bboxes']]  # Deep copy
            self.polygons = [{'class_id': p['class_id'], 'points': p['points'][:]} for p in state['polygons']]  # Deep copy
            
            # Refresh display
            self.display_annotations()
            self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        """
        Updates the enabled/disabled state of undo and redo buttons based on history position.
        """
        # Enable/disable undo button
        if self.history_index > 0:
            self.undo_button.config(state=tk.NORMAL)
        else:
            self.undo_button.config(state=tk.DISABLED)

        # Enable/disable redo button
        if self.history_index < len(self.history) - 1:
            self.redo_button.config(state=tk.NORMAL)
        else:
            self.redo_button.config(state=tk.DISABLED)

    # --------------------------------------------------
    # Image Navigation / Display
    # --------------------------------------------------

    def on_image_select(self, event):
        """
        Handler when user selects a new image from the Treeview.
        """
        selected = self.image_tree.selection()
        if selected:
            relative_image_path = selected[0]
            image_path = os.path.join(self.folder_path, relative_image_path)
            self.load_image(image_path)

    def load_image(self, image_path=None):
        """
        Loads an image from 'image_path' into the canvas, reads associated bboxes, 
        and updates the internal state (self.current_image_index, etc.).
        """
        if image_path:
            self.image_path = image_path
            # The image_files list now stores relative paths, so we need to find the index based on that
            self.current_image_index = self.image_files.index(os.path.relpath(image_path, self.folder_path))
        else:
            # Fallback: user manually picks a file (this will be outside the project structure)
            # For now, we'll just load it but it won't be part of the project's image_files list
            messagebox.showwarning("Manual Load", "Manually loaded images are not part of the project's dataset structure and won't be saved.")
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not self.image_path:
                return
            self.current_image_index = -1 # Indicate it's not part of the project's image list

        # Read and resize the image to fit the canvas
        # WARNING: This can distort aspect ratio. Consider preserving ratio if you prefer.
        original_image = cv2.imread(self.image_path)
        if original_image is None:
            messagebox.showerror("Error", f"Failed to load image: {self.image_path}\nFile might be missing, corrupted, or in an unsupported format.")
            self.image = None # Set self.image to None first
            self.original_image = None
            self.image_name_label.config(text=f"Error loading: {os.path.basename(self.image_path)}")
            self.bboxes = []
            self.polygons = []
            self.display_image()       # Call display_image to show error on canvas
            self.display_annotations() # Clear any annotation drawings from canvas
            return

        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # Store original image for dynamic resizing/zoom
        self.original_image = original_image
        # Resize to current canvas size
        if self.original_image is not None: # Ensure original_image is valid before resize
            self.image = cv2.resize(self.original_image, (self.canvas_width, self.canvas_height))
        else: # Should not happen if the above check is in place, but as a safeguard
            self.image = None
            messagebox.showerror("Error", "Internal error: Original image became None before resizing.")
            return

        # Draw the image in the canvas
        self.display_image()

        # Load bounding boxes from label file
        # Construct label path based on the relative path of the image within the dataset folder
        relative_image_path_for_label = os.path.relpath(self.image_path, self.folder_path)
        label_relative_path = os.path.splitext(relative_image_path_for_label)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)
        
        # Ensure the label directory exists for nested paths
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        self.bboxes, self.polygons = read_annotations_from_file(label_path, self.image.shape)

        # Show annotations
        self.display_annotations()

        # Update status if we have bboxes or not
        # Use the relative path as the item ID for status tracking
        relative_image_path = os.path.relpath(self.image_path, self.folder_path)
        new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
        self.image_status[relative_image_path] = new_status
        self.image_tree.item(relative_image_path, tags=(new_status,))
        self.save_statuses()
        self.update_status_labels()

        # Update the info panel
        self.image_name_label.config(text=relative_image_path)

        # Maintain class selection
        if self.selected_class_index is not None:
            self.class_listbox.selection_set(self.selected_class_index)

        # If image loaded successfully, save it as the last opened image for this project
        if self.image is not None and self.image_path and self.current_image_index != -1:
            relative_image_path = os.path.relpath(self.image_path, self.folder_path)
            self.project['last_opened_image_relative'] = relative_image_path
            self._save_project_config()

    def _save_project_config(self):
        """Saves the current self.project dictionary to its JSON file."""
        if not hasattr(self, 'project') or 'project_name' not in self.project:
            print("Error: Project name not found, cannot save project config.")
            return

        project_name = self.project['project_name']
        # Sanitize project_name for use as a filename, consistent with ProjectManager
        safe_project_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name).rstrip()
        if not safe_project_filename:
            safe_project_filename = "Untitled_Project" # Fallback, should match ProjectManager logic
        
        project_file_path = os.path.join(PROJECTS_DIR, f"{safe_project_filename}.json")

        try:
            with open(project_file_path, "w") as f:
                json.dump(self.project, f, indent=4)
        except Exception as e:
            # Log this error or show a non-intrusive message, as this save is a background task
            print(f"Error saving project configuration to {project_file_path}: {e}")
            # Optionally, inform the user via a status bar message or a one-time dialog
            # messagebox.showwarning("Save Warning", f"Could not save last opened image state: {e}", parent=self.root)


    def display_image(self):
        """
        Clears the canvas and displays the current self.image (already resized).
        """
        self.canvas.delete("all")
        if self.image is None:
            # Display an error message on the canvas if image data is missing
            self.canvas.create_text(
                self.canvas_width / 2 if hasattr(self, 'canvas_width') and self.canvas_width > 0 else 300,
                self.canvas_height / 2 if hasattr(self, 'canvas_height') and self.canvas_height > 0 else 300,
                text="Error: Image data is missing or invalid.",
                fill="red",
                font=("Arial", 12)
            )
            if hasattr(self, 'tk_image'): # Clear previous tk_image if it exists
                self.tk_image = None
            return

        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def navigate_image(self, direction):
        """
        Moves to the next or previous image based on 'direction' (+1 or -1).
        """
        self.root.focus_set()
        if self.current_image_index == -1: # If manually loaded image
            return

        new_index = self.current_image_index + direction
        if 0 <= new_index < len(self.image_files):
            self.current_image_index = new_index
            # image_files now contains relative paths, so join with folder_path
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))

    def on_mouse_wheel(self, event):
        """
        Scroll up/down to navigate images (some platforms: event.delta, others: event.num).
        """
        if event.delta:
            if event.delta > 0:
                self.navigate_image(-1)
            else:
                self.navigate_image(+1)
        elif event.num == 5:
            self.navigate_image(+1)
        elif event.num == 4:
            self.navigate_image(-1)
    
    # --------------------------------------------------
    # Drawing / Editing Bounding Boxes
    # --------------------------------------------------
    
    def on_click(self, event):
        """
        Handles mouse clicks for both bounding box and polygon drawing, and polygon point dragging.
        For polygons, it adds points when drawing or starts point dragging when editing.
        For boxes, it starts a new box.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please select or load an image first.")
            return

        if self.annotation_mode == 'box':
            self.current_bbox = [event.x, event.y, event.x, event.y]
            self.rect = self.canvas.create_rectangle(
                event.x, event.y, event.x, event.y,
                outline="blue", width=2, tags="bbox"
            )
        elif self.annotation_mode == 'polygon':
            x, y = event.x, event.y # Define x, y once at the top

            if self.polygon_drawing_active:
                # We are in the middle of drawing a polygon, add the new point
                self.current_polygon_points.append((x, y))
                self.draw_current_polygon_drawing()
            else:
                # We are NOT actively drawing a polygon. This click could be:
                # 1. Starting to drag an existing vertex
                # 2. Starting to drag an entire existing polygon
                # 3. Starting to draw a NEW polygon (if not 1 or 2)

                # Check for vertex dragging
                if self.hover_polygon_index != -1 and self.hover_point_index != -1:
                    self.dragging_point = True
                    self.drag_polygon_index = self.hover_polygon_index
                    self.drag_point_index = self.hover_point_index
                    self.canvas.config(cursor="fleur")
                # Check for whole polygon movement
                elif self.hover_polygon_index != -1 and self.is_point_in_polygon(
                        x, y, self.polygons[self.hover_polygon_index]['points']): # Use x,y from above
                    self.dragging_whole_polygon = True
                    self.drag_whole_polygon_index = self.hover_polygon_index
                    self.polygon_move_start = (x, y) # Use x,y from above
                    self.canvas.config(cursor="fleur")
                else:
                    # Not editing an existing polygon, so start drawing a new one
                    self.polygon_drawing_active = True
                    self.current_polygon_points = [(x, y)] # Initialize with the first point
                    self.draw_current_polygon_drawing()
                
    def on_drag(self, event):
        """
        Update the rectangle shape as the user drags the mouse (for box mode),
        or update polygon point position (for polygon point dragging).
        """
        if self.annotation_mode == 'box' and self.current_bbox is not None:
            self.current_bbox[2] = event.x
            self.current_bbox[3] = event.y
            self.canvas.coords(self.rect, *self.current_bbox)
        elif self.annotation_mode == 'polygon' and self.dragging_whole_polygon:
            # Move entire polygon by delta
            dx = event.x - self.polygon_move_start[0]
            dy = event.y - self.polygon_move_start[1]
            pts = self.polygons[self.drag_whole_polygon_index]['points']
            self.polygons[self.drag_whole_polygon_index]['points'] = [
                (px + dx, py + dy) for px, py in pts]
            self.polygon_move_start = (event.x, event.y)
            self.display_annotations()
        elif self.annotation_mode == 'polygon' and self.dragging_point:
            # Update the dragged polygon point position
            if 0 <= self.drag_polygon_index < len(self.polygons) and \
               0 <= self.drag_point_index < len(self.polygons[self.drag_polygon_index]['points']):
                self.polygons[self.drag_polygon_index]['points'][self.drag_point_index] = (event.x, event.y)
                self.display_annotations()  # Refresh to show updated polygon
    
    def on_release(self, event):
        """
        Finalize the bounding box, store it with the currently selected class (only for box mode).
        For polygon mode, finishes polygon point dragging.
        """
        if self.annotation_mode == 'box':
            if self.image is None or self.current_bbox is None:
                return

            x1, y1, x2, y2 = self.current_bbox
            img_h, img_w = self.image.shape[:2]

            # Constrain the coords to the canvas boundaries
            x1 = max(0, min(x1, img_w))
            x2 = max(0, min(x2, img_w))
            y1 = max(0, min(y1, img_h))
            y2 = max(0, min(y2, img_h))

            x_min, y_min = min(x1, x2), min(y1, y2)
            width, height = abs(x2 - x1), abs(y2 - y1)

            # Default to class index 0 if none selected
            selected_class_index = self.class_listbox.curselection()
            class_id = selected_class_index[0] if selected_class_index else 0

            self.bboxes.append((x_min, y_min, width, height, class_id))
            self.current_bbox = None
            self.rect = None

            self.display_annotations()
            self.save_history()
        elif self.annotation_mode == 'polygon' and self.dragging_whole_polygon:
            # Finish whole polygon dragging
            self.dragging_whole_polygon = False
            self.drag_whole_polygon_index = -1
            self.canvas.config(cursor="")  # Reset cursor
            self.save_history()  # Save state after polygon move
        elif self.annotation_mode == 'polygon' and self.dragging_point:
            # Finish polygon point dragging
            self.dragging_point = False
            self.drag_polygon_index = -1
            self.drag_point_index = -1
            self.canvas.config(cursor="")  # Reset cursor
            self.save_history()  # Save state after point edit

    def on_motion(self, event):
        """
        Handles mouse motion for hover detection over polygon points.
        Changes cursor and updates hover state for visual feedback.
        """
        if self.annotation_mode != 'polygon' or self.dragging_point or self.polygon_drawing_active:
            return
        
        # Reset hover state
        prev_hover_polygon = self.hover_polygon_index
        prev_hover_point = self.hover_point_index
        self.hover_polygon_index = -1
        self.hover_point_index = -1
        
        # Check if mouse is over any polygon point
        for poly_idx, poly_data in enumerate(self.polygons):
            points = poly_data['points']
            for point_idx, (px, py) in enumerate(points):
                # Check if mouse is within 8 pixels of the point
                distance = ((event.x - px) ** 2 + (event.y - py) ** 2) ** 0.5
                if distance <= 8:
                    self.hover_polygon_index = poly_idx
                    self.hover_point_index = point_idx
                    break
            if self.hover_polygon_index != -1:
                break
        
        # Update cursor based on hover state
        if self.hover_polygon_index != -1:
            self.canvas.config(cursor="hand2")  # Show draggable cursor
        else:
            self.canvas.config(cursor="")  # Reset to default cursor
        
        # Refresh display if hover state changed (for visual feedback)
        if (prev_hover_polygon != self.hover_polygon_index or 
            prev_hover_point != self.hover_point_index):
            self.display_annotations()

    def on_double_click(self, event):
        """
        Completes the current polygon drawing.
        """
        if self.annotation_mode == 'polygon':
            self.complete_polygon()

    def draw_current_polygon_drawing(self):
        """
        Draws the current polygon points and connecting lines on the canvas.
        """
        self.clear_current_polygon_drawing()
        if len(self.current_polygon_points) > 0:
            # Draw points
            for x, y in self.current_polygon_points:
                self.canvas.create_oval(x-3, y-3, x+3, y+3, fill="red", outline="red", tags="current_polygon_point")
            
            # Draw lines
            if len(self.current_polygon_points) > 1:
                coords = []
                for p in self.current_polygon_points:
                    coords.extend(p)
                line_id = self.canvas.create_line(coords, fill="red", width=2, tags="current_polygon_line")
                self.polygon_line_ids.append(line_id)
              # Draw closing line if more than 2 points
            if len(self.current_polygon_points) > 2:
                x1, y1 = self.current_polygon_points[-1]
                x2, y2 = self.current_polygon_points[0]
                line_id = self.canvas.create_line(x1, y1, x2, y2, fill="red", width=2, dash=(4, 2), tags="current_polygon_line")
                self.polygon_line_ids.append(line_id)

    def clear_current_polygon_drawing(self):
        """
        Clears the temporary polygon drawing from the canvas.
        """
        self.canvas.delete("current_polygon_point")
        self.canvas.delete("current_polygon_line")
        self.polygon_line_ids = []

    def cancel_current_polygon(self):
        """
        Cancels the current polygon drawing and resets the drawing state.
        """
        if self.polygon_drawing_active and len(self.current_polygon_points) > 0:
            self.clear_current_polygon_drawing()
            self.current_polygon_points = []
            self.polygon_drawing_active = False
            messagebox.showinfo("Polygon Cancelled", "Current polygon drawing has been cancelled.")
        elif self.polygon_drawing_active:
            # No points drawn yet, just reset the state
            self.polygon_drawing_active = False

    def is_point_in_polygon(self, x, y, points):
        """
        Returns True if point (x, y) is inside polygon defined by list of (px, py) points.
        """
        inside = False
        n = len(points)
        for i in range(n):
            j = (i - 1) % n
            xi, yi = points[i]
            xj, yj = points[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
        return inside

    def complete_polygon(self):
        """
        Finalizes the current polygon and adds it to the list of polygons.
        """
        if len(self.current_polygon_points) < 3:
            messagebox.showwarning("Polygon Error", "A polygon must have at least 3 points.")
            self.clear_current_polygon_drawing()
            self.current_polygon_points = []
            self.polygon_drawing_active = False  # Stop drawing mode
            return

        selected_class_index = self.class_listbox.curselection()
        class_id = selected_class_index[0] if selected_class_index else 0

        self.polygons.append({
            'class_id': class_id,
            'points': self.current_polygon_points[:] # Store a copy
        })
        self.clear_current_polygon_drawing()
        self.current_polygon_points = []
        self.polygon_drawing_active = False  # Stop drawing mode to prevent accidental clicks
        self.display_annotations()
        self.save_history()

    def display_annotations(self):
        """
        Clears and redraws all bounding boxes and polygons on the canvas,
        and updates the info frame to show details (with copy/delete).
        """
        self.canvas.delete("bbox")
        self.canvas.delete("polygon")
        # Clear existing widgets in the info frame
        for widget in self.bbox_info_frame.winfo_children():
            widget.destroy()
 
        # Display Bounding Boxes
        for i, (x, y, w, h, class_id) in enumerate(self.bboxes):
            color = self.class_colors.get(class_id, "red")
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color, width=2, tags="bbox"
            )
            self.canvas.create_text(
                x, y - 10,
                text=self.class_names[class_id],
                fill=color, anchor=tk.NW, tags="bbox", font=("Arial", 8, "bold")
            )
 
            # Build a small row in the bbox_info_frame for each bounding box
            bbox_info_row = tk.Frame(self.bbox_info_frame, bd=1, relief="solid", padx=2, pady=2)
            bbox_info_row.pack(fill=tk.X, pady=2)
 
            tk.Label(
                bbox_info_row,
                text=f"Box: {self.class_names[class_id]}",
                font=("Arial", 9)
            ).grid(row=0, column=0, sticky="w")
            tk.Label(
                bbox_info_row,
                text=f"Pos:({x},{y}) Size:({w},{h})",
                font=("Arial", 8)
            ).grid(row=1, column=0, sticky="w")
 
            tk.Button(
                bbox_info_row,
                text="Copy",
                command=lambda bbox=(x, y, w, h, class_id): self.copy_bbox(bbox),
                font=("Arial", 8)
            ).grid(row=0, column=1, padx=2, sticky="e")
            tk.Button(
                bbox_info_row,
                text="Delete",
                command=lambda i=i, type='bbox': self.delete_annotation(i, type),
                font=("Arial", 8)
            ).grid(row=1, column=1, padx=2, sticky="e")
            bbox_info_row.grid_columnconfigure(0, weight=1) # Allow label to expand        # Display Polygons
        for i, poly_data in enumerate(self.polygons):
            class_id = poly_data['class_id']
            points = poly_data['points']
            color = self.class_colors.get(class_id, "blue")
            
            if len(points) > 1:
                coords = []
                for p_x, p_y in points:
                    coords.extend([p_x, p_y])
                self.canvas.create_polygon(
                    coords,
                    outline=color,
                    fill="", # No fill for polygons
                    width=2,
                    tags="polygon"
                )
                # Draw class name near the first point
                if points:
                    self.canvas.create_text(
                        points[0][0], points[0][1] - 10,
                        text=self.class_names[class_id],
                        fill=color, anchor=tk.NW, tags="polygon", font=("Arial", 8, "bold")
                    )
                
                # Draw polygon points as small circles for drag handles
                for point_idx, (px, py) in enumerate(points):
                    # Check if this point is being hovered
                    if i == self.hover_polygon_index and point_idx == self.hover_point_index:
                        # Highlight hovered point
                        self.canvas.create_oval(
                            px-5, py-5, px+5, py+5,
                            fill="yellow", outline="orange", width=2, tags="polygon"
                        )
                    else:
                        # Regular polygon point
                        self.canvas.create_oval(
                            px-3, py-3, px+3, py+3,
                            fill=color, outline="white", width=1, tags="polygon"
                        )
 
            # Build a small row in the bbox_info_frame for each polygon
            poly_info_row = tk.Frame(self.bbox_info_frame, bd=1, relief="solid", padx=2, pady=2)
            poly_info_row.pack(fill=tk.X, pady=2)
 
            tk.Label(
                poly_info_row,
                text=f"Poly: {self.class_names[class_id]}",
                font=("Arial", 9)
            ).grid(row=0, column=0, sticky="w")
            tk.Label(
                poly_info_row,
                text=f"Points: {len(points)}",
                font=("Arial", 8)
            ).grid(row=1, column=0, sticky="w")
 
            tk.Button(
                poly_info_row,
                text="Delete",
                command=lambda i=i, type='polygon': self.delete_annotation(i, type),
                font=("Arial", 8)
            ).grid(row=0, column=1, rowspan=2, padx=2, sticky="ns")
            poly_info_row.grid_columnconfigure(0, weight=1) # Allow label to expand

    def on_canvas_resize(self, event):
        """
        Callback when the canvas is resized: scales the image and annotations.
        """
        new_width, new_height = event.width, event.height
        # Avoid zero-size during initialization
        if hasattr(self, 'canvas_width') and hasattr(self, 'canvas_height') and hasattr(self, 'image'):
            scale_x = new_width / self.canvas_width
            scale_y = new_height / self.canvas_height
            # Scale bounding boxes
            self.bboxes = [
                (int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y), class_id)
                for x, y, w, h, class_id in self.bboxes
            ]
            # Scale polygons
            scaled_polygons = []
            for poly in self.polygons:
                scaled_polygons.append({
                    'class_id': poly['class_id'],
                    'points': [(int(px * scale_x), int(py * scale_y)) for px, py in poly['points']]
                })
            self.polygons = scaled_polygons
        # Resize image for display (consider zoom level)
        if hasattr(self, 'original_image'):
            if self.original_image is None:
                self.image = None # Original image is missing
            else:
                disp_w = int(new_width * self.zoom_level)
                disp_h = int(new_height * self.zoom_level)
                if disp_w > 0 and disp_h > 0: # Ensure dimensions are positive
                    try:
                        self.image = cv2.resize(self.original_image, (disp_w, disp_h))
                    except cv2.error as e:
                        # Log the error or show a message if appropriate
                        print(f"OpenCV error during resize in on_canvas_resize: {e}")
                        self.image = None # Set image to None on resize error
                else:
                    self.image = None # Invalid dimensions for resize
        else:
            self.image = None # No original image to resize

        # Update stored canvas size
        self.canvas_width, self.canvas_height = new_width, new_height
        # Redraw
        self.display_image()
        self.display_annotations()

    def delete_annotation(self, index, annotation_type):
        """
        Deletes the annotation (bbox or polygon) at 'index' of the specified type.
        """
        if annotation_type == 'bbox':
            if 0 <= index < len(self.bboxes):
                del self.bboxes[index]
        elif annotation_type == 'polygon':
            if 0 <= index < len(self.polygons):
                del self.polygons[index]
        self.display_annotations()
        self.save_history()

    # --------------------------------------------------
    # Copy/Paste Features
    # --------------------------------------------------

    def copy_bbox(self, bbox):
        """
        Copies a single bounding box to a local list for potential pasting.
        """
        self.copied_bbox_list.append(bbox)
        self.update_copied_bbox_display()

    def paste_all_bboxes(self):
        """
        Pastes all copied bounding boxes into the current image (appends to self.bboxes).
        """
        if self.copied_bbox_list:
            # You could also adjust positions if you want them offset or at exact coords
            for bbox in self.copied_bbox_list:
                self.bboxes.append(bbox)
            self.display_annotations()
            self.save_history()
        else:
            messagebox.showinfo("Info", "No bounding boxes copied to paste.")

    def update_copied_bbox_display(self):
        """
        Refreshes the copy_frame with the list of bounding boxes currently copied.
        """
        for widget in self.copy_frame.winfo_children():
            widget.destroy()

        if not self.copied_bbox_list:
            tk.Label(self.copy_frame, text="Copied Bounding Boxes: None", font=("Arial", 12)).pack(pady=10)
        else:
            for bbox in self.copied_bbox_list:
                x, y, w, h, class_id = bbox
                label_text = f"Class {self.class_names[class_id]}, ({x}, {y}), ({w}, {h})"
                tk.Label(self.copy_frame, text=label_text, font=("Arial", 12)).pack(pady=5)

    # --------------------------------------------------
    # Save / Delete Image
    # --------------------------------------------------

    def save_labels(self, *args):
        """
        Saves the current bounding boxes to a .txt file in YOLO format.
        """
        if not self.image_path or self.current_image_index == -1: # Don't save for manually loaded images
            return

        # Construct label path based on relative image path
        relative_image_path = os.path.relpath(self.image_path, self.folder_path)
        label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)

        # Ensure label directory exists
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        write_annotations_to_file(label_path, self.bboxes, self.polygons, self.image.shape)

        print(f"Saved labels for {self.image_path}")

        new_status = "edited" if (self.bboxes or self.polygons) else "viewed"
        self.image_status[relative_image_path] = new_status
        self.image_tree.item(relative_image_path, tags=(new_status,))
        self.save_statuses()
        self.update_status_labels()

    def delete_image(self):
        """
        Deletes the current image file and its label file from disk, then updates the UI.
        """
        if self.current_image_index == -1:
            messagebox.showwarning("Warning", "No image selected to delete.")
            return

        if self.current_image_index == -1:
            messagebox.showwarning("Warning", "Cannot delete manually loaded image.")
            return

        relative_image_path = self.image_files[self.current_image_index]
        image_path = os.path.join(self.folder_path, relative_image_path)
        label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_relative_path)

        if not messagebox.askyesno("Confirm Delete", f"Delete {relative_image_path} and its label?"):
            return

        try:
            os.remove(image_path)
            if os.path.exists(label_path):
                os.remove(label_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error deleting files: {e}")
            return

        # Remove from internal lists and UI
        del self.image_files[self.current_image_index]
        self.image_tree.delete(relative_image_path) # Use relative path as iid
        
        # Remove from status dictionary
        if relative_image_path in self.image_status:
            del self.image_status[relative_image_path]

        self.canvas.delete("all")
        self.image_name_label.config(text="")
        self.bboxes = [] # Clear bboxes
        self.polygons = [] # Clear polygons
        
        # Recreate bbox_info_frame inside the info_canvas
        for widget in self.bbox_info_frame.winfo_children():
            widget.destroy()
        self.bbox_info_frame.destroy() # Destroy the old frame
        self.bbox_info_frame = tk.Frame(self.info_canvas)
        self.info_canvas.create_window((0, 0), window=self.bbox_info_frame, anchor="nw")
        self.info_canvas.configure(scrollregion=self.info_canvas.bbox("all")) # Reset scroll region

        if self.image_files:
            self.current_image_index = min(self.current_image_index, len(self.image_files) - 1)
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
        else:
            self.current_image_index = -1

        self.update_status_labels()
        self.save_history() # Save history after image deletion

    # --------------------------------------------------
    # Class/Editor Utilities
    # --------------------------------------------------

    def clear_class_selection(self):
        """
        Clears any selected class in the listbox and resets the copied bounding box list.
        """
        self.class_listbox.selection_clear(0, tk.END)
        self.selected_class_index = None
        self.copied_bbox_list = []
        self.update_copied_bbox_display()
        self.root.focus_set()

    def on_key_press(self, event):
        """
        If the user presses a digit (1-9), select that class in the listbox.
        """
        if event.char.isdigit():
            idx = int(event.char) - 1
            if 0 <= idx < len(self.class_names):
                self.class_listbox.selection_clear(0, tk.END)
                self.class_listbox.selection_set(idx)
                self.selected_class_index = idx

    # --------------------------------------------------
    # YOLO Model Loading / Auto-Annotation
    # --------------------------------------------------

    def load_model(self):
        """
        Allows the user to browse for a YOLO .pt model and load it into memory for inference.
        """
        model_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")]
        )
        if model_path:
            try:
                self.model = YOLO(model_path)
                messagebox.showinfo("Success", f"Model loaded successfully from {model_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def auto_annotate_dataset_threaded(self):
        """
        Spawns a separate thread to auto-annotate images using the loaded model,
        while showing a progress bar in a modal dialog.
        """
        if self.model is None:
            messagebox.showerror("Model Not Loaded", "Please load a YOLO model first.")
            return

        self.auto_annotate_button.config(state=tk.DISABLED)

        # Create a modal progress window
        self.progress_win = tk.Toplevel(self.root)
        self.progress_win.title("Auto Annotation Progress")
        self.progress_win.transient(self.root)
        self.progress_win.grab_set()

        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(self.progress_win, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(padx=20, pady=10)

        # Progress label
        self.progress_label = tk.Label(self.progress_win, text="0/0 images processed")
        self.progress_label.pack(pady=5)

        # Cancel button
        self.cancel_button = tk.Button(self.progress_win, text="Cancel", command=self.cancel_annotation)
        self.cancel_button.pack(pady=5)

        # Center the progress window
        self.progress_win.update_idletasks()
        main_width = self.root.winfo_width()
        main_height = self.root.winfo_height()
        main_x = self.root.winfo_x()
        main_y = self.root.winfo_y()
        progress_width = 300
        progress_height = 100
        x = main_x + (main_width - progress_width) // 2
        y = main_y + (main_height - progress_height) // 2
        self.progress_win.geometry(f"{progress_width}x{progress_height}+{x}+{y}")

        # Initialize cancellation flag
        self.cancel_event = threading.Event()

        # Start the annotation in a background thread
        threading.Thread(target=self.auto_annotate_dataset, daemon=True).start()

    def cancel_annotation(self):
        """
        Sets the cancellation event so the auto-annotation thread can stop.
        """
        if self.cancel_event:
            self.cancel_event.set()
        if hasattr(self, 'progress_win') and self.progress_win.winfo_exists():
            self.progress_win.destroy()
        self.auto_annotate_button.config(state=tk.NORMAL)

    def update_progress(self, percent, current, total):
        """
        Updates the progress bar and label in the auto-annotation dialog.
        """
        if hasattr(self, 'progress_win') and self.progress_win.winfo_exists():
            self.progress_var.set(percent)
            if hasattr(self, 'progress_label') and self.progress_label.winfo_exists():
                self.progress_label.config(text=f"{current}/{total} images processed")
            self.progress_win.update_idletasks()

    def auto_annotate_dataset(self):
        """
        The main logic for auto-annotating each image in the dataset. Runs in a background thread.
        """
        conf_threshold = self.confidence_threshold.get()
        flagged_images = []
        total_images = len(self.image_files)
        processed_count = 0

        try:
            for idx, image_file in enumerate(self.image_files):
                processed_count = idx + 1

                if self.cancel_event and self.cancel_event.is_set():
                    break  # Exit loop if cancellation requested

                image_path = os.path.join(self.folder_path, image_file)
                label_filename = os.path.splitext(image_file)[0] + '.txt'
                label_path = os.path.join(self.label_folder, label_filename)

                # Run model inference
                results = self.model(image_path, conf=conf_threshold, verbose=False)
                detections = results[0].boxes  # bounding boxes from YOLO

                bboxes = []
                uncertain = False
                img_h, img_w = None, None

                # If no detections, set status to 'viewed'
                # Use the relative path as the item ID for status tracking
                relative_image_path = image_file

                if not detections:
                    self.image_status[relative_image_path] = "viewed"
                else:
                    # Gather bounding boxes
                    for box in detections:
                        if self.cancel_event.is_set():
                            break  # Early exit for cancellation
 
                        conf_score = box.conf[0].item()
                        class_id = int(box.cls[0])
 
                        if img_h is None or img_w is None:
                            # Extract original shape from YOLO results
                            img_h, img_w = results[0].orig_shape[:2]
 
                        if class_id >= len(self.class_names):
                            # Skip detection of classes not in our list
                            continue
 
                        # Convert from normalized coords to absolute pixel coords
                        x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
                        x_center_abs = x_center * img_w
                        y_center_abs = y_center * img_h
                        width_abs = width * img_w
                        height_abs = height * img_h
 
                        x_min = int(x_center_abs - width_abs / 2)
                        y_min = int(y_center_abs - height_abs / 2)
 
                        bboxes.append((x_min, y_min, int(width_abs), int(height_abs), class_id, conf_score))
 
                        if conf_score < conf_threshold:
                            uncertain = True
 
                    if bboxes:
                        if uncertain:
                            flagged_images.append(relative_image_path)
                            self.image_status[relative_image_path] = "review_needed"
                        else:
                            # Write bounding boxes to file
                            # Read existing annotations to preserve polygons if any
                            existing_bboxes, existing_polygons = read_annotations_from_file(label_path, (img_h, img_w))
                            

                            # Combine new bboxes with existing polygons
                            # Note: This assumes auto-annotation only adds bboxes, not replaces existing polygons.
                            # If auto-annotation should overwrite all previous annotations, clear existing_bboxes too.
                            new_bboxes_for_file = []
                            for (x, y, w, h, cid, _) in bboxes:
                                new_bboxes_for_file.append((x, y, w, h, cid))
 
                            write_annotations_to_file(label_path, new_bboxes_for_file, existing_polygons, (img_h, img_w))
                            self.image_status[relative_image_path] = "edited"
                    else:
                        # If YOLO found detections but we skipped them all
                        self.image_status[relative_image_path] = "viewed"

                # Update progress
                progress_percent = (processed_count / total_images) * 100
                self.root.after(0, self.update_progress, progress_percent, processed_count, total_images)

                # Handle cancellation during processing
                if self.cancel_event.is_set():
                    break

        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Error", f"Annotation failed: {str(e)}"))
        finally:
            # Cleanup / finalize
            self.save_statuses()
            self.root.after(0, self.update_status_labels)

            if hasattr(self, 'progress_win') and self.progress_win.winfo_exists():
                self.root.after(0, self.progress_win.destroy)

            self.root.after(0, lambda: self.auto_annotate_button.config(state=tk.NORMAL))

            # Refresh the treeview tags with updated statuses
            for relative_image_path in self.image_files:
                self.image_tree.item(relative_image_path, tags=(self.image_status.get(relative_image_path, "not_viewed"),))

            if self.cancel_event.is_set():
                self.root.after(0, lambda: messagebox.showinfo(
                    "Cancelled",
                    f"Annotation cancelled. Processed {processed_count}/{total_images} images."
                ))
            elif flagged_images:
                self.root.after(0, lambda: messagebox.showwarning(
                    "Review Needed",
                    f"{len(flagged_images)} images have low-confidence detections requiring review."
                ))
            else:
                self.root.after(0, lambda: messagebox.showinfo(
                    "Complete",
                    "Auto-annotation finished successfully!"
                ))

    # --------------------------------------------------
    # YAML Export Window
    # --------------------------------------------------

    def export_yaml_window(self):
        """
        Opens a configuration window for exporting a new dataset YAML, optionally splitting
        the dataset into train/val/test, or performing in-sample validation.
        """
        export_win = tk.Toplevel(self.root)
        export_win.title("Export YAML Settings")
        export_win.transient(self.root)
        export_win.grab_set()

        # Export to current or custom
        export_to_current_var = tk.BooleanVar(value=True)

        def toggle_custom(*args):
            if export_to_current_var.get():
                custom_export_entry.config(state="disabled")
                custom_export_button.config(state="disabled")
            else:
                custom_export_entry.config(state="normal")
                custom_export_button.config(state="normal")

        export_to_current_var.trace("w", toggle_custom)

        current_frame = tk.Frame(export_win)
        current_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Checkbutton(
            current_frame,
            text="Export to Current Dataset Location",
            variable=export_to_current_var
        ).pack(side=tk.LEFT)

        custom_frame = tk.Frame(export_win)
        custom_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(custom_frame, text="Custom Export Location:").pack(side=tk.LEFT)
        custom_export_entry = tk.Entry(custom_frame, width=40)
        custom_export_entry.pack(side=tk.LEFT, padx=5)
        custom_export_button = tk.Button(
            custom_frame,
            text="Browse",
            command=lambda: custom_export_entry.insert(0, filedialog.askdirectory(title="Select Export Folder") or "")
        )
        custom_export_button.pack(side=tk.LEFT)
        custom_export_entry.config(state="disabled")
        custom_export_button.config(state="disabled")

        # Split option: in-sample or full split
        split_option = tk.StringVar(value="in_sample")  # "in_sample" or "split"
        split_frame = tk.Frame(export_win)
        split_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(split_frame, text="Validation Mode:").pack(side=tk.LEFT)
        tk.Radiobutton(split_frame, text="In-Sample Validation", variable=split_option, value="in_sample").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(split_frame, text="Split Data", variable=split_option, value="split").pack(side=tk.LEFT, padx=5)

        # Test data option
        test_data_var = tk.BooleanVar(value=False)

        def toggle_test_data(*args):
            if split_option.get() == "split":
                test_data_check.config(state="normal")
            else:
                test_data_check.config(state="disabled")

        split_option.trace("w", toggle_test_data)

        test_data_frame = tk.Frame(export_win)
        test_data_frame.pack(fill=tk.X, padx=10, pady=5)
        test_data_check = tk.Checkbutton(test_data_frame, text="Include Test Data (6:2:2)", variable=test_data_var)
        test_data_check.pack(side=tk.LEFT)
        test_data_check.config(state="disabled")

        # Include validation in final YAML
        include_val_var = tk.BooleanVar(value=self.validation)
        val_frame = tk.Frame(export_win)
        val_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Checkbutton(
            val_frame,
            text="Include Validation in YAML",
            variable=include_val_var
        ).pack(side=tk.LEFT)

        # Buttons
        button_frame = tk.Frame(export_win)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def export_yaml():
            try:
                with open(self.yaml_path, "r") as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load YAML file:\n{e}")
                return

            # Adjust 'val' if user unchecks validation
            if not include_val_var.get():
                data["val"] = ""


            # Determine export folder
            if export_to_current_var.get():
                base_export_folder = self.folder_path
            else:
                base_export_folder = custom_export_entry.get().strip()
                if not base_export_folder:
                    messagebox.showerror("Error", "Please select a custom export location.")
                    return

            export_folder = os.path.join(base_export_folder, "exported")
            os.makedirs(export_folder, exist_ok=True)

            # Decide if in-sample or train/val(/test) split
            if split_option.get() == "in_sample":
                train_folder = os.path.join(export_folder, "train")
                os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
                os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)

                # In-sample => val is the same folder as train
                val_folder = train_folder
                data["train"] = train_folder
                data["val"] = val_folder

            else:
                # Identify images that have labeled bounding boxes or polygons
                annotated = []
                for relative_image_path in self.image_files:
                    label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
                    src_label_path = os.path.join(self.label_folder, label_relative_path)
                    if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
                        # Check if the file actually contains annotations (not just an empty file)
                        with open(src_label_path, 'r') as f:
                            if any(line.strip() for line in f): # Check if any line is not empty
                                annotated.append(relative_image_path)

                num_annotated = len(annotated)
                if num_annotated == 0:
                    messagebox.showwarning("Warning", "No annotated images found for export.")
                    return

                if test_data_var.get():
                    num_train = int(num_annotated * 0.6)
                    num_val = int(num_annotated * 0.2)
                    num_test = num_annotated - num_train - num_val
                    train_files = annotated[:num_train]
                    val_files = annotated[num_train:num_train + num_val]
                    test_files = annotated[num_train + num_val:]
                else:
                    num_train = int(num_annotated * 0.8)
                    train_files = annotated[:num_train]
                    val_files = annotated[num_train:]
                    test_files = []

                # Create subfolders
                train_folder = os.path.join(export_folder, "train")
                os.makedirs(os.path.join(train_folder, "images"), exist_ok=True)
                os.makedirs(os.path.join(train_folder, "labels"), exist_ok=True)

                val_folder = os.path.join(export_folder, "val")
                os.makedirs(os.path.join(val_folder, "images"), exist_ok=True)
                os.makedirs(os.path.join(val_folder, "labels"), exist_ok=True)

                data["train"] = train_folder
                data["val"] = val_folder

                if test_files:
                    test_folder = os.path.join(export_folder, "test")
                    os.makedirs(os.path.join(test_folder, "images"), exist_ok=True)
                    os.makedirs(os.path.join(test_folder, "labels"), exist_ok=True)
                    data["test"] = test_folder
                else:
                    data["test"] = ""


            # Write final dataset.yaml
            export_yaml_path = os.path.join(export_folder, "dataset.yaml")
            try:
                with open(export_yaml_path, "w") as f:
                    yaml.dump(data, f, sort_keys=False)
            except Exception as e:
                messagebox.showerror("Error", f"Could not export YAML:\n{e}")
                return

            # Copy files accordingly
            if split_option.get() == "in_sample":
                # Everything goes into 'train'
                all_labeled = []
                for relative_image_path in self.image_files:
                    label_relative_path = os.path.splitext(relative_image_path)[0] + '.txt'
                    src_label_path = os.path.join(self.label_folder, label_relative_path)
                    if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
                        with open(src_label_path, 'r') as f:
                            if any(line.strip() for line in f):
                                all_labeled.append(relative_image_path)

                copy_files_recursive(all_labeled, self.folder_path,
                           os.path.join(train_folder, "images"),
                           self.label_folder,
                           os.path.join(train_folder, "labels"))
            else:
                copy_files_recursive(train_files, self.folder_path,
                           os.path.join(train_folder, "images"),
                           self.label_folder,
                           os.path.join(train_folder, "labels"))
                copy_files_recursive(val_files, self.folder_path,
                           os.path.join(val_folder, "images"),
                           self.label_folder,
                           os.path.join(val_folder, "labels"))
                if test_files:
                    test_folder = os.path.join(export_folder, "test")
                    copy_files_recursive(test_files, self.folder_path,
                               os.path.join(test_folder, "images"),
                               self.label_folder,
                               os.path.join(test_folder, "labels"))

            messagebox.showinfo(
                "Success",
                f"Export complete!\nYAML and annotated images exported to:\n{export_folder}"
            )
            export_win.destroy()

        tk.Button(button_frame, text="Export", command=export_yaml).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, padx=5)    # --------------------------------------------------
    # Export Format Conversion Functions
    # --------------------------------------------------
    
    @staticmethod
    def convert_to_coco_format(image_files, all_bboxes, all_polygons, class_names, base_folder):
        """
        Converts annotations to COCO format.
        Returns a COCO-formatted dictionary.
        """
        coco_data = {
            "info": {
                "description": "Dataset exported from BBox & Polygon Annotator",
                "version": "1.0",
                "year": datetime.now().year,
                "contributor": "BBox & Polygon Annotator v9",
                "date_created": datetime.now().isoformat()
            },
            "licenses": [
                {
                    "id": 1,
                    "name": "Unknown",
                    "url": ""
                }
            ],
            "images": [],
            "annotations": [],
            "categories": []
        }
        
        # Add categories
        for i, class_name in enumerate(class_names):
            coco_data["categories"].append({
                "id": i,
                "name": class_name,
                "supercategory": "object"
            })
        
        annotation_id = 1
        
        for img_idx, image_path in enumerate(image_files):
            # Get image dimensions
            full_image_path = os.path.join(base_folder, image_path)
            if os.path.exists(full_image_path):
                import cv2
                img = cv2.imread(full_image_path)
                height, width = img.shape[:2]
            else:
                width, height = 640, 480  # Default if image not found
            
            # Add image info
            coco_data["images"].append({
                "id": img_idx,
                "width": width,
                "height": height,
                "file_name": os.path.basename(image_path)
            })
            
            # Add bounding box annotations
            if image_path in all_bboxes:
                for bbox in all_bboxes[image_path]:
                    x, y, w, h, class_id = bbox
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_id": class_id,
                        "bbox": [x, y, w, h],
                        "area": w * h,
                        "iscrowd": 0
                    })
                    annotation_id += 1
            
            # Add polygon annotations
            if image_path in all_polygons:
                for polygon in all_polygons[image_path]:
                    class_id = polygon['class_id']
                    points = polygon['points']
                    
                    # Convert points to flat list [x1, y1, x2, y2, ...]
                    segmentation = []
                    for x, y in points:
                        segmentation.extend([float(x), float(y)])
                    
                    # Calculate bounding box from polygon
                    xs = [p[0] for p in points]
                    ys = [p[1] for p in points]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    bbox_w, bbox_h = x_max - x_min, y_max - y_min
                   
                    # Calculate area (simple bounding box area)
                    area = bbox_w * bbox_h
                   
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": img_idx,
                        "category_id": class_id,
                        "segmentation": [segmentation],
                        "bbox": [x_min, y_min, bbox_w, bbox_h],
                        "area": area,                        "iscrowd": 0
                    })
                    annotation_id += 1
        
        return coco_data
    
    @staticmethod
    def convert_to_pascal_voc_format(image_path, bboxes, polygons, class_names, image_shape):
        """
        Converts annotations for a single image to Pascal VOC XML format.
        Returns XML string.
        """
        height, width = image_shape[:2]
        
        # Create root element
        annotation = ET.Element("annotation")
        
        # Add folder
        folder = ET.SubElement(annotation, "folder")
        folder.text = os.path.dirname(image_path) or "images"
        
        # Add filename
        filename = ET.SubElement(annotation, "filename")
        filename.text = os.path.basename(image_path)
        
        # Add source
        source = ET.SubElement(annotation, "source")
        database = ET.SubElement(source, "database")
        database.text = "BBox & Polygon Annotator"
        
        # Add size
        size = ET.SubElement(annotation, "size")
        width_elem = ET.SubElement(size, "width")
        width_elem.text = str(width)
        height_elem = ET.SubElement(size, "height")
        height_elem.text = str(height)
        depth = ET.SubElement(size, "depth")
        depth.text = "3"
        
        # Add segmented
        segmented = ET.SubElement(annotation, "segmented")
        segmented.text = "1" if polygons else "0"
        
        # Add bounding box objects
        for bbox in bboxes:
            x, y, w, h, class_id = bbox
            
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = class_names[class_id]
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(x))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(y))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(x + w))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(y + h))
        
        # Add polygon objects (as additional objects with polygon tag)
        for polygon in polygons:
            class_id = polygon['class_id']
            points = polygon['points']
            
            obj = ET.SubElement(annotation, "object")
            name = ET.SubElement(obj, "name")
            name.text = class_names[class_id]
            
            pose = ET.SubElement(obj, "pose")
            pose.text = "Unspecified"
            
            truncated = ET.SubElement(obj, "truncated")
            truncated.text = "0"
            
            difficult = ET.SubElement(obj, "difficult")
            difficult.text = "0"
            
            # Add bounding box from polygon
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            
            bndbox = ET.SubElement(obj, "bndbox")
            xmin = ET.SubElement(bndbox, "xmin")
            xmin.text = str(int(x_min))
            ymin = ET.SubElement(bndbox, "ymin")
            ymin.text = str(int(y_min))
            xmax = ET.SubElement(bndbox, "xmax")
            xmax.text = str(int(x_max))
            ymax = ET.SubElement(bndbox, "ymax")
            ymax.text = str(int(y_max))
            
            # Add polygon points
            polygon_elem = ET.SubElement(obj, "polygon")
            for i, (px, py) in enumerate(points):
                point = ET.SubElement(polygon_elem, f"point{i+1}")
                point.set("x", str(int(px)))
                point.set("y", str(int(py)))
        
        return ET.tostring(annotation, encoding='unicode')
    
    @staticmethod
    def convert_to_csv_format(image_files, all_bboxes, all_polygons, class_names):
        """
        Converts annotations to CSV format.
        Returns list of rows for CSV writing.
        """
        rows = []
        headers = ["image_name", "annotation_type", "class_name", "class_id", "coordinates", "area"]
        rows.append(headers)
        
        for image_path in image_files:
            image_name = os.path.basename(image_path)
            
            # Add bounding boxes
            if image_path in all_bboxes:
                for bbox in all_bboxes[image_path]:
                    x, y, w, h, class_id = bbox
                    coordinates = f"x={x},y={y},w={w},h={h}"
                    area = w * h
                    rows.append([
                        image_name, "bbox", class_names[class_id], class_id, 
                        coordinates, area
                    ])
            
            # Add polygons
            if image_path in all_polygons:
                for polygon in all_polygons[image_path]:
                    class_id = polygon['class_id']
                    points = polygon['points']
                    coordinates = ";".join([f"{x},{y}" for x, y in points])
                    
                    # Calculate approximate area
                    if len(points) >= 3:
                        # Shoelace formula for polygon area
                        area = 0.5 * abs(sum(points[i][0] * (points[(i+1) % len(points)][1] - points[i-1][1]) 
                                           for i in range(len(points))))
                    else:
                        area = 0
                    
                    rows.append([
                        image_name, "polygon", class_names[class_id], class_id,
                        coordinates, area
                    ])
        
        return rows

    def export_format_selection_window(self):
        """
        Opens a window for selecting export format and configuring export options.
        """
        export_win = tk.Toplevel(self.root)
        export_win.title("Export Annotations")
        export_win.transient(self.root)
        export_win.grab_set()
        center_window(export_win, 500, 450)

        # Format selection
        format_frame = tk.LabelFrame(export_win, text="Export Format")
        format_frame.pack(fill=tk.X, padx=10, pady=5)

        format_var = tk.StringVar(value="yaml")
        formats = [
            ("YAML (YOLO Training)", "yaml"),
            ("COCO JSON", "coco"),
            ("Pascal VOC XML", "voc"),
            ("CSV Spreadsheet", "csv"),
            ("JSON (Generic)", "json")
        ]

        for text, value in formats:
            tk.Radiobutton(format_frame, text=text, variable=format_var, value=value).pack(anchor=tk.W, padx=5, pady=2)

        # Export location
        location_frame = tk.LabelFrame(export_win, text="Export Location")
        location_frame.pack(fill=tk.X, padx=10, pady=5)

        export_to_current_var = tk.BooleanVar(value=True)
        tk.Checkbutton(location_frame, text="Export to Current Dataset Location", 
                      variable=export_to_current_var).pack(anchor=tk.W, padx=5, pady=2)

        custom_frame = tk.Frame(location_frame)
        custom_frame.pack(fill=tk.X, padx=5, pady=2)
        tk.Label(custom_frame, text="Custom Location:").pack(side=tk.LEFT)
        custom_export_entry = tk.Entry(custom_frame, width=40)
        custom_export_entry.pack(side=tk.LEFT, padx=5)
        custom_export_button = tk.Button(custom_frame, text="Browse",
            command=lambda: custom_export_entry.insert(0, filedialog.askdirectory(title="Select Export Folder") or ""))
        custom_export_button.pack(side=tk.LEFT)

        def toggle_custom(*args):
            if export_to_current_var.get():
                custom_export_entry.config(state="disabled")
                custom_export_button.config(state="disabled")
            else:
                custom_export_entry.config(state="normal")
                custom_export_button.config(state="normal")

        export_to_current_var.trace("w", toggle_custom)
        toggle_custom()

        # YAML-specific options (only shown when YAML is selected)
        yaml_frame = tk.LabelFrame(export_win, text="YAML Options (YOLO Training)")
        yaml_frame.pack(fill=tk.X, padx=10, pady=5)

        split_option = tk.StringVar(value="in_sample")
        tk.Radiobutton(yaml_frame, text="In-Sample Validation", 
                      variable=split_option, value="in_sample").pack(anchor=tk.W, padx=5, pady=2)
        tk.Radiobutton(yaml_frame, text="Split Data (Train/Val/Test)", 
                      variable=split_option, value="split").pack(anchor=tk.W, padx=5, pady=2)

        test_data_var = tk.BooleanVar(value=False)
        test_data_check = tk.Checkbutton(yaml_frame, text="Include Test Data (60:20:20 split)", 
                                        variable=test_data_var)
        test_data_check.pack(anchor=tk.W, padx=20, pady=2)

        include_val_var = tk.BooleanVar(value=self.validation)
        tk.Checkbutton(yaml_frame, text="Include Validation in YAML", 
                      variable=include_val_var).pack(anchor=tk.W, padx=5, pady=2)

        def toggle_yaml_options(*args):
            if format_var.get() == "yaml":
                yaml_frame.pack(fill=tk.X, padx=10, pady=5)
                if split_option.get() == "split":
                    test_data_check.config(state="normal")
                else:
                    test_data_check.config(state="disabled")
            else:
                yaml_frame.pack_forget()

        def toggle_test_data(*args):
            if split_option.get() == "split" and format_var.get() == "yaml":
                test_data_check.config(state="normal")
            else:
                test_data_check.config(state="disabled")

        format_var.trace("w", toggle_yaml_options)
        split_option.trace("w", toggle_test_data)
        toggle_yaml_options()

        # Export options
        options_frame = tk.LabelFrame(export_win, text="Export Options")
        options_frame.pack(fill=tk.X, padx=10, pady=5)

        include_images_var = tk.BooleanVar(value=True)
        tk.Checkbutton(options_frame, text="Copy Images to Export Folder", 
                      variable=include_images_var).pack(anchor=tk.W, padx=5, pady=2)

        include_unannotated_var = tk.BooleanVar(value=False)
        tk.Checkbutton(options_frame, text="Include Unannotated Images", 
                      variable=include_unannotated_var).pack(anchor=tk.W, padx=5, pady=2)

        # Buttons
        button_frame = tk.Frame(export_win)
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        def perform_export():
            selected_format = format_var.get()
            
            # Determine export folder
            if export_to_current_var.get():
                base_export_folder = self.folder_path
            else:
                base_export_folder = custom_export_entry.get().strip()
                if not base_export_folder:
                    messagebox.showerror("Error", "Please select a custom export location.")
                    return

            if selected_format == "yaml":
                # Use existing YAML export functionality
                export_win.destroy()
                self.export_yaml_window()
            else:
                # Use new format export functionality
                self.export_to_format(
                    selected_format, base_export_folder, 
                    include_images_var.get(), include_unannotated_var.get()
                )
                export_win.destroy()

        tk.Button(button_frame, text="Export", command=perform_export).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, padx=5)

    def export_to_format(self, format, base_folder, copy_images, include_unannotated):
        """
        Exports the annotations to the selected format (COCO, Pascal VOC, CSV, or JSON).
        """
        try:
            # Prepare output directories
            os.makedirs(base_folder, exist_ok=True)
            
            # Collect all annotations from all images
            all_bboxes = {}
            all_polygons = {}
            annotated_images = []
            
            for image_path in self.image_files:
                label_relative_path = os.path.splitext(image_path)[0] + '.txt'
                label_full_path = os.path.join(self.label_folder, label_relative_path)
                
                if os.path.exists(label_full_path) and os.path.getsize(label_full_path) > 0:
                    # Read annotations from file
                    full_image_path = os.path.join(self.folder_path, image_path)
                    if os.path.exists(full_image_path):
                        import cv2
                        img = cv2.imread(full_image_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            bboxes, polygons = read_annotations_from_file(label_full_path, (height, width))
                            
                            if bboxes or polygons:
                                all_bboxes[image_path] = bboxes
                                all_polygons[image_path] = polygons
                                annotated_images.append(image_path)
            
            # Group images by annotation status
            unannotated_images = set(self.image_files) - set(annotated_images)

            if format == "coco":
                # Convert annotations to COCO format
                coco_data = BoundingBoxEditor.convert_to_coco_format(
                    self.image_files, 
                    all_bboxes,
                    all_polygons,
                    self.class_names,
                    self.folder_path
                )

                # Save COCO JSON file
                with open(os.path.join(base_folder, "annotations.json"), "w") as f:
                    json.dump(coco_data, f, indent=2)

                messagebox.showinfo("Export Complete", "COCO format export successful.")

            elif format == "voc":
                # Pascal VOC export: one XML file per image
                for image_path in annotated_images:
                    bboxes = all_bboxes.get(image_path, [])
                    polygons = all_polygons.get(image_path, [])
                    
                    # Get image dimensions
                    full_image_path = os.path.join(self.folder_path, image_path)
                    if os.path.exists(full_image_path):
                        import cv2
                        img = cv2.imread(full_image_path)
                        if img is not None:
                            height, width = img.shape[:2]
                            xml_str = BoundingBoxEditor.convert_to_pascal_voc_format(
                                image_path, bboxes, polygons, self.class_names, (height, width)
                            )

                            # Save XML file
                            image_name = os.path.splitext(os.path.basename(image_path))[0]
                            xml_file_path = os.path.join(base_folder, f"{image_name}.xml")
                            with open(xml_file_path, "w") as xml_file:
                                xml_file.write(xml_str)

                messagebox.showinfo("Export Complete", "Pascal VOC format export successful.")

            elif format == "csv":
                # CSV export: single CSV file for all annotations
                csv_rows = BoundingBoxEditor.convert_to_csv_format(
                    self.image_files, all_bboxes, all_polygons, self.class_names
                )

                # Save CSV file
                csv_file_path = os.path.join(base_folder, "annotations.csv")
                with open(csv_file_path, "w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerows(csv_rows)

                messagebox.showinfo("Export Complete", "CSV format export successful.")

            elif format == "json":
                # Generic JSON export: all annotations in a single JSON file
                json_data = {
                    "images": [],
                    "annotations": [],
                    "categories": [{"id": i, "name": name} for i, name in enumerate(self.class_names)]
                }

                # Add image entries and annotations
                annotation_id = 1
                for img_idx, image_path in enumerate(self.image_files):
                    # Get image dimensions
                    full_image_path = os.path.join(self.folder_path, image_path)
                    width, height = 640, 480  # Default
                    if os.path.exists(full_image_path):
                        import cv2
                        img = cv2.imread(full_image_path)
                        if img is not None:
                            height, width = img.shape[:2]
                    
                    json_data["images"].append({
                        "id": img_idx,
                        "file_name": os.path.basename(image_path),
                        "width": width,
                        "height": height
                    })

                    # Add bounding box annotations
                    if image_path in all_bboxes:
                        for bbox in all_bboxes[image_path]:
                            x, y, w, h, class_id = bbox
                            json_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_idx,
                                "category_id": class_id,
                                "bbox": [x, y, w, h],
                                "area": w * h,
                                "type": "bbox"
                            })
                            annotation_id += 1
                    
                    # Add polygon annotations
                    if image_path in all_polygons:
                        for polygon in all_polygons[image_path]:
                            class_id = polygon['class_id']
                            points = polygon['points']
                            json_data["annotations"].append({
                                "id": annotation_id,
                                "image_id": img_idx,
                                "category_id": class_id,
                                "polygon": points,
                                "type": "polygon"
                            })
                            annotation_id += 1

                # Save JSON file
                json_file_path = os.path.join(base_folder, "annotations.json")
                with open(json_file_path, "w") as json_file:
                    json.dump(json_data, json_file, indent=2)

                messagebox.showinfo("Export Complete", "JSON format export successful.")

            # Optionally copy images to export folder
            if copy_images:
                images_to_copy = annotated_images
                if include_unannotated:
                    images_to_copy = self.image_files
                
                for image_path in images_to_copy:
                    src_image_path = os.path.join(self.folder_path, image_path)
                    dst_image_path = os.path.join(base_folder, os.path.basename(image_path))
                    if os.path.exists(src_image_path):
                        shutil.copy(src_image_path, dst_image_path)

        except Exception as e:
            messagebox.showerror("Export Error", f"An error occurred during export:\n{e}")

    def reload_classes_from_yaml(self):
        """
        Reloads class names from the YAML file and updates the class listbox and display.
        Useful when YAML file has been modified externally.
        """
        try:
            with open(self.yaml_path, "r") as f:
                data = yaml.safe_load(f)
            
            # Load class names from YAML
            raw_names = data.get("names", ["person"])
            if isinstance(raw_names, dict):
                # Convert dict to list sorted by integer keys
                new_class_names = [raw_names[k] for k in sorted(raw_names.keys(), key=lambda x: int(x))]
            else:
                new_class_names = raw_names
            
            # Update class names and UI
            self.class_names = new_class_names
            
            # Refresh class listbox
            self.class_listbox.delete(0, tk.END)
            for class_name in self.class_names:
                self.class_listbox.insert(tk.END, class_name)
            
            # Update colors and display
            self.update_class_colors()
            self.display_annotations()
            
            messagebox.showinfo("Classes Reloaded", f"Successfully reloaded {len(self.class_names)} classes from YAML file.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to reload classes from YAML: {str(e)}")


# --------------------------------------------------
# ProjectManager Class
# --------------------------------------------------

class ProjectManager:
    """
    Manages creating, opening, and deleting projects.
    Displays projects in an integrated panel.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Project Manager")
        center_window(self.root, 750, 500)  # Adjusted size

        self.main_container = ttk.Frame(root, padding="10")
        self.main_container.pack(fill=tk.BOTH, expand=True)

        # PanedWindow for resizable layout
        self.paned_window = ttk.PanedWindow(self.main_container, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Action Buttons
        self.actions_frame = ttk.Labelframe(self.paned_window, text="Actions", padding="10")
        self.paned_window.add(self.actions_frame, weight=1) # Smaller weight for actions panel

        self.new_project_button = ttk.Button(self.actions_frame, text="New Project", command=self.new_project)
        self.new_project_button.pack(pady=5, fill=tk.X)

        self.open_selected_button = ttk.Button(self.actions_frame, text="Open Selected", command=self._open_selected_project_action, state=tk.DISABLED)
        self.open_selected_button.pack(pady=5, fill=tk.X)
        
        self.refresh_button = ttk.Button(self.actions_frame, text="Refresh List", command=self._populate_project_list)
        self.refresh_button.pack(pady=5, fill=tk.X)

        self.delete_selected_button = ttk.Button(self.actions_frame, text="Delete Selected", command=self._delete_selected_project_action, state=tk.DISABLED)
        self.delete_selected_button.pack(pady=5, fill=tk.X)
        
        ttk.Button(self.actions_frame, text="Quit", command=self.root.quit).pack(pady=20, fill=tk.X, side=tk.BOTTOM)

        # Right Pane: Project List
        self.projects_list_frame = ttk.Labelframe(self.paned_window, text="Existing Projects", padding="10")
        self.paned_window.add(self.projects_list_frame, weight=3) # Larger weight for project list

        columns = ("project_name", "dataset_path", "last_modified_date")
        self.project_tree = ttk.Treeview(
            self.projects_list_frame,
            columns=columns,
            show="headings",
            selectmode="browse"
        )
        self.project_tree.heading("project_name", text="Project Name")
        self.project_tree.column("project_name", anchor=tk.W, width=200, stretch=tk.NO)
        self.project_tree.heading("dataset_path", text="Dataset Path")
        self.project_tree.column("dataset_path", anchor=tk.W, width=300) # Allow dataset_path to expand
        self.project_tree.heading("last_modified_date", text="Last Modified Date")
        self.project_tree.column("last_modified_date", anchor=tk.W, width=150, stretch=tk.NO)
        
        self.project_tree_scrollbar = ttk.Scrollbar(self.projects_list_frame, orient="vertical", command=self.project_tree.yview)
        self.project_tree.configure(yscrollcommand=self.project_tree_scrollbar.set)
        
        self.project_tree_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.project_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.project_tree.bind("<<TreeviewSelect>>", self._on_project_select)
        self.project_tree.bind("<Double-1>", self._open_selected_project_action)

        self._populate_project_list()

    def _populate_project_list(self):
        """Populates the project treeview with projects from PROJECTS_DIR."""
        for item in self.project_tree.get_children():
            self.project_tree.delete(item)

        project_files = [f for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]
        if not project_files:
            self.project_tree.insert("", tk.END, iid="no_projects_placeholder", values=("No projects found.", "", ""), tags=("placeholder",))
            self._on_project_select() # Ensure buttons are disabled
            return

        for f_name in sorted(project_files):
            project_name_display = os.path.splitext(f_name)[0]
            dataset_path_display = "N/A"
            full_path = os.path.join(PROJECTS_DIR, f_name)
            try:
                last_modified_display = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime("%Y-%m-%d %H:%M:%S")
            except Exception as e:
                logging.error(f"Error getting last modified time for project file {f_name}", exc_info=True)
                last_modified_display = ""
            try:
                with open(full_path, "r") as f:
                    project_data = json.load(f)
                    dataset_path_display = project_data.get("dataset_path", "N/A")
            except Exception as e:
                logging.error(f"Error reading project file {f_name}", exc_info=True)
                messagebox.showerror(
                    "Error Reading Project",
                    f"Error reading project file {f_name}:\n{e}"
                )

            self.project_tree.insert("", tk.END, iid=f_name, values=(project_name_display, dataset_path_display, last_modified_display))
        self._on_project_select() 

    def _on_project_select(self, event=None):
        """Handles selection changes in the project treeview."""
        selected_item_ids = self.project_tree.selection()
        if selected_item_ids:
            # Check if the selected item is not the placeholder
            first_selected_iid = selected_item_ids[0]
            if first_selected_iid != "no_projects_placeholder":
                self.open_selected_button.config(state=tk.NORMAL)
                self.delete_selected_button.config(state=tk.NORMAL)
                return
        
        self.open_selected_button.config(state=tk.DISABLED)
        self.delete_selected_button.config(state=tk.DISABLED)

    def _open_selected_project_action(self, event=None):
        """Loads the selected project and opens the editor."""
        selected_item_ids = self.project_tree.selection()
        if not selected_item_ids:
            messagebox.showwarning("No Project Selected", "Please select a project from the list to open.")
            return
        
        project_file_iid = selected_item_ids[0]
        if project_file_iid == "no_projects_placeholder":
             messagebox.showwarning("No Project Selected", "Please create or select a valid project.")
             return

        full_path = os.path.join(PROJECTS_DIR, project_file_iid)
        try:
            with open(full_path, "r") as f:
                project = json.load(f)
            
            self.root.destroy() 
            self.open_editor(project) 
        except Exception as e:
            messagebox.showerror("Error Opening Project", f"Could not load project '{project_file_iid}':\n{e}")
            self._populate_project_list()

    def _delete_selected_project_action(self):
        """Deletes the selected project's .json file."""
        selected_item_ids = self.project_tree.selection()
        if not selected_item_ids:
            messagebox.showwarning("No Project Selected", "Please select a project to delete.")
            return

        project_file_iid = selected_item_ids[0]
        if project_file_iid == "no_projects_placeholder":
             messagebox.showwarning("No Project Selected", "Cannot delete placeholder item.")
             return
             
        project_name_display = self.project_tree.item(project_file_iid)['values'][0]

        if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete project '{project_name_display}'?\nThis will delete the project file ({project_file_iid}) but NOT the dataset itself."):
            full_path = os.path.join(PROJECTS_DIR, project_file_iid)
            try:
                os.remove(full_path)
                messagebox.showinfo("Project Deleted", f"Project '{project_name_display}' deleted successfully.")
            except Exception as e:
                messagebox.showerror("Error Deleting Project", f"Could not delete project file '{project_file_iid}':\n{e}")
            finally:
                self._populate_project_list() 

    def new_project(self):
        """
        Opens a dialog to create a new project: user specifies project name + dataset path.
        """
        new_win = tk.Toplevel(self.root)
        new_win.transient(self.root)
        new_win.grab_set()
        new_win.title("New Project")

        # Use ttk style for consistency
        form_frame = ttk.Frame(new_win, padding="10")
        form_frame.pack(expand=True, fill=tk.BOTH)

        ttk.Label(form_frame, text="Project Name:").grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        name_entry = ttk.Entry(form_frame, width=40)
        name_entry.grid(row=0, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        ttk.Label(form_frame, text="Dataset Path:").grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        dataset_entry = ttk.Entry(form_frame, width=40)
        dataset_entry.grid(row=1, column=1, padx=5, pady=5, sticky=tk.EW)

        def browse_dataset():
            folder = filedialog.askdirectory(title="Select Dataset Folder")
            if folder:
                dataset_entry.delete(0, tk.END)
                dataset_entry.insert(0, folder)

        ttk.Button(form_frame, text="Browse", command=browse_dataset).grid(row=1, column=2, padx=5, pady=5)

        buttons_frame = ttk.Frame(form_frame)
        buttons_frame.grid(row=2, column=0, columnspan=3, pady=10)
        
        def create_project_action():
            project_name = name_entry.get().strip()
            dataset_path = dataset_entry.get().strip()
            if not project_name or not dataset_path:
                messagebox.showerror("Error", "Project name and dataset path are required.", parent=new_win)
                return
            
            if not os.path.isdir(dataset_path):
                messagebox.showerror("Error", "Dataset path must be a valid directory.", parent=new_win)
                return

            project = {
                "project_name": project_name,
                "dataset_path": dataset_path,
                "label_path": os.path.join(dataset_path, "labels") # Consistent with BoundingBoxEditor
            }
            # Sanitize project_name for use as a filename
            safe_project_filename = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in project_name).rstrip()
            if not safe_project_filename:
                safe_project_filename = "Untitled_Project"
            project_file = os.path.join(PROJECTS_DIR, f"{safe_project_filename}.json")

            if os.path.exists(project_file):
                if not messagebox.askyesno("Overwrite Project?", f"Project '{safe_project_filename}.json' already exists. Overwrite?", parent=new_win):
                    return

            try:
                with open(project_file, "w") as f:
                    json.dump(project, f, indent=4)
                messagebox.showinfo("Project Created", f"Project '{project_name}' created successfully as '{safe_project_filename}.json'.", parent=new_win)
                new_win.destroy()
                self._populate_project_list() # Refresh the list in ProjectManager
                # Do not automatically open the editor, let user open from the list
                # self.root.destroy() 
                # self.open_editor(project)
            except Exception as e:
                messagebox.showerror("Error Creating Project", f"Could not save project file:\n{e}", parent=new_win)


        ttk.Button(buttons_frame, text="Create Project", command=create_project_action).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Cancel", command=new_win.destroy).pack(side=tk.LEFT, padx=5)
        
        form_frame.columnconfigure(1, weight=1) # Make entry expand
        center_window(new_win, 500, 150) # Adjusted size for new project dialog
        name_entry.focus_set()


    def open_editor(self, project):
        """
        Launches the BoundingBoxEditor in a new Tk window with the chosen project loaded.
        """
        editor_root = tk.Tk()
        BoundingBoxEditor(editor_root, project)
        editor_root.mainloop()

# --------------------------------------------------
# Main Execution
# --------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    # Global exception handler for Tkinter callbacks
    def report_callback_exception(self, exc, val, tb):
        logging.error("Exception in Tkinter callback", exc_info=(exc, val, tb))
        messagebox.showerror("Error", f"An unexpected error occurred:\n{val}")
    tk.Tk.report_callback_exception = report_callback_exception
    # Apply a theme
    style = ttk.Style(root)
    available_themes = style.theme_names()
    # print(f"Available themes: {available_themes}") # For debugging
    if "clam" in available_themes: # 'clam', 'alt', 'default', 'classic' are common
        style.theme_use("clam")
    elif "vista" in available_themes and os.name == 'nt': # Good for Windows
         style.theme_use("vista")
    # else, it will use the default system theme

    try:
        pm = ProjectManager(root)
        root.mainloop()
    except Exception as e:
        logging.exception("Fatal error during application initialization")
        messagebox.showerror("Fatal Error", f"A fatal error occurred:\n{e}")
