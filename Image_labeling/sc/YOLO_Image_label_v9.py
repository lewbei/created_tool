import os
import shutil
import json
import yaml
import threading

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk

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

def write_bboxes_to_file(label_path, bboxes, image_shape):
    """
    Writes bounding boxes in YOLO format to a label file.

    :param label_path: Path to the .txt label file.
    :param bboxes: List of bounding boxes [ (x, y, w, h, class_id), ... ] in pixel coords.
    :param image_shape: (height, width) of the image used for normalization.
    """
    img_h, img_w = image_shape[:2]
    with open(label_path, 'w') as label_file:
        for x, y, w, h, class_id in bboxes:
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width_norm = w / img_w
            height_norm = h / img_h
            label_file.write(f"{class_id} {x_center} {y_center} {width_norm} {height_norm}\n")

def read_bboxes_from_file(label_path, image_shape):
    """
    Reads YOLO-format bounding boxes from a label file and converts them to pixel coords.

    :param label_path: Path to the .txt label file.
    :param image_shape: (height, width) of the image used for denormalization.
    :return: List of bounding boxes [ (x, y, w, h, class_id), ... ] in pixel coords.
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes

    img_h, img_w = image_shape[:2]
    with open(label_path, 'r') as label_file:
        for line in label_file:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            x_center_abs = x_center * img_w
            y_center_abs = y_center * img_h
            width_abs = width * img_w
            height_abs = height * img_h
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            bboxes.append((x_min, y_min, int(width_abs), int(height_abs), int(class_id)))
    return bboxes

def copy_files(file_list, images_src_dir, images_dst, labels_src_dir, labels_dst):
    """
    Copies images and their corresponding label files to specified destination folders.
    """
    for image_file in file_list:
        label_file = os.path.splitext(image_file)[0] + '.txt'
        src_image_path = os.path.join(images_src_dir, image_file)
        src_label_path = os.path.join(labels_src_dir, label_file)
        try:
            shutil.copy(src_image_path, os.path.join(images_dst, image_file))
            shutil.copy(src_label_path, os.path.join(labels_dst, label_file))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export {image_file}:\n{e}")

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

        # Color mapping for classes
        self.update_class_colors()

        # Dictionary to hold status of each image
        self.image_status = {}

        # -----------------------------
        # Main UI Layout
        # -----------------------------
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left Pane: Treeview for image list
        self.setup_image_list_panel()

        # Middle Pane: Canvas (image) + Info Panel
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.setup_canvas()
        self.setup_info_panel()

        # Right Pane: Class List + Actions
        self.class_frame = tk.Frame(self.content_frame, width=200)
        self.class_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.setup_class_panel()

        # Top Bar: Buttons (Auto Annotate, Save, Load Model, Export)
        self.setup_top_bar()

        # Status Bar: labeling progress counters
        self.setup_status_bar()

        # Initialize image-related variables
        self.image = None
        self.image_path = None
        self.bboxes = []
        self.current_bbox = None
        self.rect = None
        self.image_files = []
        self.current_image_index = -1
        self.selected_class_index = None

        # Load dataset images + statuses
        self.load_dataset()

        # Additional Key Bindings
        self.setup_bindings()

    # --------------------------------------------------
    # Setup / Layout Methods
    # --------------------------------------------------

    def setup_image_list_panel(self):
        """
        Creates the left-side panel containing the Treeview of all images.
        """
        self.image_list_frame = tk.Frame(self.main_frame, width=200)
        self.image_list_frame.pack(side=tk.LEFT, fill=tk.Y)

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
        # Mouse wheel for switching images
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)  # Linux scroll up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)  # Linux scroll down

        # Drawing bounding boxes
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def setup_info_panel(self):
        """
        Creates the right-side panel (inside the middle frame) to show bounding box info.
        """
        self.info_frame = tk.Frame(self.content_frame, width=300)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)

        self.info_label = tk.Label(self.info_frame, text="Bounding Box Info", font=("Arial", 14))
        self.info_label.pack(pady=10)

        self.image_name_label = tk.Label(self.info_frame, text="", font=("Arial", 12))
        self.image_name_label.pack(pady=5)

        self.bbox_info_frame = tk.Frame(self.info_frame)
        self.bbox_info_frame.pack(pady=10, fill=tk.BOTH, expand=True)

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

        self.auto_annotate_button = tk.Button(buttons_frame, text="Auto Annotate", command=self.auto_annotate_dataset_threaded)
        self.auto_annotate_button.pack(side=tk.LEFT)

        self.save_button = tk.Button(buttons_frame, text="Save Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT)

        self.load_model_button = tk.Button(buttons_frame, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT)

        self.export_yaml_button = tk.Button(buttons_frame, text="Export YAML", command=self.export_yaml_window)
        self.export_yaml_button.pack(side=tk.LEFT)

    def setup_status_bar(self):
        """
        Creates a simple status bar showing the counts of images in each status category.
        """
        top_frame = self.root.nametowidget(self.root.winfo_children()[0])  # The top-level frame we already used
        status_frame = tk.Frame(top_frame)
        status_frame.pack(side=tk.RIGHT, padx=10)

        self.status_labels = {}
        statuses = [
            ("Viewed", "viewed"),
            ("Labeled", "edited"),
            ("Review Needed", "review_needed"),
            ("Non-viewed", "not_viewed")
        ]
        for display_name, tag in statuses:
            frame = tk.Frame(status_frame)
            frame.pack(side=tk.LEFT, padx=5)
            label = tk.Label(frame, text=f"{display_name}: 0")
            label.pack()
            self.status_labels[display_name] = label

    def setup_bindings(self):
        """
        Binds additional keyboard shortcuts for navigation and actions.
        """
        # Ctrl+S to save
        self.root.bind("<Control-s>", lambda event: self.save_labels())

        # ESC to clear selection
        self.root.bind("<Escape>", lambda event: self.clear_class_selection())

        # Down/Up arrow to navigate images
        self.root.bind("<Down>", lambda event: self.navigate_image(+1))
        self.root.bind("<Up>", lambda event: self.navigate_image(-1))

        # Prevent Listbox from also capturing up/down
        self.class_listbox.bind("<Down>", lambda e: "break")
        self.class_listbox.bind("<Up>", lambda e: "break")

        # Digit-based quick class selection
        self.root.bind("<Key>", self.on_key_press)

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
                "names": ["person"]
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

        # Gather image files from dataset folder
        self.image_files = [
            f for f in os.listdir(self.folder_path)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]
        if not self.image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return

        # Load statuses from JSON if available
        self.load_statuses()

        # Insert items into Treeview
        for image_file in self.image_files:
            status = self.image_status.get(image_file, "not_viewed")
            self.image_tree.insert("", tk.END, iid=image_file, values=(image_file,), tags=(status,))

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
        for image_file in self.image_files:
            status = self.image_status.get(image_file, "not_viewed")
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
            self.display_bboxes()

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
    # Image Navigation / Display
    # --------------------------------------------------

    def on_image_select(self, event):
        """
        Handler when user selects a new image from the Treeview.
        """
        selected = self.image_tree.selection()
        if selected:
            image_file = selected[0]
            image_path = os.path.join(self.folder_path, image_file)
            self.load_image(image_path)

    def load_image(self, image_path=None):
        """
        Loads an image from 'image_path' into the canvas, reads associated bboxes, 
        and updates the internal state (self.current_image_index, etc.).
        """
        if image_path:
            self.image_path = image_path
            self.current_image_index = self.image_files.index(os.path.basename(image_path))
        else:
            # Fallback: user manually picks a file
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not self.image_path:
                return
            self.current_image_index = self.image_files.index(os.path.basename(self.image_path))

        # Read and resize the image to fit the canvas
        # WARNING: This can distort aspect ratio. Consider preserving ratio if you prefer.
        original_image = cv2.imread(self.image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        # (Optional) store original shape for more accurate bounding boxes
        # But for now, we fix the display size at 500 x 720
        self.image = cv2.resize(original_image, (500, 720))

        # Draw the image in the canvas
        self.display_image()

        # Load bounding boxes from label file
        label_filename = os.path.splitext(os.path.basename(self.image_path))[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_filename)
        self.bboxes = read_bboxes_from_file(label_path, self.image.shape)

        # Show bounding boxes
        self.display_bboxes()

        # Update status if we have bboxes or not
        current_item_id = os.path.basename(self.image_path)
        new_status = "edited" if self.bboxes else "viewed"
        self.image_status[current_item_id] = new_status
        self.image_tree.item(current_item_id, tags=(new_status,))
        self.save_statuses()
        self.update_status_labels()

        # Update the info panel
        self.image_name_label.config(text=current_item_id)

        # Maintain class selection
        if self.selected_class_index is not None:
            self.class_listbox.selection_set(self.selected_class_index)

    def display_image(self):
        """
        Clears the canvas and displays the current self.image (already resized).
        """
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def navigate_image(self, direction):
        """
        Moves to the next or previous image based on 'direction' (+1 or -1).
        """
        self.root.focus_set()
        new_index = self.current_image_index + direction
        if 0 <= new_index < len(self.image_files):
            self.current_image_index = new_index
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
        Start a new bounding box from the click point.
        """
        if self.image is None:
            messagebox.showwarning("No Image", "Please select or load an image first.")
            return
        self.current_bbox = [event.x, event.y, event.x, event.y]
        self.rect = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline="blue", width=2, tags="bbox"
        )

    def on_drag(self, event):
        """
        Update the rectangle shape as the user drags the mouse.
        """
        if self.current_bbox is not None:
            self.current_bbox[2] = event.x
            self.current_bbox[3] = event.y
            self.canvas.coords(self.rect, *self.current_bbox)

    def on_release(self, event):
        """
        Finalize the bounding box, store it with the currently selected class.
        """
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

        self.display_bboxes()

    def display_bboxes(self):
        """
        Clears and redraws all bounding boxes in self.bboxes on the canvas,
        and updates the info frame to show each bbox details (with copy/delete).
        """
        self.canvas.delete("bbox")
        for widget in self.bbox_info_frame.winfo_children():
            widget.destroy()

        for i, (x, y, w, h, class_id) in enumerate(self.bboxes):
            color = self.class_colors.get(class_id, "red")
            self.canvas.create_rectangle(
                x, y, x + w, y + h,
                outline=color, width=2, tags="bbox"
            )
            self.canvas.create_text(
                x, y - 10,
                text=self.class_names[class_id],
                fill=color, anchor=tk.NW, tags="bbox"
            )

            # Build a small row in the bbox_info_frame for each bounding box
            bbox_info = tk.Frame(self.bbox_info_frame)
            bbox_info.pack(fill=tk.X, pady=2)

            tk.Label(
                bbox_info,
                text=f"Class: {self.class_names[class_id]}, Pos:({x},{y}), Size:({w},{h})"
            ).pack(side=tk.LEFT)

            tk.Button(
                bbox_info,
                text="Copy",
                command=lambda bbox=(x, y, w, h, class_id): self.copy_bbox(bbox)
            ).pack(side=tk.RIGHT)
            tk.Button(
                bbox_info,
                text="Delete",
                command=lambda i=i: self.delete_bbox(i)
            ).pack(side=tk.RIGHT)

    def delete_bbox(self, index):
        """
        Deletes the bounding box at 'index'.
        """
        if 0 <= index < len(self.bboxes):
            del self.bboxes[index]
            self.display_bboxes()

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
            self.display_bboxes()
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
        if not self.image_path:
            return

        label_filename = os.path.splitext(os.path.basename(self.image_path))[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_filename)
        write_bboxes_to_file(label_path, self.bboxes, self.image.shape)

        print(f"Saved labels for {self.image_path}")

        current_item_id = os.path.basename(self.image_path)
        new_status = "edited" if self.bboxes else "viewed"
        self.image_status[current_item_id] = new_status
        self.image_tree.item(current_item_id, tags=(new_status,))
        self.save_statuses()
        self.update_status_labels()

    def delete_image(self):
        """
        Deletes the current image file and its label file from disk, then updates the UI.
        """
        if self.current_image_index == -1:
            messagebox.showwarning("Warning", "No image selected to delete.")
            return

        current_image_filename = self.image_files[self.current_image_index]
        image_path = os.path.join(self.folder_path, current_image_filename)
        label_filename = os.path.splitext(current_image_filename)[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_filename)

        if not messagebox.askyesno("Confirm Delete", f"Delete {current_image_filename} and its label?"):
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
        self.image_tree.delete(current_image_filename)

        self.canvas.delete("all")
        self.image_name_label.config(text="")
        self.bbox_info_frame.destroy()
        self.bbox_info_frame = tk.Frame(self.info_frame)
        self.bbox_info_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        if self.image_files:
            self.current_image_index = min(self.current_image_index, len(self.image_files) - 1)
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
        else:
            self.current_image_index = -1

        self.update_status_labels()

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
                if not detections:
                    self.image_status[image_file] = "viewed"
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
                            flagged_images.append(image_file)
                            self.image_status[image_file] = "review_needed"
                        else:
                            # Write bounding boxes to file
                            with open(label_path, 'w') as lf:
                                for (x, y, w, h, cid, _) in bboxes:
                                    x_center_norm = (x + w / 2) / img_w
                                    y_center_norm = (y + h / 2) / img_h
                                    w_norm = w / img_w
                                    h_norm = h / img_h
                                    lf.write(f"{cid} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")
                            self.image_status[image_file] = "edited"
                    else:
                        # If YOLO found detections but we skipped them all
                        self.image_status[image_file] = "viewed"

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
            for image_file in self.image_files:
                self.image_tree.item(image_file, tags=(self.image_status.get(image_file, "not_viewed"),))

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
                # Identify images that have labeled bounding boxes
                annotated = []
                for image_file in self.image_files:
                    label_file = os.path.splitext(image_file)[0] + '.txt'
                    src_label_path = os.path.join(self.label_folder, label_file)
                    if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
                        annotated.append(image_file)

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
                all_labeled = [
                    f for f in self.image_files
                    if os.path.exists(os.path.join(self.label_folder, os.path.splitext(f)[0] + '.txt'))
                    and os.path.getsize(os.path.join(self.label_folder, os.path.splitext(f)[0] + '.txt')) > 0
                ]
                copy_files(all_labeled, self.folder_path,
                           os.path.join(train_folder, "images"),
                           self.label_folder,
                           os.path.join(train_folder, "labels"))
            else:
                copy_files(train_files, self.folder_path,
                           os.path.join(train_folder, "images"),
                           self.label_folder,
                           os.path.join(train_folder, "labels"))
                copy_files(val_files, self.folder_path,
                           os.path.join(val_folder, "images"),
                           self.label_folder,
                           os.path.join(val_folder, "labels"))
                if test_files:
                    test_folder = os.path.join(export_folder, "test")
                    copy_files(test_files, self.folder_path,
                               os.path.join(test_folder, "images"),
                               self.label_folder,
                               os.path.join(test_folder, "labels"))

            messagebox.showinfo(
                "Success",
                f"Export complete!\nYAML and annotated images exported to:\n{export_folder}"
            )
            export_win.destroy()

        tk.Button(button_frame, text="Export", command=export_yaml).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, padx=5)

# --------------------------------------------------
# ProjectManager Class
# --------------------------------------------------

class ProjectManager:
    """
    A simple Tk-based manager for creating/opening projects. Each project is saved
    as a JSON file in 'PROJECTS_DIR' with details like project name and dataset path.
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Project Manager")
        center_window(self.root, 300, 200)

        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        tk.Label(self.main_frame, text="Select an option:", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.main_frame, text="New Project", command=self.new_project).pack(pady=5)
        tk.Button(self.main_frame, text="Open Project", command=self.open_project).pack(pady=5)
        tk.Button(self.main_frame, text="Quit", command=root.quit).pack(pady=5)

    def new_project(self):
        """
        Opens a dialog to create a new project: user specifies project name + dataset path.
        """
        new_win = tk.Toplevel(self.root)
        new_win.transient(self.root)
        new_win.grab_set()
        new_win.title("New Project")

        tk.Label(new_win, text="Project Name:").grid(row=0, column=0, padx=5, pady=5)
        name_entry = tk.Entry(new_win)
        name_entry.grid(row=0, column=1, padx=5, pady=5)

        tk.Label(new_win, text="Dataset Path:").grid(row=1, column=0, padx=5, pady=5)
        dataset_entry = tk.Entry(new_win, width=50)
        dataset_entry.grid(row=1, column=1, padx=5, pady=5)

        def browse_dataset():
            folder = filedialog.askdirectory(title="Select Dataset Folder")
            if folder:
                dataset_entry.delete(0, tk.END)
                dataset_entry.insert(0, folder)

        tk.Button(new_win, text="Browse", command=browse_dataset).grid(row=1, column=2, padx=5, pady=5)

        def create_project():
            project_name = name_entry.get().strip()
            dataset_path = dataset_entry.get().strip()
            if not project_name or not dataset_path:
                messagebox.showerror("Error", "Project name and dataset path are required.")
                return

            project = {
                "project_name": project_name,
                "dataset_path": dataset_path,
                "label_path": os.path.join(dataset_path, "labels")
            }
            project_file = os.path.join(PROJECTS_DIR, f"{project_name}.json")
            with open(project_file, "w") as f:
                json.dump(project, f)

            messagebox.showinfo("Project Created", f"Project '{project_name}' created successfully.")
            new_win.destroy()
            self.root.destroy()
            self.open_editor(project)

        tk.Button(new_win, text="Create Project", command=create_project).grid(row=2, column=1, pady=10)

        # Center this pop-up window
        new_win.update_idletasks()
        req_width = new_win.winfo_reqwidth()
        req_height = new_win.winfo_reqheight()
        self.root.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        pos_x = parent_x + (parent_width - req_width) // 2
        pos_y = parent_y + (parent_height - req_height) // 2
        new_win.geometry(f"{req_width}x{req_height}+{pos_x}+{pos_y}")

    def open_project(self):
        """
        Lists all available project .json files in PROJECTS_DIR in a Listbox,
        and lets the user select one to open.
        """
        project_files = [f for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]
        if not project_files:
            messagebox.showinfo("No Projects", "No project files found in the projects folder.")
            return

        open_win = tk.Toplevel(self.root)
        open_win.transient(self.root)
        open_win.grab_set()
        open_win.title("Select a Project")

        tk.Label(open_win, text="Select a Project:", font=("Arial", 14)).pack(pady=10)
        listbox = tk.Listbox(open_win, width=50, height=10)
        listbox.pack(padx=10, pady=10)

        for f in project_files:
            listbox.insert(tk.END, f)

        tk.Button(
            open_win,
            text="Open",
            command=lambda: self.load_selected_project(listbox, open_win)
        ).pack(pady=5)

        # Center the pop-up
        open_win.update_idletasks()
        req_width = open_win.winfo_reqwidth()
        req_height = open_win.winfo_reqheight()
        self.root.update_idletasks()
        parent_x = self.root.winfo_x()
        parent_y = self.root.winfo_y()
        parent_width = self.root.winfo_width()
        parent_height = self.root.winfo_height()
        pos_x = parent_x + (parent_width - req_width) // 2
        pos_y = parent_y + (parent_height - req_height) // 2
        open_win.geometry(f"{req_width}x{req_height}+{pos_x}+{pos_y}")

    def load_selected_project(self, listbox, open_win):
        """
        Loads the project file selected from the listbox and opens the bounding box editor.
        """
        selection = listbox.curselection()
        if selection:
            project_file = listbox.get(selection[0])
            full_path = os.path.join(PROJECTS_DIR, project_file)
            with open(full_path, "r") as f:
                project = json.load(f)

            open_win.destroy()
            self.root.destroy()
            self.open_editor(project)

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
    pm = ProjectManager(root)
    root.mainloop()
