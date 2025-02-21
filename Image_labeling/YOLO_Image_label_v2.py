import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import os
import shutil
import yaml
import json
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np

PROJECTS_DIR = os.path.join(os.getcwd(), "projects")
os.makedirs(PROJECTS_DIR, exist_ok=True)

# --- BoundingBoxEditor Class ---
class BoundingBoxEditor:
    def __init__(self, root, project):
        self.root = root
        self.project = project
        self.root.title(f"Bounding Box Editor - Project: {project['project_name']}")
        # Use project settings for dataset folder; label folder is auto-created as a subfolder named "labels".
        self.folder_path = project["dataset_path"]
        self.label_folder = os.path.join(self.folder_path, "labels")
        os.makedirs(self.label_folder, exist_ok=True)

        # YAML file handling (create a default one if missing)
        self.yaml_path = os.path.join(self.folder_path, "dataset.yaml")
        if not os.path.exists(self.yaml_path):
            default_yaml = {
                'names': {0: 'person'},
                'paths': {
                    'dataset': self.folder_path,
                    'train': os.path.join(self.folder_path, "train"),
                    'val': os.path.join(self.folder_path, "val")
                }
            }
            with open(self.yaml_path, 'w') as f:
                yaml.dump(default_yaml, f)
        with open(self.yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        raw_names = data.get('names', {0: 'person'})
        if isinstance(raw_names, dict):
            self.class_names = [raw_names[k] for k in sorted(raw_names.keys())]
        else:
            self.class_names = list(raw_names)
        self.paths = data.get('paths', {'dataset': self.folder_path, 'train': '', 'val': ''})
        self.validation = bool(self.paths.get('val'))
        
        # Initialize unique colors for each class.
        self.update_class_colors()
        # Dictionary to hold image statuses (e.g. "edited", "viewed", "not_viewed")
        self.image_status = {}

        # Create main frames.
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Left frame: Treeview for image list.
        self.image_list_frame = tk.Frame(self.main_frame, width=200)
        self.image_list_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.image_tree = ttk.Treeview(self.image_list_frame, columns=("filename",), show="headings", selectmode="browse")
        self.image_tree.heading("filename", text="Images")
        self.image_tree.column("filename", anchor=tk.W)
        self.image_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_tree.tag_configure("edited", background="lightgreen")
        self.image_tree.tag_configure("viewed", background="lightblue")
        self.image_tree.tag_configure("not_viewed", background="white")
        self.scrollbar = tk.Scrollbar(self.image_list_frame, orient="vertical", command=self.image_tree.yview)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.image_tree.configure(yscrollcommand=self.scrollbar.set)
        self.image_tree.bind("<<TreeviewSelect>>", self.on_image_select)

        # Center frame: Canvas and info panel.
        self.content_frame = tk.Frame(self.main_frame)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas = tk.Canvas(self.content_frame, width=500, height=720)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)
        self.info_frame = tk.Frame(self.content_frame, width=300)
        self.info_frame.pack(side=tk.LEFT, fill=tk.Y)
        self.info_label = tk.Label(self.info_frame, text="Bounding Box Info", font=("Arial", 14))
        self.info_label.pack(pady=10)
        self.image_name_label = tk.Label(self.info_frame, text="", font=("Arial", 12))
        self.image_name_label.pack(pady=5)
        self.bbox_info_frame = tk.Frame(self.info_frame)
        self.bbox_info_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        # Right frame: Class list and related buttons.
        self.class_frame = tk.Frame(self.content_frame, width=200)
        self.class_frame.pack(side=tk.RIGHT, fill=tk.Y)
        self.class_label = tk.Label(self.class_frame, text="Classes", font=("Arial", 14))
        self.class_label.pack(pady=10)
        self.class_listbox = tk.Listbox(self.class_frame)
        self.class_listbox.pack(pady=10, fill=tk.BOTH, expand=True)
        for cls in self.class_names:
            self.class_listbox.insert(tk.END, cls)
        self.class_entry = tk.Entry(self.class_frame)
        self.class_entry.pack(pady=5, fill=tk.X, padx=5)
        
        # Buttons to add, update, and remove classes.
        btn_frame = tk.Frame(self.class_frame)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Add", command=self.add_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Update", command=self.update_class).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Remove", command=self.remove_class).pack(side=tk.LEFT, padx=2)
        self.clear_selection_button = tk.Button(self.class_frame, text="Clear Selection", command=self.clear_class_selection)
        self.clear_selection_button.pack(pady=10)
        self.paste_all_button = tk.Button(self.class_frame, text="Paste All", command=self.paste_all_bboxes)
        self.paste_all_button.pack(pady=10)
        self.delete_image_button = tk.Button(self.class_frame, text="Delete Image", command=self.delete_image)
        self.delete_image_button.pack(pady=10)
        self.copy_frame = tk.Frame(self.class_frame)
        self.copy_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        self.copy_frame.bind("<MouseWheel>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-4>", self.on_mouse_wheel)
        self.copy_frame.bind("<Button-5>", self.on_mouse_wheel)
        self.copied_bbox_list = []
        self.update_copied_bbox_display()

        # Initialize image and bounding box variables.
        self.image = None
        self.image_path = None
        self.bboxes = []
        self.current_bbox = None
        self.rect = None
        self.image_files = []
        self.current_image_index = -1
        self.selected_class_index = None

        # Top-level buttons.
        self.auto_annotate_button = tk.Button(root, text="Auto Annotate", command=self.auto_annotate_dataset)
        self.auto_annotate_button.pack(side=tk.LEFT)
        self.save_button = tk.Button(root, text="Save Labels", command=self.save_labels)
        self.save_button.pack(side=tk.LEFT)
        self.load_model_button = tk.Button(root, text="Load Model", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT)
        # Changed from "Edit YAML" to "Export YAML"
        self.export_yaml_button = tk.Button(root, text="Export YAML", command=self.export_yaml_window)
        self.export_yaml_button.pack(side=tk.LEFT)

        # Bind events for drawing and navigation.
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.root.bind("s", lambda event: self.save_labels())
        self.root.bind("<Escape>", lambda event: self.clear_class_selection())
        self.root.bind("<Down>", self.next_image)
        self.root.bind("<Up>", self.previous_image)
        self.class_listbox.bind("<Down>", lambda e: "break")
        self.class_listbox.bind("<Up>", lambda e: "break")
        self.root.bind("<Key>", self.on_key_press)
        
        # Load dataset images and statuses.
        self.load_dataset()

    def add_class(self):
        new_class = self.class_entry.get().strip()
        if new_class:
            self.class_listbox.insert(tk.END, new_class)
            self.class_names.append(new_class)
            self.update_class_colors()  # Refresh the color mapping, if necessary.
            self.class_entry.delete(0, tk.END)
    
    def update_class(self):
        selection = self.class_listbox.curselection()
        if selection:
            index = selection[0]
            new_val = self.class_entry.get().strip()
            if new_val:
                self.class_listbox.delete(index)
                self.class_listbox.insert(index, new_val)
                self.class_names[index] = new_val
                self.update_class_colors()  # Update the colors since names might change.
                self.class_entry.delete(0, tk.END)
    
    def remove_class(self):
        selection = self.class_listbox.curselection()
        if selection:
            index = selection[0]
            self.class_listbox.delete(index)
            self.class_names.pop(index)
            self.update_class_colors()

    
    def update_class_colors(self):
        predefined_colors = ["red", "green", "blue", "yellow", "cyan", "magenta", "orange", "purple", "brown", "pink"]
        self.class_colors = {i: predefined_colors[i % len(predefined_colors)] for i in range(len(self.class_names))}

    def on_key_press(self, event):
        if event.char.isdigit():
            idx = int(event.char) - 1
            if 0 <= idx < len(self.class_names):
                self.class_listbox.selection_clear(0, tk.END)
                self.class_listbox.selection_set(idx)
                self.selected_class_index = idx

    def load_model(self):
        model_path = filedialog.askopenfilename(title="Select YOLO Model", filetypes=[("PyTorch Model", "*.pt"), ("All Files", "*.*")])
        if model_path:
            try:
                self.model = YOLO(model_path)
                messagebox.showinfo("Success", f"Model loaded successfully from {model_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def update_copied_bbox_display(self):
        for widget in self.copy_frame.winfo_children():
            widget.destroy()
        if not self.copied_bbox_list:
            tk.Label(self.copy_frame, text="Copied Bounding Boxes: None", font=("Arial", 12)).pack(pady=10)
        else:
            for bbox in self.copied_bbox_list:
                x, y, w, h, class_id = bbox
                tk.Label(self.copy_frame,
                         text=f"Class {self.class_names[class_id]}, ({x}, {y}), ({w}, {h})",
                         font=("Arial", 12)).pack(pady=5)

    def update_class_listbox(self):
        self.class_listbox.delete(0, tk.END)
        for cls in self.class_names:
            self.class_listbox.insert(tk.END, cls)
        self.update_class_colors()

    # --- Methods to persist image statuses ---
    def save_statuses(self):
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            with open(status_file, "w") as f:
                json.dump(self.image_status, f)

    def load_statuses(self):
        if self.folder_path:
            status_file = os.path.join(self.folder_path, "image_status.json")
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    self.image_status = json.load(f)
            else:
                self.image_status = {}

    # --- New Export YAML Window ---
    def export_yaml_window(self):
        export_win = tk.Toplevel(self.root)
        export_win.title("Export YAML Settings")
    
        # --- Export Destination Option ---
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
        
        # --- Validation Option ---
        include_val_var = tk.BooleanVar(value=self.validation)
        val_frame = tk.Frame(export_win)
        val_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Checkbutton(
            val_frame,
            text="Include Validation",
            variable=include_val_var
        ).pack(side=tk.LEFT)
        
        # --- Export Button(s) ---
        button_frame = tk.Frame(export_win)
        button_frame.pack(fill=tk.X, padx=10, pady=10)
        
        def export_yaml():
            # Load current YAML content.
            try:
                with open(self.yaml_path, "r") as f:
                    data = yaml.safe_load(f)
            except Exception as e:
                messagebox.showerror("Error", f"Could not load YAML file:\n{e}")
                return
            
            # Update validation setting in YAML.
            if not include_val_var.get():
                if "paths" in data:
                    data["paths"]["val"] = ""
            
            # Determine the base export folder.
            if export_to_current_var.get():
                base_export_folder = self.folder_path
            else:
                base_export_folder = custom_export_entry.get().strip()
                if not base_export_folder:
                    messagebox.showerror("Error", "Please select a custom export location.")
                    return
            
            # Create an "exported" folder in the chosen destination.
            export_folder = os.path.join(base_export_folder, "exported")
            os.makedirs(export_folder, exist_ok=True)
            
            # Create subfolders for train.
            train_folder = os.path.join(export_folder, "train")
            train_images_folder = os.path.join(train_folder, "images")
            train_labels_folder = os.path.join(train_folder, "labels")
            os.makedirs(train_images_folder, exist_ok=True)
            os.makedirs(train_labels_folder, exist_ok=True)
            
            # If validation is enabled, create corresponding subfolders.
            if include_val_var.get():
                val_folder = os.path.join(export_folder, "val")
                val_images_folder = os.path.join(val_folder, "images")
                val_labels_folder = os.path.join(val_folder, "labels")
                os.makedirs(val_images_folder, exist_ok=True)
                os.makedirs(val_labels_folder, exist_ok=True)
            else:
                val_folder = ""
            
            # Update the YAML file paths to point to the new export structure.
            if "paths" in data:
                data["paths"]["train"] = train_folder
                data["paths"]["val"] = val_folder
            
            # Write the updated YAML file into the exported folder.
            export_yaml_path = os.path.join(export_folder, "dataset.yaml")
            try:
                with open(export_yaml_path, "w") as f:
                    yaml.dump(data, f)
            except Exception as e:
                messagebox.showerror("Error", f"Could not export YAML:\n{e}")
                return
            
            # --- Copy Annotated Images Only ---
            # Iterate over all image files and copy those that have a corresponding non-empty label file.
            for image_file in self.image_files:
                label_file = os.path.splitext(image_file)[0] + '.txt'
                src_label_path = os.path.join(self.label_folder, label_file)
                # Check if label file exists and is non-empty.
                if os.path.exists(src_label_path) and os.path.getsize(src_label_path) > 0:
                    src_image_path = os.path.join(self.folder_path, image_file)
                    dst_image_path = os.path.join(train_images_folder, image_file)
                    dst_label_path = os.path.join(train_labels_folder, label_file)
                    try:
                        shutil.copy(src_image_path, dst_image_path)
                        shutil.copy(src_label_path, dst_label_path)
                    except Exception as e:
                        messagebox.showerror("Error", f"Failed to export {image_file}:\n{e}")
            
            messagebox.showinfo("Success", f"Export complete!\nYAML and annotated images exported to:\n{export_folder}")
            export_win.destroy()
        
        tk.Button(button_frame, text="Export", command=export_yaml).pack(side=tk.RIGHT, padx=5)
        tk.Button(button_frame, text="Cancel", command=export_win.destroy).pack(side=tk.RIGHT, padx=5)


    def edit_yaml_window(self):
        # This method is no longer used.
        pass

    def load_dataset(self):
        if not self.folder_path:
            messagebox.showerror("Error", "Dataset folder not set.")
            return
        for item in self.image_tree.get_children():
            self.image_tree.delete(item)
        self.image_files = [f for f in os.listdir(self.folder_path)
                            if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if not self.image_files:
            messagebox.showinfo("No Images", "No images found in the selected folder.")
            return
        self.load_statuses()
        for image_file in self.image_files:
            status = self.image_status.get(image_file, "not_viewed")
            self.image_status[image_file] = status
            self.image_tree.insert("", tk.END, iid=image_file, values=(image_file,), tags=(status,))
        self.save_statuses()

    def auto_annotate_dataset(self):
        if self.model is None:
            messagebox.showerror("Model Not Loaded", "Please load a YOLO model first.")
            return
        if not self.folder_path:
            messagebox.showerror("Dataset Not Loaded", "Please load a dataset first.")
            return
        for image_file in self.image_files:
            image_path = os.path.join(self.folder_path, image_file)
            label_filename = os.path.splitext(image_file)[0] + '.txt'
            label_path = os.path.join(self.label_folder, label_filename)
            self.run_inference_and_save_labels(image_path, label_path)
        messagebox.showinfo("Auto Annotate", "Auto annotation complete for all images.")

    def run_inference_and_save_labels(self, image_path, label_path):
        if self.model is None:
            messagebox.showerror("Model Not Loaded", "Please load a YOLO model first.")
            return
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model(image_path, conf=0.5)
        bboxes = []
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id >= len(self.class_names):
                continue
            x_center, y_center, width, height = box.xywhn[0].cpu().numpy()
            img_height, img_width = image.shape[:2]
            x_center_abs = x_center * img_width
            y_center_abs = y_center * img_height
            width_abs = width * img_width
            height_abs = height * img_height
            x_min = int(x_center_abs - width_abs / 2)
            y_min = int(y_center_abs - height_abs / 2)
            bboxes.append((x_min, y_min, int(width_abs), int(height_abs), class_id))
        with open(label_path, 'w') as label_file:
            for bbox in bboxes:
                x, y, w, h, class_id = bbox
                img_height, img_width = image.shape[:2]
                x_center = (x + w/2) / img_width
                y_center = (y + h/2) / img_height
                width = w / img_width
                height = h / img_height
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

    def on_image_select(self, event):
        selected = self.image_tree.selection()
        if selected:
            image_file = selected[0]
            image_path = os.path.join(self.folder_path, image_file)
            self.load_image(image_path)

    def load_image(self, image_path=None):
        if image_path:
            self.image_path = image_path
            self.current_image_index = self.image_files.index(os.path.basename(image_path))
        else:
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
            if not self.image_path:
                return
            self.current_image_index = self.image_files.index(os.path.basename(self.image_path))
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.image = cv2.resize(self.image, (500, 720))
        self.display_image()
        label_filename = os.path.splitext(os.path.basename(self.image_path))[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_filename)
        self.bboxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as label_file:
                for line in label_file:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    img_height, img_width = self.image.shape[:2]
                    x_center_abs = x_center * img_width
                    y_center_abs = y_center * img_height
                    width_abs = width * img_width
                    height_abs = height * img_height
                    x_min = int(x_center_abs - width_abs / 2)
                    y_min = int(y_center_abs - height_abs / 2)
                    self.bboxes.append((x_min, y_min, int(width_abs), int(height_abs), int(class_id)))
        self.display_bboxes()
        current_item_id = os.path.basename(self.image_path)
        new_status = "edited" if self.bboxes else "viewed"
        self.image_status[current_item_id] = new_status
        self.image_tree.item(current_item_id, tags=(new_status,))
        self.save_statuses()
        self.image_name_label.config(text=current_item_id)
        if self.selected_class_index is not None:
            self.class_listbox.selection_set(self.selected_class_index)

    def display_image(self):
        self.canvas.delete("all")
        self.tk_image = ImageTk.PhotoImage(image=Image.fromarray(self.image))
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_image)

    def display_bboxes(self):
        self.canvas.delete("bbox")
        for widget in self.bbox_info_frame.winfo_children():
            widget.destroy()
        for i, bbox in enumerate(self.bboxes):
            x, y, w, h, class_id = bbox
            color = self.class_colors.get(class_id, "red")
            self.canvas.create_rectangle(x, y, x+w, y+h, outline=color, width=2, tags="bbox")
            self.canvas.create_text(x, y-10, text=self.class_names[class_id], fill=color, anchor=tk.NW, tags="bbox")
            bbox_info = tk.Frame(self.bbox_info_frame)
            bbox_info.pack(fill=tk.X, pady=2)
            tk.Label(bbox_info, text=f"Class: {self.class_names[class_id]}, Position: ({x}, {y}), Size: ({w}, {h})").pack(side=tk.LEFT)
            tk.Button(bbox_info, text="Copy", command=lambda bbox=bbox: self.copy_bbox(bbox)).pack(side=tk.RIGHT)
            tk.Button(bbox_info, text="Delete", command=lambda i=i: self.delete_bbox(i)).pack(side=tk.RIGHT)

    def copy_bbox(self, bbox):
        self.copied_bbox_list.append(bbox)
        self.update_copied_bbox_display()

    def paste_all_bboxes(self):
        if self.copied_bbox_list:
            for bbox in self.copied_bbox_list:
                self.bboxes.append(bbox)
            self.display_bboxes()
        else:
            messagebox.showinfo("Info", "No bounding boxes copied to paste.")

    def on_click(self, event):
        self.current_bbox = [event.x, event.y, event.x, event.y]
        self.rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="blue", width=2, tags="bbox")

    def on_drag(self, event):
        if self.current_bbox is not None:
            self.current_bbox[2] = event.x
            self.current_bbox[3] = event.y
            self.canvas.coords(self.rect, *self.current_bbox)

    def on_release(self, event):
        if self.current_bbox is not None:
            x1, y1, x2, y2 = self.current_bbox
            x_min, y_min = min(x1, x2), min(y1, y2)
            width, height = abs(x2-x1), abs(y2-y1)
            selected_class_index = self.class_listbox.curselection()
            class_id = selected_class_index[0] if selected_class_index else 0
            self.bboxes.append((x_min, y_min, width, height, class_id))
            self.current_bbox = None
            self.rect = None
            self.display_bboxes()

    def delete_bbox(self, index):
        del self.bboxes[index]
        self.display_bboxes()

    def save_labels(self):
        if not self.image_path:
            return
        label_filename = os.path.splitext(os.path.basename(self.image_path))[0] + '.txt'
        label_path = os.path.join(self.label_folder, label_filename)
        with open(label_path, 'w') as label_file:
            for bbox in self.bboxes:
                x, y, w, h, class_id = bbox
                img_height, img_width = self.image.shape[:2]
                x_center = (x+w/2) / img_width
                y_center = (y+h/2) / img_height
                width = w / img_width
                height = h / img_height
                label_file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
        print(f"Saved labels for {self.image_path}")
        if self.paths.get('train'):
            train_images_folder = os.path.join(self.paths['train'], "images")
            train_labels_folder = os.path.join(self.paths['train'], "labels")
            os.makedirs(train_images_folder, exist_ok=True)
            os.makedirs(train_labels_folder, exist_ok=True)
            try:
                shutil.copy(self.image_path, os.path.join(train_images_folder, os.path.basename(self.image_path)))
                shutil.copy(label_path, os.path.join(train_labels_folder, os.path.basename(label_path)))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to train folder:\n{e}")
        if self.paths.get('val'):
            val_images_folder = os.path.join(self.paths['val'], "images")
            val_labels_folder = os.path.join(self.paths['val'], "labels")
            os.makedirs(val_images_folder, exist_ok=True)
            os.makedirs(val_labels_folder, exist_ok=True)
            try:
                shutil.copy(self.image_path, os.path.join(val_images_folder, os.path.basename(self.image_path)))
                shutil.copy(label_path, os.path.join(val_labels_folder, os.path.basename(label_path)))
            except Exception as e:
                messagebox.showerror("Error", f"Failed to copy to validation folder:\n{e}")
        current_item_id = os.path.basename(self.image_path)
        new_status = "edited" if self.bboxes else "viewed"
        self.image_status[current_item_id] = new_status
        self.image_tree.item(current_item_id, tags=(new_status,))
        self.save_statuses()

    def next_image(self, event):
        self.root.focus_set()
        if self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))

    def previous_image(self, event):
        self.root.focus_set()
        if self.current_image_index > 0:
            self.current_image_index -= 1
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))

    def clear_class_selection(self):
        self.class_listbox.selection_clear(0, tk.END)
        self.selected_class_index = None
        self.copied_bbox_list = []
        self.update_copied_bbox_display()
        self.root.focus_set()

    def delete_image(self):
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
        del self.image_files[self.current_image_index]
        self.image_tree.delete(current_image_filename)
        self.canvas.delete("all")
        self.image_name_label.config(text="")
        self.bbox_info_frame.destroy()
        self.bbox_info_frame = tk.Frame(self.info_frame)
        self.bbox_info_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        if self.image_files:
            self.current_image_index = min(self.current_image_index, len(self.image_files)-1)
            self.load_image(os.path.join(self.folder_path, self.image_files[self.current_image_index]))
        else:
            self.current_image_index = -1

    def on_mouse_wheel(self, event):
        if event.delta:
            if event.delta > 0:
                self.previous_image(event)
            else:
                self.next_image(event)
        elif event.num == 5:
            self.next_image(event)
        elif event.num == 4:
            self.previous_image(event)



class ProjectManager:
    def __init__(self, root):
        self.root = root
        self.root.title("Project Manager")
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        tk.Label(self.main_frame, text="Select an option:", font=("Arial", 14)).pack(pady=10)
        tk.Button(self.main_frame, text="New Project", command=self.new_project).pack(pady=5)
        tk.Button(self.main_frame, text="Open Project", command=self.open_project).pack(pady=5)
        tk.Button(self.main_frame, text="Quit", command=root.quit).pack(pady=5)

    def new_project(self):
        new_win = tk.Toplevel(self.root)
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
        # Automatically set label path as a "labels" subfolder of the dataset.
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
            # Save the project JSON in the PROJECTS_DIR.
            project_file = os.path.join(PROJECTS_DIR, f"{project_name}.json")
            with open(project_file, "w") as f:
                json.dump(project, f)
            messagebox.showinfo("Project Created", f"Project '{project_name}' created successfully.")
            new_win.destroy()
            self.root.destroy()  # Close the project manager window.
            self.open_editor(project)
        tk.Button(new_win, text="Create Project", command=create_project).grid(row=2, column=1, pady=10)

    def open_project(self):
        # List all JSON files in the projects folder.
        project_files = [f for f in os.listdir(PROJECTS_DIR) if f.endswith(".json")]
        if not project_files:
            messagebox.showinfo("No Projects", "No project files found in the projects folder.")
            return
        open_win = tk.Toplevel(self.root)
        open_win.title("Select a Project")
        tk.Label(open_win, text="Select a Project:", font=("Arial", 14)).pack(pady=10)
        listbox = tk.Listbox(open_win, width=50, height=10)
        listbox.pack(padx=10, pady=10)
        for f in project_files:
            listbox.insert(tk.END, f)
        def load_selected_project():
            selection = listbox.curselection()
            if selection:
                project_file = listbox.get(selection[0])
                full_path = os.path.join(PROJECTS_DIR, project_file)
                with open(full_path, "r") as f:
                    project = json.load(f)
                open_win.destroy()
                self.root.destroy()  # Close the project manager window.
                self.open_editor(project)
        tk.Button(open_win, text="Open", command=load_selected_project).pack(pady=5)
    
    def open_editor(self, project):
        # This method should create and launch your BoundingBoxEditor.
        # For example:
        editor_root = tk.Tk()
        editor = BoundingBoxEditor(editor_root, project)
        editor_root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    pm = ProjectManager(root)
    root.mainloop()
