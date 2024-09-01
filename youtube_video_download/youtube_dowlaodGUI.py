import os
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import sys

class YouTubeDownloader:
    def __init__(self, root):
        self.root = root
        self.root.title("YouTube Video Downloader")
        self.stop_download = False

        # URL input
        tk.Label(root, text="YouTube Video URL:").grid(row=0, column=0, padx=10, pady=10)
        self.url_entry = tk.Entry(root, width=50)
        self.url_entry.grid(row=0, column=1, padx=10, pady=10)

        # Save path input
        tk.Label(root, text="Save Path:").grid(row=1, column=0, padx=10, pady=10)
        self.save_path_entry = tk.Entry(root, width=50)
        self.save_path_entry.grid(row=1, column=1, padx=10, pady=10)
        tk.Button(root, text="Browse", command=self.browse_directory).grid(row=1, column=2, padx=10, pady=10)

        # Status label
        self.status_label = tk.Label(root, text="Status: Waiting to start...")
        self.status_label.grid(row=2, column=0, columnspan=3, padx=10, pady=10)

        # Progress bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        # Download and Stop buttons
        tk.Button(root, text="Download", command=self.start_download).grid(row=4, column=1, pady=10)
        tk.Button(root, text="Stop", command=self.stop_download_thread).grid(row=4, column=2, pady=10)

    def browse_directory(self):
        folder_selected = filedialog.askdirectory()
        if folder_selected:
            self.save_path_entry.delete(0, tk.END)
            self.save_path_entry.insert(0, folder_selected)

    def start_download(self):
        video_url = self.url_entry.get()
        save_path = self.save_path_entry.get()
        if not video_url or not save_path:
            messagebox.showwarning("Input Error", "Please provide both the video URL and save path.")
            return
        self.stop_download = False
        self.progress["value"] = 0
        self.status_label.config(text="Status: Downloading...")
        self.download_thread = threading.Thread(target=self.download_youtube_video, args=(video_url, save_path))
        self.download_thread.start()

    def stop_download_thread(self):
        self.stop_download = True
        if self.download_thread and self.download_thread.is_alive():
            messagebox.showinfo("Info", "Stopping the download...")

    def download_youtube_video(self, url, save_path):
        try:
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            
            # Path to yt-dlp binary
            ytdlp_path = os.path.join(os.path.dirname(__file__), 'yt-dlp.exe')
            
            command = [
                ytdlp_path,
                '-o', f'{save_path}/%(title)s.%(ext)s',
                '-f', 'best',
                url
            ]

            # Suppress the console window
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, creationflags=creationflags)

            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    self.parse_progress(output.strip())

            process.wait()
            if process.returncode == 0:
                self.status_label.config(text="Status: Download complete")
                messagebox.showinfo("Success", f"Video downloaded successfully and saved to {save_path}")
                if messagebox.askyesno("Open Folder", "Do you want to open the folder?"):
                    os.startfile(save_path)
            else:
                self.status_label.config(text="Status: Error during download")
                messagebox.showerror("Error", "An error occurred during the download.")

        except Exception as e:
            self.status_label.config(text="Status: Error occurred")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def parse_progress(self, output):
        # Parse yt-dlp's output for progress
        if '%' in output:
            try:
                percent = float(output.split('%')[0].split()[-1])
                self.progress["value"] = percent
                self.root.update_idletasks()
            except ValueError:
                pass

if __name__ == "__main__":
    root = tk.Tk()
    app = YouTubeDownloader(root)
    root.mainloop()