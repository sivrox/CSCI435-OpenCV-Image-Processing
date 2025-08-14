import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import os
dataset = "image_dataset"

#GUI Application
class VisualToolApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CSCI435 OpenCV Project")
        self.root.geometry("1080x1920")
        self.root.configure(bg="#CFCFCF")
        self.image_path = None
        self.home_screen()

    def clear_window(self):
        for widget in self.root.winfo_children():
            widget.destroy()

    def home_screen(self):
        self.clear_window()
        self.root.configure(bg="#e6f2ff")

        title = tk.Label(self.root, text="CSCI435: Visual Landmark & Scene Analysis Application", font=("League Spartan", 24, "bold"), foreground="#003366", background="#e6f2ff")
        title.pack(pady=100)

        label = tk.Label(self.root, text="Select an Image", font=("Papyrus", 24, "bold"), foreground="#004d66", background="#e6f2ff")
        label.pack(pady=10, anchor="center")

        upload_btn = tk.Button(self.root, text="Upload Image", font=("Century Gothic", 10, "bold"), foreground="#e6f2ff", width=20, height=2, command=self.upload_image, border=3, background="#003366")
        upload_btn.pack(pady=10, anchor="center")

        select_btn = tk.Button(self.root, text="Choose from Dataset", font=("Century Gothic", 10, "bold"), foreground="#e6f2ff", width=20, height=2, command=self.select_from_dataset, border=3, background="#003366")
        select_btn.pack(pady=10, anchor="center")

        sub_title = tk.Label(self.root, text="Group Members:\n\nSuwathi Rajasekhar (8393606) | "
                             "Adalia Alex (8373759) | "
                             "Ashfina Abbas (8420427) | "
                             "Sahil Rao (7929298) | "
                             "Sivajith Ajith Kumar (8404744)", font=("Century Gothic", 11, "bold"), foreground="#003366", background="#e6f2ff")
        sub_title.pack(pady=80, anchor="center")

    def feature_screen(self):
        self.clear_window()

        title = tk.Label(self.root, text="CSCI435: Visual Landmark & Scene Analysis Application", font=("League Spartan", 24, "bold"), foreground="#003366", background="#e6f2ff")
        title.pack(pady=100)

        label = tk.Label(self.root, text="Choose a Feature", font=("Papyrus", 24, "bold"), foreground="#003366", background="#e6f2ff")
        label.pack(pady=2)

        features = [
            ("Face & Landmark Detection", self.run_face_landmark),
            ("Image Time Classification", self.run_day_night_class),
            ("Image Quality Assessment", self.run_image_assessment),
            ("Find Similar Images", self.run_similarity),
            ("Annotate your Image", self.run_annotation)
        ]

        for text, command in features:
            btn = tk.Button(self.root, text=text, font=("Century Gothic", 10, "bold"), width=30, height=2, command=command, border=3, foreground="#e6f2ff", background="#003366")
            btn.pack(pady=8, anchor="center")
        
        back_btn = tk.Button(self.root, text="Back", font=("Century Gothic", 10, "bold"), width=10, height=1, command=self.home_screen, foreground="#e6f2ff", background="#BC2020")
        back_btn.pack(pady=10, anchor="center")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
        if file_path:
            self.image_path = file_path
            self.feature_screen()

    def select_from_dataset(self):
        files = os.listdir(dataset)
        if not files:
            messagebox.showwarning("Dataset Empty", "No images found in dataset folder.")
            return

        #popup window with images from dataset
        win = Toplevel(self.root)
        win.title("Select Image from Dataset")
        row, col = 0, 0

        for filename in files:
            img_path = os.path.join(dataset, filename)
            img = Image.open(img_path)
            img.thumbnail((200, 200))
            img_tk = ImageTk.PhotoImage(img)

            def make_callback(path=img_path, name=filename):
                return lambda: self.set_image_and_proceed(path, name, win)

            btn = tk.Button(win, image=img_tk, command=make_callback())
            btn.image = img_tk
            btn.grid(row=row, column=col, padx=6, pady=6)
            col += 1
            if col > 5:
                col = 0
                row += 1

    def set_image_and_proceed(self, path, name, window):
        self.image_path = path
        window.destroy()
        self.feature_screen()

    def run_face_landmark(self):
        from features_opencv import feature1_landmark_face
        if self.image_path:
            feature1_landmark_face(self.image_path)

    def run_similarity(self):
        from features_opencv import feature5_find_similarity
        if self.image_path:
            feature5_find_similarity(self.image_path)

    def run_annotation(self):
        from features_opencv import feature6_annotate_img
        self.clear_window()
        title = tk.Label(self.root, text="CSCI435: Visual Landmark & Scene Analysis Application", font=("League Spartan", 24, "bold"), foreground="#003366", background="#e6f2ff")
        title.pack(pady=100)

        tk.Label(self.root, text="Annotate your Image", font=("Papyrus", 24, "bold"), foreground="#003366", background="#e6f2ff").pack(pady=10)

        instruction = (
            "Keyboard Commands:\n\n"
            "r - Draw Rectangle\n"
            "c - Draw Circle\n"
            "t - Add Text\n"
            "s - Save Annotated Image\n"
            "  ESC - Exit Annotation"
        )

        tk.Label(self.root, text=instruction, justify="center", font=("Century Gothic", 11, "bold"), foreground="#003366", background="#e6f2ff").pack(pady=5)

        #button to start annotating
        launch_btn = tk.Button(self.root, text="Start Drawing", font=("Century Gothic", 10, "bold"), command=lambda: feature6_annotate_img(self.image_path), foreground="#e6f2ff", background="#003366", width=20, height=2)
        launch_btn.pack(pady=10, anchor="center")

        #back to features button
        back_btn = tk.Button(self.root, text="Back", font=("Century Gothic", 10, "bold"), command=self.feature_screen, width=10, height=1, foreground="#e6f2ff", background="#BC2020")
        back_btn.pack(pady=5, anchor="center")

    def run_day_night_class(self):
        from features_opencv import feature2_classify_day_night
        if self.image_path:
            feature2_classify_day_night(self.image_path)

    def run_image_assessment(self):
        from features_opencv import feature3_quality_assessment
        if self.image_path:
            feature3_quality_assessment(self.image_path)

# === Run App ===
if __name__ == "__main__":
    root = tk.Tk()
    app = VisualToolApp(root)
    root.mainloop()
