import customtkinter as ctk
from PIL import Image
import os
import cv2

LOG_DIR = "unknown_logs"
LOG_FILE = os.path.join(LOG_DIR, "unknown_log.txt")

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class UnknownViewer(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Unknown Detections")
        self.geometry("950x600")
        self.resizable(False, False)

        title = ctk.CTkLabel(self, text="Unknown Faces Log", font=("Arial", 22))
        title.pack(pady=15)

        self.scroll_frame = ctk.CTkScrollableFrame(self, width=900, height=500)
        self.scroll_frame.pack(pady=10, fill="both", expand=True)

        self.load_logs()

    def load_logs(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()

        if not os.path.exists(LOG_FILE):
            ctk.CTkLabel(self.scroll_frame, text="No unknown logs found.", font=("Arial", 16)).pack(pady=20)
            return

        with open(LOG_FILE, "r") as file:
            entries = [line.strip() for line in file if "saved to" in line]

        if not entries:
            ctk.CTkLabel(self.scroll_frame, text="No entries found.", font=("Arial", 16)).pack(pady=20)
            return

        # Header row
        header = ctk.CTkFrame(self.scroll_frame)
        header.pack(fill="x", padx=20, pady=(0, 10))

        headers = ["Image", "Date", "Time", "Delete"]
        for i, text in enumerate(headers):
            ctk.CTkLabel(header, text=text, font=("Arial", 14, "bold")).grid(row=0, column=i, padx=10, sticky="w")
            header.grid_columnconfigure(i, weight=1)

        # Log rows
        for row_index, entry in enumerate(reversed(entries)):
            try:
                parts = entry.split(" - ")
                timestamp = parts[0]
                img_path = parts[2].replace("saved to ", "").strip()
                if not os.path.exists(img_path):
                    continue

                date_part, time_part = timestamp.split("_")
                time_part = time_part.replace("-", ":")

                row = ctk.CTkFrame(self.scroll_frame)
                row.pack(fill="x", padx=20, pady=8)

                row.grid_columnconfigure((0, 1, 2, 3), weight=1)

                # Thumbnail
                img = cv2.imread(img_path)
                img = cv2.resize(img, (120, 90))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                thumb_img = ctk.CTkImage(dark_image=pil_img, size=(120, 90))

                def open_image(path=img_path):
                    top = ctk.CTkToplevel(self)
                    top.title("Full Image")
                    top.attributes("-topmost", True)
                    full = Image.open(path)
                    full.thumbnail((600, 400))
                    full_ctk = ctk.CTkImage(dark_image=full, size=full.size)
                    img_label = ctk.CTkLabel(top, image=full_ctk, text="")
                    img_label.image = full_ctk
                    img_label.pack(padx=10, pady=10)

                img_label = ctk.CTkLabel(row, image=thumb_img, text="")
                img_label.image = thumb_img
                img_label.bind("<Button-1>", lambda e, p=img_path: open_image(p))
                img_label.grid(row=0, column=0, padx=10, sticky="w")

                # Date & Time
                ctk.CTkLabel(row, text=date_part, font=("Arial", 13)).grid(row=0, column=1, padx=10, sticky="w")
                ctk.CTkLabel(row, text=time_part, font=("Arial", 13)).grid(row=0, column=2, padx=10, sticky="w")

                # Delete
                def delete_log(path=img_path, line=entry):
                    if os.path.exists(path):
                        os.remove(path)
                    with open(LOG_FILE, "r") as f:
                        lines = f.readlines()
                    with open(LOG_FILE, "w") as f:
                        for l in lines:
                            if l.strip() != line:
                                f.write(l)
                    self.load_logs()

                del_btn = ctk.CTkButton(row, text="Delete", width=80, command=delete_log)
                del_btn.grid(row=0, column=3, padx=10, sticky="e")

            except Exception as e:
                print("Error:", e)
                continue


if __name__ == "__main__":
    app = UnknownViewer()
    app.mainloop()
