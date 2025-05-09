import customtkinter as ctk
import os
import shutil
import subprocess

SAVE_DIR = "dataset/known_faces"

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

class DatasetManager(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Dataset Manager")
        self.geometry("600x500")
        self.resizable(False, False)

        label = ctk.CTkLabel(self, text="Persons in dataset:", font=("Arial", 18))
        label.pack(pady=10)

        self.frame = ctk.CTkScrollableFrame(self, width=550, height=400)
        self.frame.pack(pady=10)

        self.refresh_list()

    def refresh_list(self):
        for widget in self.frame.winfo_children():
            widget.destroy()

        persons = [d for d in os.listdir(SAVE_DIR) if os.path.isdir(os.path.join(SAVE_DIR, d))]
        if not persons:
            ctk.CTkLabel(self.frame, text="No persons found.", font=("Arial", 14)).pack(pady=10)
            return

        for person in persons:
            row = ctk.CTkFrame(self.frame)
            row.pack(fill="x", pady=5, padx=10)

            label = ctk.CTkLabel(row, text=person, font=("Arial", 16))
            label.pack(side="left", padx=10)

            del_btn = ctk.CTkButton(row, text="Delete", fg_color="red", command=lambda p=person: self.confirm_delete(p))
            del_btn.pack(side="right", padx=5)

            upd_btn = ctk.CTkButton(row, text="Update", command=lambda p=person: self.update_person(p))
            upd_btn.pack(side="right", padx=5)

    def confirm_delete(self, person):
        confirm = ctk.CTkToplevel(self)
        confirm.title("Confirm Deletion")
        confirm.attributes("-topmost", True)
        confirm.geometry("300x150")
        msg = ctk.CTkLabel(confirm, text=f"Are you sure you want to delete '{person}'?", font=("Arial", 14))
        msg.pack(pady=20)

        def do_delete():
            shutil.rmtree(os.path.join(SAVE_DIR, person))
            confirm.destroy()
            self.refresh_list()

        ctk.CTkButton(confirm, text="Yes", command=do_delete).pack(pady=5)
        ctk.CTkButton(confirm, text="No", command=confirm.destroy).pack()

    def update_person(self, person):
        subprocess.Popen(["python", "video_capture_prompt.py", "--name", person])

if __name__ == "__main__":
    app = DatasetManager()
    app.mainloop()
