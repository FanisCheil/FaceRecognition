import customtkinter as ctk
import tkinter.messagebox
import subprocess
import os

# Set appearance
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ------------------- MAIN MENU ----------------------
def open_main_menu():
    menu = ctk.CTkToplevel()
    menu.geometry("500x450")
    menu.title("Face Recognition System")

    # Close both windows if menu is closed
    menu.protocol("WM_DELETE_WINDOW", lambda: app.destroy())

    label = ctk.CTkLabel(menu, text="Choose an action", font=("Arial", 20))
    label.pack(pady=30)

    def access_dataset():
        subprocess.Popen(["python", "dataset_manager.py"])

    def access_unknowns():
        subprocess.Popen(["python", "see_unknowns.py"])

    def run_recognition():
        subprocess.Popen(["python", "real_time_recognition.py"])

    def add_person():
        subprocess.Popen(["python", "video_capture_prompt.py"])


    btn1 = ctk.CTkButton(menu, text="Start Live Recognition", command=run_recognition)
    btn1.pack(pady=15)

    btn2 = ctk.CTkButton(menu, text="Add New Person", command=add_person)
    btn2.pack(pady=15)

    btn3 = ctk.CTkButton(menu, text="Access Dataset", command=access_dataset)
    btn3.pack(pady=15)
    btn4 = ctk.CTkButton(menu, text="See Unknown Captures", command=access_unknowns)
    btn4.pack(pady=15)


# ------------------- LOGIN WINDOW -------------------
app = ctk.CTk()
app.geometry("400x300")
app.title("Face Recognition Login")

title = ctk.CTkLabel(app, text="Login", font=("Arial", 24))
title.pack(pady=20)

username_entry = ctk.CTkEntry(app, placeholder_text="Username")
username_entry.pack(pady=10)

password_entry = ctk.CTkEntry(app, placeholder_text="Password", show="*")
password_entry.pack(pady=10)

def login():
    user = username_entry.get()
    password = password_entry.get()

    if user == "admin" and password == "1234":
        tkinter.messagebox.showinfo("Login Successful", "Welcome!")
        open_main_menu()
        app.withdraw()
    else:
        tkinter.messagebox.showerror("Error", "Incorrect credentials.")

login_btn = ctk.CTkButton(app, text="Login", command=login)
login_btn.pack(pady=20)

app.mainloop()
