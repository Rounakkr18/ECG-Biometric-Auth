import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import numpy as np
from src import auth_utils

class ECGAuthApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ECG Biometric Authentication")

        # Register Button
        self.btn_register = tk.Button(root, text="Register User", command=self.register_user, width=20, height=2)
        self.btn_register.pack(pady=10)

        # Login Button
        self.btn_login = tk.Button(root, text="Login User", command=self.login_user, width=20, height=2)
        self.btn_login.pack(pady=10)

    def register_user(self):
        username = simpledialog.askstring("Register", "Enter username:")
        if not username:
            return

        file_paths = filedialog.askopenfilenames(
            title="Select ECG Beat Files for Registration",
            filetypes=[("NumPy files", "*.npy")]
        )
        if not file_paths:
            return

        ecg_samples = []
        for path in file_paths:
            try:
                sample = np.load(path)
                ecg_samples.append(sample)
            except Exception as e:
                messagebox.showerror("Error", f"Error loading {path}: {e}")
                return

        auth_utils.register_user(username, ecg_samples)
        messagebox.showinfo("Success", f"User '{username}' registered with {len(ecg_samples)} beats.")

    def login_user(self):
        file_path = filedialog.askopenfilename(
            title="Select ECG Beat File for Login",
            filetypes=[("NumPy files", "*.npy")]
        )
        if not file_path:
            return

        try:
            ecg_sample = np.load(file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Error loading file: {e}")
            return

        user, score = auth_utils.authenticate_user(ecg_sample)
        messagebox.showinfo("Login Result", f"User: {user}\nSimilarity: {score:.4f}")

if __name__ == "__main__":
    root = tk.Tk()
    app = ECGAuthApp(root)
    root.mainloop()
