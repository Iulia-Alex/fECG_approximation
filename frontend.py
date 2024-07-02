import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
from unet_main import unet_main
import os
import re

# Handle model and file selection from frontend window
def handle_selection():
    selected_model = model_var.get()
    selected_phase = phase_var.get()
    phase = selected_phase.split()[0]
    selected_channel = channel_var.get()
    channel = int(selected_channel.split()[1]) - 1
    print(channel)
    file_path = filedialog.askopenfilename(title="Select a File")
    if file_path:
        file_label.configure(text=f"Selected File: {file_path}")
        directory = os.path.dirname(file_path) + "/"

        # Get the basename
        basename = os.path.basename(file_path)
        basename_without_ext = os.path.splitext(basename)[0]

        # Extract the numeric suffix
        match = re.search(r'(\d+)$', basename_without_ext)
        if match:
            numeric_suffix = match.group(1)
        else:
            numeric_suffix = ""

        hr, hr1, real_hr = unet_main(selected_model, directory, int(numeric_suffix), phase, channel)

        # Update the HR labels
        hr_label.configure(text=f"HR estimated value with first peak detector: {hr} bpm")
        hr1_label.configure(text=f"HR estimated value with second peak detector: {hr1} bpm")
        hrr_label.configure(text=f"Real HR value: {real_hr[0][0]} bpm")

root = ctk.CTk()
root.geometry('600x600')
root.title("Fetal Heart Rate Approximation - Iulia Alexandra Orvas")

# Dropdown menus for model, phase and channel selection
model_var = ctk.StringVar(value="UNet")
model_label = ctk.CTkLabel(root, text="Select model:")
model_label.pack(pady=10)

model_dropdown = ctk.CTkOptionMenu(root, variable=model_var, values=["UNet", "AttentionUnet"])
model_dropdown.pack(pady=10)

phase_var = ctk.StringVar(value="Fetal phase")
phase_label = ctk.CTkLabel(root, text="Select which phase to use for reconstruction:")
phase_label.pack(pady=10)

phase_dropdown = ctk.CTkOptionMenu(root, variable=phase_var, values=["Fetal phase", "Mixture phase"])
phase_dropdown.pack(pady=10)

channel_var = ctk.StringVar(value="Channel 1")
channel_label = ctk.CTkLabel(root, text="Select which channel to use:")
channel_label.pack(pady=10)

channel_dropdown = ctk.CTkOptionMenu(root, variable=channel_var, values=["Channel 1", "Channel 2", "Channel 3", "Channel 4"])
channel_dropdown.pack(pady=10)

load_button = ctk.CTkButton(root, text="Load file and plot signals", command=handle_selection)
load_button.pack(pady=20)

file_label = ctk.CTkLabel(root, text="Selected file: None")
file_label.pack(pady=10)

hrr_label = ctk.CTkLabel(root, text="Real HR value: ")
hrr_label.pack(pady=10)

hr_label = ctk.CTkLabel(root, text="HR estimated value with first peak detector: ")
hr_label.pack(pady=10)

hr1_label = ctk.CTkLabel(root, text="HR estimated value with second peak detector: ")
hr1_label.pack(pady=10)

root.mainloop()