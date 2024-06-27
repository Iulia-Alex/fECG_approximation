import customtkinter as ctk
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np

# Function to handle model selection and file loading
def handle_selection():
    selected_model = model_var.get()
    file_path = filedialog.askopenfilename(title="Select a File")
    if file_path:
        file_label.config(text=f"Selected File: {file_path}")
        plot_signals()

# Function to plot two example signals
def plot_signals():
    # Generate example signals
    x = np.linspace(0, 10, 1000)
    signal1 = np.sin(x)
    signal2 = np.cos(x)

    # Create a new window for plotting
    plot_window = ctk.CTkToplevel()
    plot_window.title("Plotted Signals")
    
    # Create the plot figure and axis
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))
    ax[0].plot(x, signal1, label='Signal 1')
    ax[0].legend()
    ax[1].plot(x, signal2, label='Signal 2')
    ax[1].legend()

    # Display the plot in the new window using Canvas
    canvas = plt.FigureCanvasTkAgg(fig, master=plot_window)
    canvas.draw()
    canvas.get_tk_widget().pack()

# Main window
root = ctk.CTk()
root.geometry('500x500')
root.title("Fetal Heart Rate Approximation")

# Dropdown menu for model selection
model_var = ctk.StringVar(value="UNet")
model_label = ctk.CTkLabel(root, text="Select Model:")
model_label.pack(pady=10)

model_dropdown = ctk.CTkOptionMenu(root, variable=model_var, values=["UNet", "AttentionUnet"])
model_dropdown.pack(pady=10)

# Button to load file and plot signals
load_button = ctk.CTkButton(root, text="Load File and Plot Signals", command=handle_selection)
load_button.pack(pady=20)

# Label to display the selected file
file_label = ctk.CTkLabel(root, text="Selected File: None")
file_label.pack(pady=10)

# Run the main loop
root.mainloop()