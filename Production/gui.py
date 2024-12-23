import os
import customtkinter as ctk
from PIL import Image, ImageTk, ImageDraw
from tkinterdnd2 import DND_FILES, TkinterDnD
from tkinter import Tk, Label, Button, Canvas, filedialog, Frame

from system import *

# Initialize the GUI window with drag-and-drop support
root = TkinterDnD.Tk()  # Enables drag-and-drop

root.title("Gate Access Control")
root.geometry("800x700+20+20")
root.configure(bg="#329171")  # Background color

# Title Label
title_label = ctk.CTkLabel(root, text="Gate Access Control", font=("Helvetica", 20, "bold"), bg_color="transparent", text_color="#FFFFFF")
title_label.pack(pady=10)

# Frame for image display
frame = ctk.CTkFrame(root, bg_color="transparent", border_width=5)
frame.pack(pady=10)

canvas = ctk.CTkCanvas(frame, width=400, height=400, bg="#ECF0F1")
canvas.pack(padx=10, pady=10)

# Label to show file status
label = ctk.CTkLabel(root, text="Drag and Drop an Image or Click 'Select Image'.", font=("Arial", 15), bg_color="transparent",text_color="#FFFFFF")
label.pack(pady=10)

# Load the labels and model
labels = read_labels_file("labels.txt")

# Load the KNN model
knn_model = joblib.load("../model_1.pkl")
threshold = 1.50

# Function to process image and extract text
def process_image_gui(file_path):
    try:
        # function salah hena
        #image = Image.open(file_path)
        plate_num = process_image(file_path, knn_model, threshold)
        text_label.configure(text=f"Plate Number: {plate_num}", font=("Arial", 18), text_color="#FFFFFF")
    except Exception as e:
        text_label.configure(text=f"Error Plate Number: {e}", fg_color="#E74C3C", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)

# Function to display the selected image
def display_image(file_path):
    try:
        img = Image.open(file_path)
        img = img.resize((400, 400))  # Resize 
        img_tk = ImageTk.PhotoImage(img)
        canvas.create_image(200, 200, anchor="center", image=img_tk)
        canvas.image = img_tk  # Keep reference to avoid garbage collection
        process_image_gui(file_path)
    except Exception as e:
        label.configure(text=f"Error loading image: {e}", fg_color="#E74C3C", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)

# Function to handle drag-and-drop
def drop_image(event):
    file_path = event.data
    if file_path.startswith('{') and file_path.endswith('}'):  # Handles paths with spaces
        file_path = file_path[1:-1]
    if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        label.configure(text="Image loaded successfully!",  fg_color="#2ECC71", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)
        display_image(file_path)
    else:
        label.configure(text="Invalid file type. Please drop an image.", fg_color="#E74C3C", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)

# Function to select an image
def select_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
    if file_path:
        label.configure(text="Image loaded successfully!", fg_color="#2ECC71", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)
        display_image(file_path)
    else:
        label.configure(text="Failed to load image.", fg_color="#E74C3C", bg_color="transparent", text_color="#FFFFFF", corner_radius=5,width=200, height=25)

# Enable drag-and-drop on the canvas
root.drop_target_register(DND_FILES)
root.dnd_bind('<<Drop>>', drop_image)

# Buttons
menu_button1 = ctk.CTkButton(root, text='Select image', text_color="#FFFFFF", command=select_image, fg_color="#08b6d9", hover_color="#087bb8", width=200, height=35, font=("Arial", 16, "bold"))
menu_button1.pack(pady=10)

# Label for extracted text
text_label =ctk.CTkLabel(root, text="Plate Number will appear here.", wraplength=500, justify="left", font=("Arial", 15), bg_color="transparent", text_color="#FFFFFF")
text_label.pack(pady=10)

# Exit button
exit_button = ctk.CTkButton(root, text="Exit", command=root.destroy, font=("Arial", 16, "bold"), bg_color="transparent", fg_color="#e00808", hover_color="#ac0808", width=100, height=35)
exit_button.pack(pady=5)

# Run the application
root.mainloop()
