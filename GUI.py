import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageEnhance
import torch
from tkinter import ttk

def load_model():
    model = torch.load('your_model.pth')
    model.eval()
    return model

def process_image(model, image_path):
    processed_image = Image.open(image_path) # provisional task
    return processed_image

# GUI class
class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        #self.model = load_model()
        
        self.original_image = None
        self.processed_image = None
        self.image_path = None
        self.brightness_levels = [1.0, 0.8, 0.6, 0.4, 0.2]
        self.setup_gui()
    def setup_gui(self):
        self.root.title('Image Processing GUI')

        # Create a main frame
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create a canvas to display the input image
        self.input_canvas = tk.Canvas(self.main_frame, width=300, height=300)
        self.input_canvas.grid(row=0, column=0)

        # Create a canvas to display the output image
        self.output_canvas = tk.Canvas(self.main_frame, width=300, height=300)
        self.output_canvas.grid(row=0, column=2)

        # Create a button to upload image
        self.upload_button = tk.Button(self.main_frame, text='画像をアップロード', command=self.upload_image)
        self.upload_button.grid(row=1, column=0)

        # Create a radio button to adjust brightness
        self.brightness_var = tk.DoubleVar(value=1.0)  # set a initial value
        self.brightness_frame = tk.Frame(self.main_frame)
        self.brightness_frame.grid(row=1, column=2)
        
        for i, level in enumerate(self.brightness_levels):
            tk.Radiobutton(
                self.brightness_frame,
                text=f'darkness level {i+1}',
                variable=self.brightness_var,
                value=level
            ).pack(side=tk.LEFT)

        # Create a button to action
        self.execute_button = tk.Button(self.main_frame, text='画像を調整', command=self.adjust_brightness)
        self.execute_button.grid(row=0, column=1)

        # Create a button to save image
        self.save_button = tk.Button(self.main_frame, text='画像を保存', command=self.save_image)
        self.save_button.grid(row=1, column=1)

    
    def upload_image(self):
        # Open th file select dialog
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.original_image = Image.open(self.image_path)
            self.display_image(self.original_image, self.input_canvas)

    def display_image(self, image, canvas):
        # Display the image on canvas 
        display_image = image.copy()  # Create a original image copy
        display_image.thumbnail((300, 300), Image.ANTIALIAS)  # Adjust the size to the canvas
        photo = ImageTk.PhotoImage(display_image)
        canvas.create_image(150, 150, image=photo)  # Place image in the center of the canvas
        canvas.image = photo  # Retain references

    def adjust_brightness(self):
        # Exception handling when images are not uploaded
        if self.original_image is None:
            messagebox.showerror("エラー", "画像がアップロードされていません")
            return

        enhancer = ImageEnhance.Brightness(self.original_image)
        brightness_level = self.brightness_var.get()
        self.processed_image = enhancer.enhance(brightness_level)
        self.display_image(self.processed_image, self.output_canvas)

    def save_image(self):
        # Save processed images
        if self.processed_image is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                     filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
            if file_path:
                self.processed_image.save(file_path)
        else:
            messagebox.showerror("エラー", "保存する画像がありません")
# Create a main window
root = tk.Tk()
app = ImageProcessingApp(root)
root.mainloop()
