import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox, Scale
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk, ImageOps, ImageFilter, ImageEnhance
import cv2
import numpy as np


class ImageProcessingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Processing App")
        self.root.resizable(False, False)
        self.image = None
        self.processed_image = None
        self.threshold = 100  # Default binary threshold
        self.current_mode = None  # Track the current mode
        self.smoothness_factor = 1.0  # Default smoothness
        self.sharpness_factor = 1.0  # Default sharpness
        self.original_image = None  # Keep a copy of the original image
        self.original_height = 0
        self.original_width = 0

        # UI Elements
        self.canvas_frame = tk.Frame(root)
        self.canvas = tk.Canvas(self.canvas_frame, width=900, height=600, bg='gray')
        self.canvas.pack()
        self.canvas_frame.grid(row=1, column=1)

        self.control_frame = tk.Frame(root)

        # Button Row (Load, Save)
        self.btn_row = tk.Frame(self.control_frame)
        self.btn_load = tk.Button(self.btn_row, text="Load Image", command=self.load_image)
        self.btn_load.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_reset = tk.Button(self.btn_row, text="Reset", command=self.init_image, state=tk.DISABLED)
        self.btn_reset.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_rotate = tk.Button(self.btn_row, text="Rotate 90°", command=self.rotate_image, state=tk.DISABLED)
        self.btn_rotate.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_resize = tk.Button(self.btn_row, text="Resize", command=self.open_resize_window, state=tk.DISABLED)
        self.btn_resize.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_save = tk.Button(self.btn_row, text="Save Image", command=self.save_image, state=tk.DISABLED)
        self.btn_save.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_row.grid(row=1, column=1)

        # Combobox Row
        self.combo_row = tk.Frame(self.control_frame)
        self.color_label = tk.Label(self.combo_row, text="Color")
        self.color_label.pack(side=tk.LEFT, padx=5)
        self.mode_combobox = ttk.Combobox(self.combo_row, values=["Original Colors", "Negative", "Grayscale", "Inverse Grayscale", "Binary", "Inverse Binary"], state=tk.DISABLED)
        self.mode_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.mode_combobox.bind("<<ComboboxSelected>>", self.apply_mode)

        self.filter_label = tk.Label(self.combo_row, text="Filter")
        self.filter_label.pack(side=tk.LEFT, padx=5)
        self.filter_combobox = ttk.Combobox(self.combo_row, values=["None", "Blur", "Sharpen", "Noise"], state=tk.DISABLED)
        self.filter_combobox.pack(side=tk.LEFT, padx=5, pady=5)
        self.filter_combobox.bind("<<ComboboxSelected>>", self.update_image)

        self.blur_combobox = ttk.Combobox(self.combo_row, values=["Gaussian", "Average (Mean)", "Median", "Maximum", "Minimum"], state=tk.DISABLED)
        self.blur_combobox.bind("<<ComboboxSelected>>", self.update_image)
        self.combo_row.grid(row=2, column=1)


        # Slider Row (Threshold, Effect Intensity)
        self.slider_row = tk.Frame(self.control_frame)
        self.threshold_slider = Scale(self.slider_row, from_=0, to=255, orient=tk.HORIZONTAL, label="                   Binary Threshold", length=200, state=tk.DISABLED)
        self.threshold_slider.set(100)
        self.threshold_slider.bind("<ButtonRelease-1>", self.update_binary_image)

        self.effect_slider = Scale(self.slider_row, from_=0, to=10, orient=tk.HORIZONTAL, label="               Effect Intensity", length=200, state=tk.DISABLED)
        self.effect_slider.set(0)
        self.effect_slider.bind("<ButtonRelease-1>", self.update_image)
        self.slider_row.grid(row=3, column=1)

        # Button Row 2 (Histogram Buttons)
        self.btn_row2 = tk.Frame(self.control_frame)
        self.btn_hist_gray = tk.Button(self.btn_row2, text="Show Grayscale Histogram", command=self.show_gray_hist, state=tk.DISABLED)
        self.btn_hist_gray.pack(side=tk.LEFT, padx=5, pady=5)

        self.btn_hist_color = tk.Button(self.btn_row2, text="Show RGB Histogram", command=self.show_color_hist, state=tk.DISABLED)
        self.btn_hist_color.pack(side=tk.RIGHT, padx=5, pady=5)
        self.btn_row2.grid(row=4, column=1)

        self.control_frame.grid(row=2, column=1)

    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not filepath:
            return
        self.loaded_image = cv2.imread(filepath)
        self.loaded_image = cv2.cvtColor(self.loaded_image, cv2.COLOR_BGR2RGB)
        self.init_image()
        self.update_control_state()
        self.original_height = self.loaded_image.shape[0]
        self.original_width = self.loaded_image.shape[1]
    
    def init_image(self):
        self.image = self.processed_image = self.original_image = self.loaded_image # Keep a copy of the original image
        self.display_image()
        self.mode_combobox.set(value="Original Colors")
        self.filter_combobox.set(value="None")
        self.blur_combobox.set(value="Gaussian")

    def display_image(self):
        if self.processed_image is None:
            return

        # Ensure image dimensions are within canvas limits
        if len(self.image.shape) == 2:
            height, width = self.image.shape
        else:
            height, width, _ = self.image.shape
        canvas_width, canvas_height = self.canvas.winfo_width(), self.canvas.winfo_height()
        # Initialize resized_image to the original processed_image by default
        resized_image = self.processed_image
        if width > canvas_width or height > canvas_height:
            scale_factor = min(canvas_width / width, canvas_height / height)
            resized_image = cv2.resize(self.processed_image, (int(width * scale_factor), int(height * scale_factor)))

        img_tk = ImageTk.PhotoImage(Image.fromarray(resized_image))
        self.canvas.image = img_tk  # Keep a reference to avoid garbage collection
        self.canvas.create_image(canvas_width // 2, canvas_height // 2, image=img_tk)

    def update_control_state(self):
        # Enable controls only if an image is loaded
        state = tk.NORMAL if self.image is not None else tk.DISABLED
        self.mode_combobox.config(state=state)
        self.filter_combobox.config(state=state)
        self.blur_combobox.config(state=state)
        self.threshold_slider.config(state=state)
        self.effect_slider.config(state=state)
        self.btn_save.config(state=state)
        self.btn_hist_color.config(state=state)
        self.btn_hist_gray.config(state=state)
        self.btn_rotate.config(state=state)
        self.btn_resize.config(state=state)
        self.btn_reset.config(state=state)
    
    def rotate_image(self):
        if self.image is not None:
            self.original_image = cv2.rotate(self.original_image, cv2.ROTATE_90_CLOCKWISE)
            self.image = self.processed_image = cv2.rotate(self.processed_image, cv2.ROTATE_90_CLOCKWISE)
            self.display_image()
    
    def resize_image(self, new_width, new_height):
        self.image = self.processed_image = cv2.resize(self.processed_image, (new_width, new_height))
        self.display_image()

    def open_resize_window(self):
        def reset_size():
            width = self.original_width
            height = self.original_height
            self.resize_image(width, height)
            resize_window.destroy()  # Close the resize window

        def apply_resize():
            try:
                width = int(width_entry.get())
                height = int(height_entry.get())
                self.resize_image(width, height)
                resize_window.destroy()  # Close the resize window
            except ValueError:
                messagebox.showerror("Invalid Input", "Please enter valid integers for width and height.")

        def cancel_resize():
            resize_window.destroy()  # Close the resize window

        # Create the pop-up window
        resize_window = tk.Toplevel(root)
        resize_window.title("Resize Image")
        resize_window.geometry("300x170")
        
        # Add input fields and labels
        tk.Label(resize_window, text="Width:").pack(pady=5)
        width_entry = tk.Entry(resize_window)
        width_entry.pack(pady=5)
        
        tk.Label(resize_window, text="Height:").pack(pady=5)
        height_entry = tk.Entry(resize_window)
        height_entry.pack(pady=5)
        
        # Add OK, Cancel, and Reset buttons
        button_frame = tk.Frame(resize_window)
        button_frame.pack(pady=10)
        tk.Button(button_frame, text="OK", command=apply_resize).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Cancel", command=cancel_resize).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="Reset Original Size", command=reset_size).pack(side=tk.LEFT, padx=5)

    def show_original(self):
        if self.image is not None:
            self.processed_image = self.image = self.original_image
            self.current_mode = None  # Reset current mode
            self.display_image()

    def apply_negative(self):
        if self.image is not None:
            self.processed_image = self.image = 255 - self.original_image
            self.display_image()

    def apply_grayscale(self):
        if self.image is not None:
            self.processed_image = self.image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image()


    def apply_inverse_grayscale(self):
        if self.image is not None:
            self.processed_image = self.image = 255 - cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
            self.display_image()

    def activate_binary(self):
        if self.image is not None:
            self.current_mode = 'binary'
            self.update_binary_image(None)

    def activate_inverse_binary(self):
        if self.image is not None:
            self.current_mode = 'inverse_binary'
            self.update_binary_image(None)

    def apply_mode(self, event):
        self.threshold_slider.pack_forget()
        selected_mode = self.mode_combobox.get()

        if selected_mode == "Original Colors":
            self.show_original()
        elif selected_mode == "Negative":
            self.apply_negative()
        elif selected_mode == "Grayscale":
            self.apply_grayscale()
        elif selected_mode == "Inverse Grayscale":
            self.apply_inverse_grayscale()
        elif selected_mode == "Binary":
            self.activate_binary()
            self.threshold_slider.pack(side=tk.LEFT, fill=tk.X)
        elif selected_mode == "Inverse Binary":
            self.activate_inverse_binary()
            self.threshold_slider.pack(side=tk.LEFT, fill=tk.X)
        else:
            self.mode_combobox.set("Original Colors")
            self.show_original()

    def update_binary_image(self, event):
        if self.image is None:
            return
        threshold = self.threshold_slider.get()
        gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)

        if self.current_mode == 'binary':
            _, self.processed_image = _, self.image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        elif self.current_mode == 'inverse_binary':
            _, self.processed_image = _, self.image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY_INV)

        self.display_image()

    def update_image(self, event):
        if self.original_image is None:
            return

        selected_filter = self.filter_combobox.get()
        effect_factor = self.effect_slider.get()
        self.blur_combobox.pack_forget()
        self.effect_slider.pack_forget()
        if selected_filter == "None":
            self.processed_image = self.image
        elif selected_filter == "Sharpen":
            self.effect_slider.pack(side=tk.RIGHT, fill=tk.X)
            if effect_factor == 0:
                self.processed_image = self.image
            else:
                # Apply sharpening
                pil_image = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
                # Apply Unsharp Mask sharpening
                pil_image = pil_image.filter(ImageFilter.UnsharpMask(radius=2, percent=effect_factor * 20))
                # Convert PIL Image back to NumPy array
                self.processed_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        elif selected_filter == "Blur":
            self.blur_combobox.pack(side=tk.LEFT, padx=5, pady=5)
            self.effect_slider.pack(side=tk.RIGHT, fill=tk.X)
            self.apply_blur()

        self.display_image()
    
    def apply_blur(self):
        effect_factor = self.effect_slider.get()
        blur_option = self.blur_combobox.get()
        if effect_factor == 0:
            self.processed_image = self.image
        else:
            if blur_option == "Gaussian":
                self.processed_image = cv2.GaussianBlur(self.image, (0, 0), abs(effect_factor))
            elif blur_option == "Average (Mean)":
                if effect_factor % 2 == 0:
                    effect_factor -= 1
                kernel_size = (effect_factor, effect_factor)
                self.processed_image = cv2.blur(self.image, kernel_size)
            elif blur_option == "Median":
                if effect_factor % 2 == 0:
                    effect_factor -= 1
                self.processed_image = cv2.medianBlur(self.image, effect_factor)
            elif blur_option == "Minimum":
                if effect_factor % 2 == 0:
                    effect_factor -= 1
                pil_image = Image.fromarray(self.image)
                pil_image = pil_image.filter(ImageFilter.MinFilter(effect_factor))
                self.processed_image = np.array(pil_image)
            elif blur_option == "Maximum":
                if effect_factor % 2 == 0:
                    effect_factor -= 1
                pil_image = Image.fromarray(self.image)
                pil_image = pil_image.filter(ImageFilter.MaxFilter(effect_factor))
                self.processed_image = np.array(pil_image)
    
    def show_gray_hist(self):
        if self.original_image.any:
        # Convert the image to grayscale and get the pixel values
            gray_img = self.original_image
            gray_img = cv2.cvtColor(gray_img, cv2.COLOR_RGB2GRAY)

            # Create a new Tkinter window for the histogram
            hist_window = tk.Toplevel(root)
            hist_window.title("Grayscale Histogram")

            gray_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 255])
            # Plot the histogram using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(gray_hist)
            ax.set_title("Grayscale Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

            # Embed the plot into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.get_tk_widget().pack()
            canvas.draw()

            # Ensure the canvas is properly destroyed when the window is closed
            def on_close_histogram():
                canvas.get_tk_widget().destroy()  # Destroy the canvas widget
                plt.close(fig)  # Close the Matplotlib figure
                hist_window.destroy()  # Destroy the Tkinter window

            hist_window.protocol("WM_DELETE_WINDOW", on_close_histogram)
    
    def show_color_hist(self):
        if self.original_image.any:
        # Convert the image to grayscale and get the pixel values
            img = self.original_image
            # Create a new Tkinter window for the histogram
            hist_window = tk.Toplevel(root)
            hist_window.title("RGB Histogram")

            # Plot the histogram using Matplotlib
            fig, ax = plt.subplots(figsize=(6, 4))
            color = ['r', 'g', 'b']
            for i in range(len(color)):
                color_hist = cv2.calcHist([img], [i], None, [256], [0, 255])
                ax.plot(color_hist, color[i])
            ax.set_title("RGB Histogram")
            ax.set_xlabel("Pixel Intensity")
            ax.set_ylabel("Frequency")

            # Embed the plot into the Tkinter window
            canvas = FigureCanvasTkAgg(fig, master=hist_window)
            canvas.get_tk_widget().pack()
            canvas.draw()

            # Ensure the canvas is properly destroyed when the window is closed
            def on_close_histogram():
                canvas.get_tk_widget().destroy()  # Destroy the canvas widget
                plt.close(fig)  # Close the Matplotlib figure
                hist_window.destroy()  # Destroy the Tkinter window

            hist_window.protocol("WM_DELETE_WINDOW", on_close_histogram)

    def save_image(self):
        if not self.processed_image.any:
            messagebox.showerror("Error", "No processed image to save!")
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if not filepath:
            return

        # Determine the image type based on shape
        if len(self.processed_image.shape) == 2:  # Grayscale or binary
            cv2.imwrite(filepath, self.processed_image)
        else:  # Color image
            cv2.imwrite(filepath, cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2RGB))

        messagebox.showinfo("Success", "Image saved successfully!")

# Main loop
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingApp(root)
    root.mainloop()
