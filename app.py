import customtkinter as ctk
from customtkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import math

# Set dark mode and modern blue theme
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class JetPhotosCheckerUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configure window geometry
        self.title("JetPhotos Quality Checker")
        self.geometry("1200x750")
        self.minsize(1000, 600)

        # Track workspace sizing and metrics variables natively
        self.file_path = None
        self.raw_image = None       # Pillow original image instance
        self.tk_image = None        # Saved reference to prevent garbage collection
        self.tk_dust_image = None   # Saved reference to prevent dust view memory dump
        self.img_x, self.img_y = 0, 0
        self.img_w, self.img_h = 0, 0
        
        # Analysis Cache Metrics
        self.detected_tilt = 0.0
        self.offset_x = 0
        self.offset_y = 0
        self.hist_data = None       # Stores computed color channel arrays

        # Configure 3-column grid layout
        self.grid_columnconfigure(0, weight=0, minsize=200)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0, minsize=320)
        self.grid_rowconfigure(0, weight=1)

        self.setup_left_sidebar()
        self.setup_center_viewer()
        self.setup_right_sidebar()

        # Bind window resize event to the canvas
        self.canvas.bind("<Configure>", self.on_window_resize)

    def setup_left_sidebar(self):
        """Left panel containing file loading and analysis toggles."""
        self.left_panel = ctk.CTkFrame(self, corner_radius=0)
        self.left_panel.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        self.title_label = ctk.CTkLabel(self.left_panel, text="JP Checker", font=ctk.CTkFont(size=20, weight="bold"))
        self.title_label.pack(padx=20, pady=20)

        self.btn_open = ctk.CTkButton(self.left_panel, text="Open Photo", command=self.open_photo)
        self.btn_open.pack(padx=20, pady=10, fill="x")

        # Feature Toggles Group
        self.toggle_frame = ctk.CTkFrame(self.left_panel, fg_color="transparent")
        self.toggle_frame.pack(padx=20, pady=20, fill="x")

        self.lbl_features = ctk.CTkLabel(self.toggle_frame, text="Overlays", font=ctk.CTkFont(weight="bold"))
        self.lbl_features.pack(anchor="w", pady=(0, 5))

        self.chk_center = ctk.CTkCheckBox(self.toggle_frame, text="Center Grid (3x3+)", command=self.redraw_canvas)
        self.chk_center.pack(anchor="w", pady=5)

        self.chk_horizon = ctk.CTkCheckBox(self.toggle_frame, text="Horizon Guide", command=self.redraw_canvas)
        self.chk_horizon.pack(anchor="w", pady=5)

        self.chk_dust = ctk.CTkCheckBox(self.toggle_frame, text="Highlight Dust", command=self.redraw_canvas)
        self.chk_dust.pack(anchor="w", pady=5)

        # Connect the analysis trigger button
        self.btn_analyze = ctk.CTkButton(self.left_panel, text="Run Analysis", fg_color="#228B22", hover_color="#006400", command=self.run_image_analysis)
        self.btn_analyze.pack(padx=20, pady=(40, 10), fill="x", side="bottom")

    def setup_center_viewer(self):
        """Center workspace showing the photo inside an interactive canvas."""
        self.center_panel = ctk.CTkFrame(self, fg_color="#1A1A1A")
        self.center_panel.grid(row=0, column=1, sticky="nsew", padx=2, pady=2)
        
        self.center_panel.grid_rowconfigure(0, weight=1)
        self.center_panel.grid_columnconfigure(0, weight=1)

        self.canvas = ctk.CTkCanvas(self.center_panel, bg="#1A1A1A", bd=0, highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=20, pady=20)

    def setup_right_sidebar(self):
        """Right sidebar containing the custom histogram canvas and metrics text fields."""
        self.right_panel = ctk.CTkFrame(self, corner_radius=0)
        self.right_panel.grid(row=0, column=2, sticky="nsew", padx=2, pady=2)

        self.stats_title = ctk.CTkLabel(self.right_panel, text="Analysis Results", font=ctk.CTkFont(size=16, weight="bold"))
        self.stats_title.pack(padx=20, pady=20, anchor="w")

        # Custom Vector Histogram Canvas Frame
        self.hist_frame = ctk.CTkFrame(self.right_panel, height=180, fg_color="#2B2B2B")
        self.hist_frame.pack(padx=20, pady=10, fill="x")
        self.hist_frame.pack_propagate(False)
        
        self.hist_canvas = ctk.CTkCanvas(self.hist_frame, bg="#2B2B2B", bd=0, highlightthickness=0)
        self.hist_canvas.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Display baseline placeholder
        self.render_empty_histogram()

        self.metrics_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.metrics_frame.pack(padx=20, pady=20, fill="x")

        self.metric_labels = {}
        metrics = [
            ("Horizon Tilt:", "0.0°", "tilt"),
            ("Center Offset:", "0px, 0px", "offset"),
            ("Exposure Status:", "Pending", "exposure")
        ]

        for label_text, default_val, key in metrics:
            row = ctk.CTkFrame(self.metrics_frame, fg_color="transparent")
            row.pack(fill="x", pady=6)
            
            lbl = ctk.CTkLabel(row, text=label_text, text_color="#AAAAAA")
            lbl.pack(side="left")
            
            val = ctk.CTkLabel(row, text=default_val, font=ctk.CTkFont(weight="bold"))
            val.pack(side="right")
            self.metric_labels[key] = val

    def render_empty_histogram(self):
        """Draws basic text inside the histogram when no image is active."""
        self.hist_canvas.delete("all")
        self.hist_canvas.create_text(
            150, 85, text="No Histogram Data\nRun Analysis to Generate",
            fill="#666666", justify="center", font=("Arial", 11)
        )

    def open_photo(self):
        """Launches native file selection browser dialog window."""
        file_types = [("Image Files", "*.jpg *.jpeg *.png")]
        self.file_path = filedialog.askopenfilename(title="Select Aviation Photo", filetypes=file_types)
        
        if self.file_path:
            self.raw_image = Image.open(self.file_path)
            # Reset metadata values on a fresh image open
            self.detected_tilt = 0.0
            self.offset_x, self.offset_y = 0, 0
            self.hist_data = None
            self.metric_labels["tilt"].configure(text="0.0°", text_color="#FFFFFF")
            self.metric_labels["offset"].configure(text="0px, 0px", text_color="#FFFFFF")
            self.metric_labels["exposure"].configure(text="Pending", text_color="#FFFFFF")
            self.render_empty_histogram()
            self.redraw_canvas()

    def run_image_analysis(self):
        """Triggers the backend automated math and histogram generation."""
        if not self.raw_image or not self.file_path:
            return

        cv_img = cv2.imread(self.file_path)
        if cv_img is not None:
            # 1. Compute Horizon Tilt
            self.detected_tilt = self.calculate_horizon_angle(cv_img)
            tilt_color = "#FF6347" if abs(self.detected_tilt) > 0.3 else "#228B22"
            self.metric_labels["tilt"].configure(text=f"{self.detected_tilt:+.1f}°", text_color=tilt_color)
            
            # 2. Compute Centering Offset
            self.offset_x, self.offset_y = self.calculate_centering_offset(cv_img)
            offset_color = "#FF6347" if (abs(self.offset_x) > 25 or abs(self.offset_y) > 25) else "#228B22"
            self.metric_labels["offset"].configure(text=f"{self.offset_x:+}px, {self.offset_y:+}px", text_color=offset_color)
            
            # 3. Compute Histogram Arrays (BGR structure from OpenCV)
            gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            hist_luma = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([cv_img], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([cv_img], [1], None, [256], [0, 256]).flatten()
            hist_r = cv2.calcHist([cv_img], [2], None, [256], [0, 256]).flatten()
            
            self.hist_data = {"luma": hist_luma, "r": hist_r, "g": hist_g, "b": hist_b}
            self.draw_vector_histogram()
            
            # 4. Check for Under/Over Exposure (Clipped values at 0 and 255)
            total_pixels = gray.size
            under_exposed = (hist_luma[0] + hist_luma[1]) / total_pixels > 0.02
            over_exposed = (hist_luma[254] + hist_luma[255]) / total_pixels > 0.02
            
            if under_exposed and over_exposed: status, s_color = "High Contrast", "#FFCC00"
            elif under_exposed: status, s_color = "Underexposed", "#FF6347"
            elif over_exposed: status, s_color = "Overexposed / Clipped", "#FF6347"
            else: status, s_color = "Good Exposure", "#228B22"
            
            self.metric_labels["exposure"].configure(text=status, text_color=s_color)
            self.redraw_canvas()

    def calculate_horizon_angle(self, cv_img):
        """Calculates horizon angle based on background linear paths."""
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=200, minLineLength=150, maxLineGap=10)
        
        angles = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line.flatten()
                if abs(x2 - x1) < 1e-5:
                    continue
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                if abs(angle) < 15.0:
                    angles.append(angle)
        
        if len(angles) > 0:
            return float(np.median(angles))
        return 0.0

    def calculate_centering_offset(self, cv_img):
        """Finds the main subject (aircraft) and calculates pixel distance from absolute center."""
        h, w, _ = cv_img.shape
        img_center_x = w // 2
        img_center_y = h // 2

        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            bx, by, bw, bh = cv2.boundingRect(largest_contour)
            subject_center_x = bx + (bw // 2)
            subject_center_y = by + (bh // 2)
            return int(subject_center_x - img_center_x), int(subject_center_y - img_center_y)
        
        return 0, 0

    def draw_vector_histogram(self):
        """Draws smooth, custom mathematical line graphs inside a native Tkinter canvas."""
        if not self.hist_data:
            return
            
        self.hist_canvas.delete("all")
        
        # Fetch actual geometry constraints of the right canvas box
        c_w = self.hist_canvas.winfo_width()
        c_h = self.hist_canvas.winfo_height()
        
        # Prevent math errors if drawn before window pops up completely
        if c_w < 10 or c_h < 10:
            c_w, c_h = 280, 160
            
        # Find maximum frequency peaks to normalize height mapping
        max_peak = max([max(self.hist_data["r"]), max(self.hist_data["g"]), max(self.hist_data["b"]), max(self.hist_data["luma"])])
        if max_peak == 0: max_peak = 1
        
        def get_coords(channel_arr):
            """Maps 256 points into pixel coordinate arrays for the canvas."""
            points = []
            for i in range(256):
                x = (i / 255) * c_w
                # Invert Y axis because canvas (0,0) starts at top-left corner
                y = c_h - ((channel_arr[i] / max_peak) * (c_h - 10))
                points.extend([x, y])
            # FIX: Explicitly convert to standard Python float primitives for NumPy 2.x/Tkinter compatibility
            return list(map(float, points))

        # 1. Draw Luminance filled graph block (Drawn behind colors)
        luma_points = get_coords(self.hist_data["luma"])
        # Formulate solid polygon coordinates by pinning base edges to bottom corners
        polygon_points = [0.0, float(c_h)] + luma_points + [float(c_w), float(c_h)]
        self.hist_canvas.create_polygon(polygon_points, fill="#444444", stipple="gray50", outline="")

        # 2. Draw sharp RGB Vector Curves
        colors_config = [("r", "#CD5C5C"), ("g", "#2E8B57"), ("b", "#4169E1")]
        for key, hex_code in colors_config:
            channel_points = get_coords(self.hist_data[key])
            self.hist_canvas.create_line(channel_points, fill=hex_code, width=1.5, smooth=True)

    def get_dust_equalized_image(self, resized_pil_img):
        """Converts the active image viewport into a high-pass equalized contrast map to expose dust spots."""
        open_cv_img = cv2.cvtColor(np.array(resized_pil_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(open_cv_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        inverted = cv2.bitwise_not(equalized)
        return Image.fromarray(cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB))

    def redraw_canvas(self):
        """Wipes vector assets and repaints image data cleanly to canvas viewports."""
        self.canvas.delete("all")

        if not self.raw_image:
            self.canvas.create_text(
                self.canvas.winfo_width() / 2,
                self.canvas.winfo_height() / 2,
                text="No Image Loaded\n\nClick 'Open Photo' to load an aircraft picture.",
                fill="#666666", justify="center", font=("Arial", 14)
            )
            return
        
        canv_w = max(self.canvas.winfo_width(), 10)
        canv_h = max(self.canvas.winfo_height(), 10)
        img_w, img_h = self.raw_image.size

        ratio = min(canv_w / img_w, canv_h / img_h)
        self.img_w = int(img_w * ratio)
        self.img_h = int(img_h * ratio)

        self.img_x = (canv_w - self.img_w) // 2
        self.img_y = (canv_h - self.img_h) // 2
        
        resized_pil = self.raw_image.resize((self.img_w, self.img_h), Image.Resampling.LANCZOS)

        if self.chk_dust.get():
            dust_pil = self.get_dust_equalized_image(resized_pil)
            self.tk_dust_image = ImageTk.PhotoImage(dust_pil)
            self.canvas.create_image(self.img_x, self.img_y, anchor="nw", image=self.tk_dust_image)
        else:
            self.tk_image = ImageTk.PhotoImage(resized_pil)
            self.canvas.create_image(self.img_x, self.img_y, anchor="nw", image=self.tk_image)
        
        if self.chk_center.get():
            self.draw_jetphotos_grid()
        
        if self.chk_horizon.get():
            self.draw_horizon_guide()

    def draw_jetphotos_grid(self):
        """Draws the tracking 3x3 grid matrix with a dedicated center 2x2 section."""
        x0, y0 = self.img_x, self.img_y
        x3, y3 = self.img_x + self.img_w, self.img_y + self.img_h

        x1 = x0 + self.img_w // 3
        x2 = x0 + (self.img_w * 2) // 3
        y1 = y0 + self.img_h // 3
        y2 = y0 + (self.img_h * 2) // 3
        
        grid_color = "#FFFFFF"
        
        self.canvas.create_line(x1, y0, x1, y3, fill=grid_color, dash=(4, 4), width=1)
        self.canvas.create_line(x2, y0, x2, y3, fill=grid_color, dash=(4, 4), width=1)
        self.canvas.create_line(x0, y1, x3, y1, fill=grid_color, dash=(4, 4), width=1)
        self.canvas.create_line(x0, y2, x3, y2, fill=grid_color, dash=(4, 4), width=1)
        
        mid_x = x0 + self.img_w // 2
        mid_y = y0 + self.img_h // 2
        
        self.canvas.create_line(mid_x, y1, mid_x, y2, fill="#FFCC00", width=1.5)
        self.canvas.create_line(x1, mid_y, x2, mid_y, fill="#FFCC00", width=1.5)

    def draw_horizon_guide(self):
        """Draws an interactive leveling reference line matching the detected image tilt."""
        x0, y0 = self.img_x, self.img_y
        x3, y3 = self.img_x + self.img_w, self.img_y + self.img_h

        mid_y = y0 + self.img_h // 2

        if abs(self.detected_tilt) < 0.01:
            self.canvas.create_line(x0, mid_y, x3, mid_y, fill="#00FF7F", width=2)
        else:
            angle_rad = math.radians(self.detected_tilt)
            half_w = self.img_w / 2

        y_offset = half_w * math.sin(angle_rad)
        left_y = mid_y + y_offset
        right_y = mid_y - y_offset
        
        self.canvas.create_line(x0, left_y, x3, right_y, fill="#00FFFF", width=2, dash=(6, 2))
        self.canvas.create_line(x0, mid_y, x3, mid_y, fill="#00FF7F", width=1, dash=(2, 8))

    def on_window_resize(self, event):
        """Handles window adjustments and refreshes the histogram view coordinates."""
        if self.raw_image:
            self.redraw_canvas()
        if self.hist_data:
            self.draw_vector_histogram()

if __name__ == "__main__":
    app = JetPhotosCheckerUI()
    app.mainloop()