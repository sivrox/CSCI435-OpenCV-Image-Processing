import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class TimeOfDayClassifier:
    """
    A class for classifying images as 'Day' or 'Night' based on various
    color space metrics and histogram analysis.

    Adheres to the 'Time-of-Day Classification' task of the project brief,
    focusing on empirical metrics like mean brightness, histogram distribution,
    and classification accuracy.
    """

    # Defined thresholds based on empirical observations from a small dataset.
    # These can be fine-tuned with a larger, more diverse dataset.
    THRESHOLDS = {
        'rgb_brightness': 115,  # Higher values typically indicate day
        'hsv_v_mean': 115,      # Higher V (Value) indicates brighter images (day)
        'ycbcr_y_mean': 115,    # Higher Y (Luminance) indicates brighter images (day)
        'xyz_y_mean': 115,      # Higher Y (Luminance) indicates brighter images (day)
        'cmyk_k_mean': 130      # Higher K (Black) indicates darker images (night)
    }

    def __init__(self, day_ref_path=None, night_ref_path=None):
        """
        Initializes the TimeOfDayClassifier.
        Optionally loads reference images for histogram comparison.

        Args:
            day_ref_path (str, optional): Path to a clear 'Day' reference image.
            night_ref_path (str, optional): Path to a clear 'Night' reference image.
        """
        self.ref_hist_day = None
        self.ref_hist_night = None

        if day_ref_path and night_ref_path:
            self._load_and_compute_reference_histograms(day_ref_path, night_ref_path)
        else:
            print("TimeOfDayClassifier Warning: Reference images for histogram comparison "
                  "were not provided. 'hsv_v_hist_comp' method will not be available.")

    # --- Private Helper Methods for Color Space Analysis and Histograms ---

    def _bgr_to_cmyk_image(self, bgr_image):
        """
        Converts a BGR OpenCV image to a CMYK image (4 channels).
        Uses NumPy for optimized, vectorized calculations.
        """
        rgb_norm = bgr_image[:, :, ::-1] / 255.0  # Convert BGR to RGB and normalize to [0, 1]
        K = 1 - np.max(rgb_norm, axis=2)          # Calculate Black (K) component

        # Initialize CMY channels, handling division by zero for pure black pixels
        C = np.zeros_like(K)
        M = np.zeros_like(K)
        Y_cmyk = np.zeros_like(K) # Renamed to avoid clash with XYZ's Y

        non_black_pixels = K != 1.0 # Identify pixels that are not pure black

        # Calculate C, M, Y for non-black pixels
        C[non_black_pixels] = (1 - rgb_norm[non_black_pixels, 0] - K[non_black_pixels]) / (1 - K[non_black_pixels])
        M[non_black_pixels] = (1 - rgb_norm[non_black_pixels, 1] - K[non_black_pixels]) / (1 - K[non_black_pixels])
        Y_cmyk[non_black_pixels] = (1 - rgb_norm[non_black_pixels, 2] - K[non_black_pixels]) / (1 - K[non_black_pixels])

        # Clip values to [0, 1] and scale to [0, 255] for uint8 representation
        C = np.clip(C, 0, 1) * 255
        M = np.clip(M, 0, 1) * 255
        Y_cmyk = np.clip(Y_cmyk, 0, 1) * 255
        K = np.clip(K, 0, 1) * 255

        # Stack as a 4-channel image in CMYK order and convert to uint8
        cmyk_image = np.stack([C, M, Y_cmyk, K], axis=-1).astype(np.uint8)
        return cmyk_image

    def _analyze_rgb(self, image):
        """Calculates mean B, G, R and overall brightness for an RGB image."""
        mean_b, mean_g, mean_r = cv2.mean(image)[:3]
        overall_brightness = (mean_b + mean_g + mean_r) / 3
        return {'mean_b': mean_b, 'mean_g': mean_g, 'mean_r': mean_r, 'overall_brightness': overall_brightness}

    def _analyze_hsv(self, image):
        """Converts to HSV and calculates mean H, S, V. Returns HSV image too."""
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mean_h, mean_s, mean_v = cv2.mean(hsv_image)[:3]
        return {'mean_h': mean_h, 'mean_s': mean_s, 'mean_v': mean_v, 'hsv_image': hsv_image}

    def _analyze_ycbcr(self, image):
        """Converts to YCbCr and calculates mean Y, Cb, Cr. Returns YCbCr image too."""
        ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb) # OpenCV uses YCrCb order
        mean_y, mean_cr, mean_cb = cv2.mean(ycbcr_image)[:3] # Order in OpenCV is Y, Cr, Cb
        return {'mean_y': mean_y, 'mean_cb': mean_cb, 'mean_cr': mean_cr, 'ycbcr_image': ycbcr_image}

    def _analyze_xyz(self, image):
        """Converts to XYZ and calculates mean X, Y, Z. Returns XYZ image too."""
        xyz_image = cv2.cvtColor(image, cv2.COLOR_BGR2XYZ)
        mean_x, mean_y, mean_z = cv2.mean(xyz_image)[:3]
        return {'mean_x': mean_x, 'mean_y': mean_y, 'mean_z': mean_z, 'xyz_image': xyz_image}

    def _analyze_cmyk(self, image):
        """Converts to CMYK using custom function and calculates mean C, M, Y, K. Returns CMYK image too."""
        cmyk_image = self._bgr_to_cmyk_image(image)
        mean_c, mean_m, mean_y_cmyk, mean_k = cv2.mean(cmyk_image)[:4]
        return {'mean_c': mean_c, 'mean_m': mean_m, 'mean_y_cmyk': mean_y_cmyk, 'mean_k': mean_k, 'cmyk_image': cmyk_image}

    def _compute_histogram(self, image_channel, bins=256, hist_range=[0, 256]):
        """
        Computes a histogram for a single image channel.
        Normalizes the histogram for consistent comparison.
        """
        if image_channel.dtype != np.uint8:
            image_channel = np.uint8(np.clip(image_channel, hist_range[0], hist_range[1]))
        hist = cv2.calcHist([image_channel], [0], None, [bins], hist_range)
        cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX) # Normalize to [0,1]
        return hist

    def _plot_histogram(self, hist, title, ax, color='blue'):
        """Plots a histogram using matplotlib for visualization."""
        ax.plot(hist, color=color)
        ax.set_title(title)
        ax.set_xlim([0, 256])
        ax.set_xlabel("Pixel Intensity (0-255)")
        ax.set_ylabel("Normalized Frequency")

    def _load_and_compute_reference_histograms(self, day_ref_path, night_ref_path):
        """
        Loads specified day and night images and computes their HSV 'V' channel histograms
        to be used as references for histogram comparison.
        """
        day_img = cv2.imread(day_ref_path)
        if day_img is None:
            print(f"TimeOfDayClassifier Error: Could not load day reference image '{day_ref_path}'. "
                  "Histogram comparison will be unavailable.")
            self.ref_hist_day = None
            return

        hsv_day = cv2.cvtColor(day_img, cv2.COLOR_BGR2HSV)
        self.ref_hist_day = self._compute_histogram(hsv_day[:, :, 2]) # V channel

        night_img = cv2.imread(night_ref_path)
        if night_img is None:
            print(f"TimeOfDayClassifier Error: Could not load night reference image '{night_ref_path}'. "
                  "Histogram comparison will be unavailable.")
            self.ref_hist_night = None
            return

        hsv_night = cv2.cvtColor(night_img, cv2.COLOR_BGR2HSV)
        self.ref_hist_night = self._compute_histogram(hsv_night[:, :, 2]) # V channel

        print("TimeOfDayClassifier: Reference histograms computed for HSV V channel.")

    # --- Public API for Classification and Evaluation ---

    def classify_image(self, image, method='hsv_v_mean'):
        """
        Classifies a single input image (or video frame) as 'Day' or 'Night'.
        This method is designed to be called externally by the main application
        for real-time or single-shot predictions.

        Args:
            image (np.array): The input OpenCV BGR image.
            method (str): The classification method to use. Valid options:
                          'rgb_brightness', 'hsv_v_mean', 'ycbcr_y_mean', 'xyz_y_mean',
                          'cmyk_k_mean', 'hsv_v_hist_comp'
        Returns:
            str: 'Day' or 'Night'
        Raises:
            ValueError: If an invalid classification method is specified.
            RuntimeError: If 'hsv_v_hist_comp' is chosen but reference histograms are not loaded.
        """
        if method == 'rgb_brightness':
            analysis = self._analyze_rgb(image)
            metric_value = analysis['overall_brightness']
            return 'Day' if metric_value > self.THRESHOLDS['rgb_brightness'] else 'Night'
        elif method == 'hsv_v_mean':
            analysis = self._analyze_hsv(image)
            metric_value = analysis['mean_v']
            return 'Day' if metric_value > self.THRESHOLDS['hsv_v_mean'] else 'Night'
        elif method == 'ycbcr_y_mean':
            analysis = self._analyze_ycbcr(image)
            metric_value = analysis['mean_y']
            return 'Day' if metric_value > self.THRESHOLDS['ycbcr_y_mean'] else 'Night'
        elif method == 'xyz_y_mean':
            analysis = self._analyze_xyz(image)
            metric_value = analysis['mean_y']
            return 'Day' if metric_value > self.THRESHOLDS['xyz_y_mean'] else 'Night'
        elif method == 'cmyk_k_mean':
            analysis = self._analyze_cmyk(image)
            metric_value = analysis['mean_k']
            # For CMYK K, higher values indicate more black (night)
            return 'Night' if metric_value > self.THRESHOLDS['cmyk_k_mean'] else 'Day'
        elif method == 'hsv_v_hist_comp':
            if self.ref_hist_day is None or self.ref_hist_night is None:
                raise RuntimeError("Reference histograms not loaded for 'hsv_v_hist_comp'. "
                                   "Ensure day_ref_path and night_ref_path were provided during initialization.")

            analysis = self._analyze_hsv(image)
            current_v_channel = analysis['hsv_image'][:, :, 2] # Extract V channel
            current_hist = self._compute_histogram(current_v_channel)

            # Compare histogram using correlation (higher correlation means more similar)
            # Other options: cv2.HISTCMP_CHISQR (lower is more similar), cv2.HISTCMP_BHATTACHARYYA (lower is more similar)
            day_similarity = cv2.compareHist(self.ref_hist_day, current_hist, cv2.HISTCMP_CORREL)
            night_similarity = cv2.compareHist(self.ref_hist_night, current_hist, cv2.HISTCMP_CORREL)

            return 'Day' if day_similarity > night_similarity else 'Night'
        else:
            raise ValueError(f"Invalid classification method: '{method}'. "
                             "Choose from 'rgb_brightness', 'hsv_v_mean', 'ycbcr_y_mean', 'xyz_y_mean', "
                             "'cmyk_k_mean', 'hsv_v_hist_comp'.")

    def evaluate_and_display_results(self, test_images_data, display_histograms=True):
        """
        Conducts a comprehensive evaluation of all defined time-of-day classification methods
        on a provided dataset. It also displays quantitative metrics (accuracy) and optionally
        visualizes histograms for analysis.

        Args:
            test_images_data (list of dict): Each dict must contain 'path' (image file path)
                                              and 'true_label' ('Day' or 'Night').
            display_histograms (bool): If True, plots HSV 'V' channel histograms for each image.
        Returns:
            dict: A dictionary containing accuracy and individual results for each classification method.
        """
        print("\n--- Time-of-Day Analysis: Individual Image Metrics ---")

        if display_histograms:
            num_images = len(test_images_data)
            # Create a figure and subplots. squeeze=False ensures axes is always 2D array
            fig, axes = plt.subplots(num_images, 2, figsize=(10, 4 * num_images), squeeze=False)

        for i, img_data in enumerate(test_images_data):
            image_path = img_data['path']
            true_label = img_data['true_label']
            image = cv2.imread(image_path)

            if image is None:
                print(f"Error: Could not load image '{image_path}'. Skipping analysis for this image.")
                continue

            print(f"\nImage: {os.path.basename(image_path)} (True: {true_label})")

            rgb_stats = self._analyze_rgb(image)
            hsv_stats = self._analyze_hsv(image)
            ycbcr_stats = self._analyze_ycbcr(image)
            xyz_stats = self._analyze_xyz(image)
            cmyk_stats = self._analyze_cmyk(image)

            print(f"  RGB Brightness: {rgb_stats['overall_brightness']:.2f}")
            print(f"  HSV Mean V: {hsv_stats['mean_v']:.2f}")
            print(f"  YCbCr Mean Y: {ycbcr_stats['mean_y']:.2f}")
            print(f"  XYZ Mean Y: {xyz_stats['mean_y']:.2f}")
            print(f"  CMYK Mean K: {cmyk_stats['mean_k']:.2f}")

            if display_histograms:
                ax_img = axes[i, 0]
                ax_hist = axes[i, 1]

                ax_img.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convert for matplotlib display
                ax_img.set_title(f"{os.path.basename(image_path)} (True: {true_label})")
                ax_img.axis('off') # Hide axes for image display

                self._plot_histogram(self._compute_histogram(hsv_stats['hsv_image'][:, :, 2]),
                                     f"HSV 'V' Hist - {os.path.basename(image_path)}", ax_hist, color='orange')

        if display_histograms:
            plt.tight_layout()
            plt.show() # Display all generated histogram plots

        print("\n--- Time-of-Day Analysis: Evaluating Classifiers ---")
        all_methods_results = {}

        methods_to_evaluate = [
            'rgb_brightness',
            'hsv_v_mean',
            'ycbcr_y_mean',
            'xyz_y_mean',
            'cmyk_k_mean',
            'hsv_v_hist_comp'
        ]

        for method in methods_to_evaluate:
            correct_predictions = 0
            total_predictions = 0

            # Skip histogram comparison if references weren't loaded
            if method == 'hsv_v_hist_comp' and (self.ref_hist_day is None or self.ref_hist_night is None):
                print(f"\nSkipping method '{method}': Reference histograms are not available.")
                all_methods_results[method] = {'accuracy': 0.0, 'individual_results': [], 'skipped': True}
                continue

            for img_data in test_images_data:
                image_path = img_data['path']
                true_label = img_data['true_label']

                image = cv2.imread(image_path)
                if image is None:
                    continue # Already warned, just skip calculation

                try:
                    predicted_label = self.classify_image(image, method)
                    is_correct = (predicted_label == true_label)
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                except RuntimeError as e:
                    # This handles the case where ref histograms are missing, but code flow already handles this
                    # Can be re-raised or logged more verbosely if needed
                    pass
                except ValueError as e:
                    print(f"Error during classification for method '{method}': {e}. Skipping image.")
                    continue

            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0
            all_methods_results[method] = {'accuracy': accuracy, 'skipped': False}
            print(f"\nMethod: {method}")
            print(f"  Accuracy: {accuracy:.2f}%")

        print("\n--- Time-of-Day Analysis: Summary of Results ---")
        best_method = None
        max_accuracy = -1.0

        for method, data in all_methods_results.items():
            if data.get('skipped', False):
                print(f"{method}: Skipped (References missing)")
            else:
                print(f"{method}: Accuracy = {data['accuracy']:.2f}%")
                if data['accuracy'] > max_accuracy:
                    max_accuracy = data['accuracy']
                    best_method = method

        if best_method:
            print(f"\nBest performing method: {best_method} with {max_accuracy:.2f}% accuracy.")
        else:
            print("\nNo methods could be fully evaluated.")

        return all_methods_results

    def demonstrate_classification_on_samples(self, sample_day_path, sample_night_path):
        """
        Demonstrates the classification of two sample images (one day, one night)
        using both a mean-based method and histogram comparison (if available).

        Args:
            sample_day_path (str): Path to a sample daytime image.
            sample_night_path (str): Path to a sample nighttime image.
        """
        print("\n--- Time-of-Day Analysis: Single Image Classification Demonstration ---")

        # --- Sample Day Image ---
        sample_day_image = cv2.imread(sample_day_path)
        if sample_day_image is None:
            print(f"Error: Could not load sample day image '{sample_day_path}'. Skipping.")
        else:
            print(f"'{os.path.basename(sample_day_path)}' (True: Day)")

            # Using 'hsv_v_mean'
            predicted_day_mean = self.classify_image(sample_day_image, method='hsv_v_mean')
            print(f"  Classified as: {predicted_day_mean} (using 'hsv_v_mean' method)")

            # Using 'hsv_v_hist_comp'
            try:
                predicted_day_hist = self.classify_image(sample_day_image, method='hsv_v_hist_comp')
                print(f"  Classified as: {predicted_day_hist} (using 'hsv_v_hist_comp' method)")
            except RuntimeError as e:
                print(f"  Classified as: N/A (using 'hsv_v_hist_comp' method) - {e}")

        # --- Sample Night Image ---
        sample_night_image = cv2.imread(sample_night_path)
        if sample_night_image is None:
            print(f"Error: Could not load sample night image '{sample_night_path}'. Skipping.")
        else:
            print(f"\n'{os.path.basename(sample_night_path)}' (True: Night)")

            # Using 'hsv_v_mean'
            predicted_night_mean = self.classify_image(sample_night_image, method='hsv_v_mean')
            print(f"  Classified as: {predicted_night_mean} (using 'hsv_v_mean' method)")

            # Using 'hsv_v_hist_comp'
            try:
                predicted_night_hist = self.classify_image(sample_night_image, method='hsv_v_hist_comp')
                print(f"  Classified as: {predicted_night_hist} (using 'hsv_v_hist_comp' method)")
            except RuntimeError as e:
                print(f"  Classified as: N/A (using 'hsv_v_hist_comp' method) - {e}")


# --- Standalone Execution Example ---
if __name__ == "__main__":
    # Define the dataset for testing. Ensure these image files are in the same directory.
    # For a real project, this dataset would be much larger and more diverse.
    my_test_dataset = [
        {'path': "image_dataset/burj_khalifa_day.jpg", 'true_label': 'Day'},
        {'path': "image_dataset/burj_khalifa_night.jpg", 'true_label': 'Night'} # This is the challenging "bright night" image
    ]

    # Define the reference images for histogram comparison.
    # These should be distinctly 'Day' and 'Night' for best results.
    # In a larger project, these might come from a dedicated 'training' subset.
    day_ref_img = "image_dataset/burj_khalifa_day.jpg"
    night_ref_img = "image_dataset/burj_khalifa_night.jpg"

    # Initialize the classifier, providing reference image paths
    classifier = TimeOfDayClassifier(day_ref_path=day_ref_img, night_ref_path=night_ref_img)

    # Run the full analysis, evaluation, and display of results
    # Set display_histograms=False if you don't want the plots to pop up during evaluation
    results = classifier.evaluate_and_display_results(my_test_dataset, display_histograms=True)

    # Demonstrate classification on specific sample images
    classifier.demonstrate_classification_on_samples(
        sample_day_path="image_dataset/dmcc_ms.jpg",
        sample_night_path="image_dataset/barsha_mosque.jpg"
    )