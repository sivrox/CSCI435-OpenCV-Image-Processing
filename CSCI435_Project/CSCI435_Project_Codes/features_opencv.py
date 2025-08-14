import cv2
import numpy as np
import os
from tkinter import messagebox
from tkinter.simpledialog import askstring
from tkinter.colorchooser import askcolor
dataset = "image_dataset"

#preprocessing input image
def resize_image_cv(image, height=600):
    h, w = image.shape[:2]
    ratio = height / h
    dim = (int(w * ratio), height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

#omputing histogram of the input image
def get_histogram(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist

#----------FEATURE 1----------
def feature1_landmark_face(input_path, output_dir="output_directory"):
    image = cv2.imread(input_path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    output = image.copy()

    #face Detection
    faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 5)
        cv2.putText(output, "Face_Detected", (x, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    #landmark detection
    img_ref = cv2.imread("image_dataset/pisa_tower.png", cv2.IMREAD_GRAYSCALE)
    img_query = img_gray  # already grayscale

    if img_ref is None:
        print(f"Error: Could not load reference image.")
    else:
        sift = cv2.SIFT_create()
        kp_ref, des_ref = sift.detectAndCompute(img_ref, None)
        kp_query, des_query = sift.detectAndCompute(img_query, None)

        if des_ref is None or des_query is None:
            print("Error: Not enough descriptors to proceed with detection.")
        else:
            bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            matches = bf.knnMatch(des_ref, des_query, k=2)

            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

            if len(good_matches) > 10:
                src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp_query[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                if M is not None:
                    h, w = img_ref.shape
                    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                    dst = cv2.perspectiveTransform(pts, M)

                    x, y, rw, rh = cv2.boundingRect(np.int32(dst))
                    cv2.rectangle(output, (x, y+10), (x + rw, y + rh), (0, 255, 0), 5)
                    cv2.putText(output, "Landmark: Pisa Tower", (x, y+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    print("Homography could not be computed.")
            else:
                print(f"Not enough good matches ({len(good_matches)}) to detect the Pisa Tower.")

    filename = os.path.basename(input_path)
    out_path = os.path.join(output_dir, f"FACE_LANDMARK_DETECTED_{filename}")
    cv2.imwrite(out_path, output)

    resized_output = resize_image_cv(output)
    cv2.imshow("Face + Landmark Detection", resized_output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#----------FEATURE 2----------
def feature2_classify_day_night(image_path, output_dir="output_directory"):
    from day_night_classification import TimeOfDayClassifier
    classifier = TimeOfDayClassifier()
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    image = resize_image_cv(image)
    result = classifier.classify_image(image, method='hsv_v_mean')
    
    label = f"Classified as: {result}"
    if result == "Day":
        cv2.rectangle(image, (10, 10), (170, 40), (0, 0, 0), -1)
        cv2.putText(image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    else:
        cv2.rectangle(image, (10, 10), (180, 40), (0, 0, 0), -1)
        cv2.putText(image, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)       
    
    filename = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"{result}_CLASSIFIED_{filename}")
    cv2.imwrite(out_path, image)

    cv2.imshow("Day/Night Classification", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#----------FEATURE 3----------
blur_threshold = 100.0
brightness_low = 80
brightness_high = 200
noise_threshold = 15

def feature3_quality_assessment(image_path, output_dir="output_directory"):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    image = resize_image_cv(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    annotated = image.copy()

    blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness_val = np.mean(gray)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_map = cv2.absdiff(gray, blurred)
    noise_val = np.std(noise_map)

    is_blurry = blur_val < blur_threshold
    lighting_issue = brightness_val < brightness_low or brightness_val > brightness_high
    is_noisy = noise_val > noise_threshold

    issues = []
    if is_blurry:
        issues.append("BLURRY")
    if lighting_issue:
        issues.append("LIGHTING ISSUE")
    if is_noisy:
        issues.append("NOISE DETECTED")

    if not issues:
        label = "GOOD PICTURE"
        color = (255, 0, 255)  # Purple
    else:
        label = " | ".join(issues)
        color = (0, 255, 255)  # Yellow for issues

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 3
    cv2.putText(annotated, label, (10, 40), font, font_scale, color, font_thickness)

    report_lines = [
        "= Image Quality Report =",
        f"Blur Variance: {round(blur_val, 2)}",
        f"Mean Brightness: {round(brightness_val, 2)}",
        f"Noise STD: {round(noise_val, 2)}",
        f"Is Blurry: {is_blurry}",
        f"Lighting Issue: {lighting_issue}",
        f"Noisy: {is_noisy}",
    ]
    start_y = h - 200
    for i, line in enumerate(report_lines):
        y = start_y + i * 30
        cv2.putText(annotated, line, (10, y), font, 0.4, (0,0,0), 1)

    #Save result to directory
    filename = os.path.basename(image_path)
    out_path = os.path.join(output_dir, f"QUALITY_ASSESSED_{filename}")
    cv2.imwrite(out_path, annotated)

    cv2.imshow("Image Quality Assessment", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#----------FEATURE 4----------
def feature5_find_similarity(input_path, output_dir="output_directory"):
    os.makedirs(os.path.join(output_dir, "similarity_output"), exist_ok=True)
    input_img = cv2.imread(input_path)
    if input_img is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    input_img = resize_image_cv(input_img)
    input_hist = get_histogram(input_img)
    similarities = []

    for filename in os.listdir(dataset):
        path = os.path.join(dataset, filename)
        img = cv2.imread(path)
        if img is None:
            continue
        hist = get_histogram(img)
        score = cv2.compareHist(input_hist, hist, cv2.HISTCMP_CORREL)
        similarities.append((filename, score, img))

    similarities.sort(key=lambda x: x[1], reverse=True)

    filename = os.path.basename(input_path)
    out_path = os.path.join(output_dir, f"similarity_output/Input_{filename}")
    cv2.imwrite(out_path, input_img)

    for i, (name, score, image) in enumerate(similarities[1:4]):
        resized_input = resize_image_cv(input_img)
        resized_match = resize_image_cv(image)
        out_path = os.path.join(output_dir, f"similarity_output/MATCH_{i}_{filename}")
        cv2.imwrite(out_path, resized_match)

        if len(resized_input.shape) == 2:
            resized_input = cv2.cvtColor(resized_input, cv2.COLOR_GRAY2BGR)
        if len(resized_match.shape) == 2:
            resized_match = cv2.cvtColor(resized_match, cv2.COLOR_GRAY2BGR)

        combined = cv2.hconcat([resized_input, resized_match])
        cv2.imshow(f"Similairity Score: {score:.2f}", combined)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#----------FEATURE 5----------
def get_user_text():
        text = askstring("Input", "Enter your text:")
        if text:
            return text
        return None

def get_user_color():
        color = askcolor(title="Choose Color")[0]
        if color:
            return tuple(int(c) for c in reversed(color))
        return None

def get_user_thickness():
        thick_int = int(askstring("Input", "Thickness:"))
        if thick_int:
            return thick_int
        return None

def get_user_scale():
        scale_int = float(askstring("Input", "Font Size:"))
        if scale_int:
            return scale_int
        return None

def feature6_annotate_img(image_path, output_dir="output_directory"):
    image = cv2.imread(image_path)
    if image is None:
        messagebox.showerror("Error", "Could not read image.")
        return
    image = resize_image_cv(image)
    drawing = False #tracks mouse moevement
    mode = 'rectangle'
    ix, iy = -1, -1 #initial coordinates of mouse click
    current_text = None

    def mouse_callback(event, x, y, flags, param):
        nonlocal drawing, ix, iy, image #nonlocal is used to access the variables outside the function

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp_img = image.copy()
            if mode == "rectangle":
                cv2.rectangle(temp_img, (ix, iy), (x, y), shape_color, thickness)
            elif mode == "circle":
                radius = int(((x-ix)**2 + (y-iy)**2)**0.5)
                cv2.circle(temp_img, (ix, iy), radius, shape_color, thickness)
            elif mode == "text":
                cv2.putText(temp_img, current_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, shape_color, thickness)
            cv2.imshow("Annotate Image", temp_img)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == "rectangle":
                cv2.rectangle(image, (ix, iy), (x, y), shape_color, thickness)
            elif mode == "circle":
                radius = int(((x-ix)**2 + (y-iy)**2)**0.5)
                cv2.circle(image, (ix, iy), radius, shape_color, thickness)
            elif mode == "text":
                cv2.putText(image, current_text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_size, shape_color, thickness)
            cv2.imshow("Annotate Image", image)

    cv2.namedWindow("Annotate Image")
    cv2.setMouseCallback("Annotate Image", mouse_callback)

    while True:
        cv2.imshow("Annotate Image", image)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('r'):
            mode = 'rectangle'
            shape_color = get_user_color()
            thickness = get_user_thickness()
        elif key == ord('c'):
            mode = 'circle'
            shape_color = get_user_color()
            thickness = get_user_thickness()
        elif key == ord('t'):
            mode = 'text'
            current_text = get_user_text()
            shape_color = get_user_color()
            font_size = get_user_scale()
            thickness = get_user_thickness()
        elif key == ord('s'):
            filename = os.path.basename(image_path)
            out_path = os.path.join(output_dir, f"ANNOTATED_{filename}")
            cv2.imwrite(out_path, image)
            messagebox.showinfo("Saved", f"Saved: {out_path}")
            break

    cv2.destroyAllWindows()