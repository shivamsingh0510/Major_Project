import cv2
import numpy as np
import pytesseract
import streamlit as st
import pandas as pd

# Set Tesseract command to default path in Google Colab
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract'

def process_frame(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny Edge Detection
    canny_edge = cv2.Canny(gray_image, 170, 200)

    # Find contours based on Edges
    contours, _ = cv2.findContours(canny_edge.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # Initialize license Plate contour and x,y,w,h coordinates
    contour_with_license_plate = None
    license_plate = None
    x, y, w, h = 0, 0, 0, 0

    # Find the contour with 4 potential corners and create ROI around it
    for contour in contours:
        # Find Perimeter of contour and it should be a closed contour
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
        if len(approx) == 4:  # see whether it is a Rect
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(contour)
            license_plate = gray_image[y:y + h, x:x + w]
            break

    if license_plate is not None:
        # Perform thresholding on the license plate region
        _, license_plate_thresh = cv2.threshold(license_plate, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Perform OCR (text recognition) on the license plate region
        text = pytesseract.image_to_string(license_plate_thresh)

        df = pd.read_csv('recognized_text.csv')
        allowed_cars = df['Recognized Text'].tolist()

        if text in allowed_cars:
            status = " vehicle  from inside (allow vehicle)"
        else:
            status = "vehicle from outside (No entry)"

        st.write(status)

        # Draw License Plate and write the Text
        image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
        image = cv2.putText(image, text, (x - 100, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                            cv2.LINE_AA)

        print("License Plate:", text)
        st.write(text);

    return image, text

def save_to_csv(text):
    # Check if the CSV file exists, if not, create it with headers
    try:
        df = pd.read_csv('recognized_text.csv')
    except FileNotFoundError:
        df = pd.DataFrame(columns=['Recognized Text'])

    # Create a new DataFrame with the new row
    new_row = pd.DataFrame({'Recognized Text': [text]})
    
    # Concatenate the new DataFrame with the existing DataFrame
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Save the DataFrame to CSV
    df.to_csv('recognized_text.csv', index=False)
    st.success("Text saved to recognized_text.csv")



def main():
    st.title("License Plate Detection and Recognition")

    # picture = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg', 'webp'])
    picture = st.camera_input("Clcik Image")

    if picture is not None:
        # Print the type and size of the uploaded image
        print("Uploaded image type:", type(picture))
        print("Uploaded image size:", picture.size)

        # Convert the uploaded image data to a numpy array
        image_array = np.array(bytearray(picture.read()), dtype=np.uint8)
        # Decode the numpy array into an OpenCV image
        frame = cv2.imdecode(image_array, 1)

        # Process the frame
        p_image, recognized_text = process_frame(frame)

        # Display the processed frame in the Streamlit app
        st.image(p_image, channels="BGR", use_column_width=True)

        if st.button("Save Text to CSV"):
            save_to_csv(recognized_text)

        csv_file_path = "recognized_text.csv"  # Replace "data.csv" with the actual file path

        # Read the CSV file
        df = pd.read_csv(csv_file_path)

        # Display the contents of the CSV file
        st.write(df)
    else:
        st.warning("Please upload an image.")

if __name__ == "__main__":
    main()

#streamlit run streamlit_app.py --server.enableXsrfProtection false