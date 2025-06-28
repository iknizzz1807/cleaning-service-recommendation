# cleaning-service-reccomendation
An AI-powered web application that analyzes an image of a room, detects various types of messes, and recommends appropriate cleaning services. The system features a customizable mapping between detected messes and the services offered.

## About The Project

This project uses a multi-stage AI pipeline to provide intelligent cleaning recommendations. A user uploads an image, and the backend processes it to identify and classify messes. The frontend then displays the annotated image along with a ranked list of suggested services, which can be configured by the user.

### Key Features

*   **AI-Powered Mess Detection:** Utilizes a custom pipeline:
    1.  **YOLO Detector:** Identifies potential messy areas in the image.
    2.  **VGG16 Filter:** Classifies if a detected area is genuinely a "mess" to reduce false positives.
    3.  **EfficientNet Classifier:** Categorizes the specific type of mess (e.g., `messy_bed`, `dirty_floor`).
*   **Customizable Service Mapping:** A user-friendly settings panel allows you to define services (e.g., "Deep Clean," "Laundry") and assign which mess types they correspond to.
*   **Ranked Recommendations:** Suggested services are returned with a confidence score and sorted by relevance.
*   **Visual Feedback:** The original image is returned with bounding boxes drawn around the detected messes.
*   **Modern Web Interface:** A clean, tab-based UI for uploading images and configuring settings.

### Tech Stack

*   **Backend:** Python, FastAPI, PyTorch, Ultralytics (YOLO), `timm`
*   **Frontend:** Vanilla JavaScript, HTML5, CSS3
*   **AI Models:** YOLO (Detection), VGG16 (Filtering), EfficientNet (Classification)

---

## Getting Started

Follow these instructions to set up and run the project locally.

### Prerequisites

*   Python 3.8+ and `pip`
*   Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2.  **Set up the Backend:**
    *   Navigate to the backend directory:
        ```bash
        cd backend
        ```
    *   Install the required Python packages:
        ```bash
        pip install fastapi uvicorn python-multipart Pillow torch torchvision ultralytics timm albumentations numpy
        ```
    *   Ensure the model weights are present in the `backend/weights/` directory.

3.  **Set up the Frontend:**
    *   The frontend consists of static files and requires a simple web server to run correctly and avoid CORS issues.

---

## Usage

1.  **Start the Backend Server:**
    *   In your terminal, from the `backend` directory, run:
        ```bash
        python app.py
        ```
    *   The AI server will be running at `http://localhost:8000`.

2.  **Start the Frontend Server:**
    *   Open a **new terminal window** and navigate to the `frontend` directory:
        ```bash
        cd frontend
        ```
    *   Start a simple Python web server on a different port (e.g., 8080):
        ```bash
        python -m http.server 8080
        ```

3.  **Use the Application:**
    *   Open your web browser and go to `http://localhost:8080`.
    *   **(Optional)** Click the **Settings** tab. Here you can create new services and assign them to different mess types.
    *   Click the **Uploader** tab.
    *   Choose an image file and click **Upload**.
    *   The results will appear below, showing the suggested services and the processed image.

## Screenshots of the app 
![image](https://github.com/user-attachments/assets/73f81ae4-e5cf-4a62-8b4e-7e67fb67fc78)
![image](https://github.com/user-attachments/assets/4597e25a-2c30-4e21-b88a-8d30797be880)
- Request for training weight: mythonggg@gmail.com
