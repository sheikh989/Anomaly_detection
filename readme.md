# **Anomaly Detection in Videos**

This project provides a system for detecting anomalies in video streams using a pretrained C3D model. It can be run as a direct Python script or as a local web application.

## **Prerequisites**

Before you begin, ensure you have the following installed on your system:

* [Git](https://git-scm.com/)  
* [Python 3.x](https://www.python.org/)  
* pip (Python package installer)

## **Installation & Setup**

Follow these steps to set up the project environment.

### **1\. Clone the Repository**

First, clone this repository to your local machine and navigate into the project directory:

git clone https://github.com/sheikh989/Anomaly_detection.git
\cd Anomaly_detection

### **2\. Download the Pretrained Model**

Download the C3D model required for the project.

* [**Download C3D Model Here**](https://drive.google.com/file/d/13FWcvSMNTEHSk1MRZ4qBxZBwzTsVtyKC/view?usp=sharing)

Once downloaded, place the model file inside the pretrained/ directory.

### **3\. Download Demo Videos**

Download the demo videos to test the application.

* [**Download Demo Videos Here**](https://drive.google.com/file/d/1lPxePkRh7yywVePPui3cisWj5Wgf9yly/view?usp=sharing)

After downloading, unzip the folder and move its contents (the video files) into the main root directory of this project (the same folder as app.py).

### **4\. Create and Activate a Virtual Environment**

It is highly recommended to use a virtual environment to manage dependencies and avoid conflicts.

**Create the environment:**

python \-m venv venv

**Activate the environment:**

* **On Windows (Command Prompt):**  
  .\\venv\\Scripts\\activate

* **On Windows (PowerShell):**  
  .\\venv\\Scripts\\Activate.ps1

* **On macOS/Linux:**  
  source venv/bin/activate

### **5\. Install Dependencies**

With your virtual environment active, install all required packages using the requirements.txt file:

pip install \-r requirements.txt

## **Usage**

You can run the project in two different ways.

### **Option 1: Run the Pipeline Demo**

This method runs the detection pipeline directly in your terminal and will prompt you to select a video.

1. Ensure your virtual environment is active.  
2. Run the pipeline\_demo.py script from your terminal:  
   python pipeline\_demo.py

3. A file dialog will open. Select one of the demo videos you downloaded to begin processing.

### **Option 2: Run the Web Application**

This method launches a local web server (using Flask) to provide a web interface for the tool.

1. Ensure your virtual environment is active.  
2. Run the app.py script from your terminal:  
   python app.py

3. Open your web browser and navigate to the local host address provided in the terminal. It will typically be:  
   http://127.0.0.1:5000
