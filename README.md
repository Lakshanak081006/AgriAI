# AgriAI - Agricultural Disease Detection System

AgriAI is a comprehensive web-based application built with Django that leverages Machine Learning and Image Processing to accurately detect crop diseases. By analyzing uploaded images of plant leaves, it identifies diseases and provides actionable insights including detailed descriptions, prevention techniques, and treatment methods to assist farmers and agricultural professionals in managing crop health effectively.

## 🌟 Features

- **Disease Identification**: Deep learning models specialized in identifying various plant diseases based on leaf imagery.
- **Actionable Recommendations**: Instant feedback including:
  - Disease Description
  - Prevention Strategies
  - Recommended Treatments
- **User-Friendly Interface**: An intuitive and visually pleasing UI allowing fast and easy image uploading and analysis.
- **Reporting & Documentation**: Access to comprehensive workflows and reports.

## 🛠️ Technology Stack

- **Backend Framework**: Django (Python)
- **Frontend**: HTML5, CSS3 
- **Machine Learning**: Deep Learning based classification (PyTorch/TensorFlow)
- **Image Processing**: OpenCV, Pillow

## 🚀 Getting Started

Follow these steps to set up the AgriAI project locally on your machine.

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Lakshanak081006/AgriAI.git
   cd AgriAI
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment:**
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Run Database Migrations:**
   ```bash
   python manage.py migrate
   ```

6. **Start the Development Server:**
   ```bash
   python manage.py runserver
   ```

7. **Access the Application:**
   Open your browser and navigate to `http://127.0.0.1:8000/`

## 📁 Project Structure

```text
AgriAI/
├── agritai/                 # Main Django project settings
├── prediction/              # Core application for ML predictions
│   ├── templates/           # HTML user pages (upload, result, base layouts)
│   ├── views.py             # Logic for image handling and predictions
│   ├── urls.py              # App-level routing
│   └── models.py            # Database models
├── requirements.txt         # Project pip dependencies
├── image_processing_flow.md # Documentation of the ML flow
└── manage.py                # Django CLI utility
```


## 📜 License

This project is open-source and available under the terms of the MIT License.
