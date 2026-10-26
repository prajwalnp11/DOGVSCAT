# 🐶 Dog vs Cat Image Classifier

A machine learning web application that classifies an uploaded image as either a **Dog** or a **Cat** using a trained deep learning model.

## 🚀 Features

- Upload an image for classification.
- Display the uploaded image on the results page.
- Predict whether the image contains a dog or a cat.
- Show a prediction score.
- Simple web interface built with Flask.

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- Flask
- Pillow
- HTML and CSS
- Git and GitHub

## 📁 Project Structure

```text
DOGVSCAT/
├── app.py
├── dog_cat_classifier.h5
├── requirements.txt
├── .gitignore
├── README.md
└── templates/
    └── index.html
```

## ⚙️ Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/prajwalnp11/DOGVSCAT.git
cd DOGVSCAT
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
```

### 3. Activate the environment

**macOS / Linux:**

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Start the application

```bash
python app.py
```

Open this URL in your browser:

http://127.0.0.1:5000

## 🧠 Model

The application loads the trained model from `dog_cat_classifier.h5` and uses it to generate predictions for uploaded images.

**Note:** The model's original class mapping and preprocessing configuration should be verified against its training code before relying on predictions.

## 🔮 Future Improvements

- Deploy the application publicly.
- Improve prediction reliability and error handling.
- Add automated testing.
- Explore containerization with Docker.

## 👨‍💻 Author

**Prajwal N P**

GitHub: https://github.com/prajwalnp11

---

*This project is built for learning and demonstrating machine learning deployment with Flask.*