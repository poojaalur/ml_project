# Netflix Movie Duration Predictor
<img width="1354" height="750" alt="image" src="https://github.com/user-attachments/assets/79016f3e-df71-46c6-80a0-172b53873595" />
<img width="1356" height="762" alt="image" src="https://github.com/user-attachments/assets/b17341f2-d127-40c4-84fb-6c66ad0de09b" />


## Description
This project is a machine learning application that predicts the duration of Netflix movies based on various features such as release year, type, rating, and country. The model is trained using Linear Regression on a dataset of Netflix titles, specifically filtered for movies with valid duration information.

The application includes a Django web interface that allows users to input movie features and receive duration predictions in minutes.

## Features
- Predict movie duration based on release year, type, rating, and country
- Web-based user interface built with Django
- Pre-trained machine learning model using Linear Regression
- Data preprocessing including label encoding and feature scaling
- Easy-to-use form for inputting movie details

## Installation

### Prerequisites
- Python 3.7+
- pip (Python package manager)

### Steps
1. Clone the repository:
   ```
   git clone <repository-url>
   cd netflix
   ```

2. Install the required dependencies:
   ```
   pip install django scikit-learn pandas numpy joblib
   ```

3. Run the model training script:
   ```
   python train_model.py
   ```
   This will generate the necessary model and encoder files (`model.pkl`, `scaler.pkl`, `type_encoder.pkl`, `rating_encoder.pkl`, `country_encoder.pkl`).

4. Run the Django development server:
   ```
   python manage.py runserver
   ```

5. Open your web browser and navigate to `http://127.0.0.1:8000/` to access the application.

## Usage
1. Start the Django server as described in the installation steps.
2. On the web interface, fill in the form with the following movie details:
   - Release Year
   - Type (Movie)
   - Rating (e.g., TV-MA, PG-13)
   - Country
3. Click the "Predict" button to get the predicted duration in minutes.
4. The result will be displayed on a new page showing the predicted duration.

## Model Details
- **Algorithm**: Linear Regression
- **Features**:
  - Release Year (numerical)
  - Type (categorical, encoded)
  - Rating (categorical, encoded)
  - Country (categorical, encoded)
- **Preprocessing**:
  - Label encoding for categorical variables
  - Standard scaling for numerical features
- **Training Data**: Netflix titles dataset, filtered for movies with valid duration information
- **Target Variable**: Movie duration in minutes

The model is trained on 80% of the data and tested on 20%. The trained model and preprocessing objects are saved using joblib for use in the Django application.

## Technologies Used
- **Python**: Core programming language
- **Django**: Web framework for the user interface
- **scikit-learn**: Machine learning library for model training and preprocessing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **joblib**: Model serialization

## Project Structure
```
netflix/
├── manage.py
├── netflix_ml_project/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── predictor/
│   ├── views.py
│   ├── urls.py
│   ├── templates/
│   │   ├── form.html
│   │   └── result.html
│   ├── static/
│   │   └── style.css
│   └── ...
├── train_model.py
├── model.pkl
├── scaler.pkl
├── type_encoder.pkl
├── rating_encoder.pkl
├── country_encoder.pkl
├── netflix_titles.csv
└── README.md
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments
- Netflix for providing the dataset
- scikit-learn and Django communities for their excellent libraries

