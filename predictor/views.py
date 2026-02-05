from django.shortcuts import render
import joblib
import numpy as np

# Load the model and encoders
model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
type_encoder = joblib.load('type_encoder.pkl')
rating_encoder = joblib.load('rating_encoder.pkl')
country_encoder = joblib.load('country_encoder.pkl')

def home(request):
    if request.method == 'POST':
        release_year = int(request.POST['release_year'])
        type_ = request.POST['type']
        rating = request.POST['rating']
        country = request.POST['country']

        # Encode the inputs
        type_encoded = type_encoder.transform([type_])[0]
        rating_encoded = rating_encoder.transform([rating])[0]
        country_encoded = country_encoder.transform([country.title()])[0]

        # Prepare the input array
        input_data = np.array([[release_year, type_encoded, rating_encoded, country_encoded]])

        # Scale the input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)[0]

        return render(request, 'result.html', {'prediction': round(prediction)})

    return render(request, 'form.html')
