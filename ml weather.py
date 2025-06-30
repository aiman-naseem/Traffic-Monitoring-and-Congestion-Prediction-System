import requests

# Your API key from OpenWeather
API_KEY = "2c2d8adc5b791c9bcb23cb33ce7c315e"
CITY = "New York"

# API Endpoint
url = f"http://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}&units=metric"

# Make the request
response = requests.get(url)

# Convert the response to JSON
weather_data = response.json()

# Extract useful fields
temperature = weather_data['main']['temp']
humidity = weather_data['main']['humidity']
weather_condition = weather_data['weather'][0]['main']  # e.g., Clear, Rain, Snow

# Print output
print(f"Temperature: {temperature}°C")
print(f"Humidity: {humidity}%")
print(f"Condition: {weather_condition}")

def encode_weather_condition(condition):
    mapping = {
        "Clear": 0,
        "Clouds": 1,
        "Rain": 2,
        "Snow": 3,
        "Thunderstorm": 4,
        "Drizzle": 5,
        "Mist": 6,
        "Fog": 7
    }
    return mapping.get(condition, -1)  # -1 is used if condition not found

weather_code = encode_weather_condition(weather_condition)

import numpy as np

# Example values from your API results
temperature = 21.5        # from OpenWeatherMap
humidity = 55             # from OpenWeatherMap
weather_code = 0          # after encoding "Clear"

# Create feature vector (shape: 1 sample x 4 features)
input_features = np.array([[temperature, humidity, weather_code]])

import joblib  # or pickle if you used that

# Load the trained model (replace with your actual filename)
model = joblib.load("traffic_volume_scaler.pkl")  # or .sav if you used pickle

# Predict
prediction = model.predict(input_features)

# Output
print(f"🚦 Predicted Traffic Congestion Level: {prediction[0]}")

