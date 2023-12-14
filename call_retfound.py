import requests
from PIL import Image
import io
import time


# Replace 'your_image.jpg' with the path to your image file
image_path = 'data/test/e_proDR/IDRiD_061test.jpg'

# Open the image file
with open(image_path, 'rb') as f:
    image_data = f.read()

# Create a dictionary containing the image file
files = {'image': (image_path, image_data)}

start_time = time.time()

# Make a POST request to the API endpoint
url = "https://623c-2001-8f8-166b-2a54-b5d7-fefa-f504-a420.ngrok-free.app/predict" #'http://127.0.0.1:5002/predict'  
response = requests.post(url, files=files)

end_time = time.time()
time_taken = end_time - start_time

# Check if the request was successful (status code 200)
if response.status_code == 200:
    result = response.json()
    print(f'Predicted category: {result["predictions"]}')
    print('time taken for this inference:', time_taken)

else:
    print(f'Error: {response.text}')
