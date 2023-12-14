# Retfound Model API

This repository contains the Retfound model fine-tuned on the [IDRID dataset](https://www.kaggle.com/datasets/mariaherrerot/idrid-dataset/) and an API built with Flask to host and interact with the model.

## Install Environment

1. Create a Python environment with conda:

    ```bash
    conda create -n retfound python=3.7.5 -y
    conda activate retfound
    ```

2. Install dependencies:

    ```bash
    git clone https://github.com/krishnaadithya/retfound_api.git
    cd retfound_api
    pip install -r requirements.txt
    ```

## Steps to Use the API

### 1. Download Model File

- Find the model `.pth` file [here](https://drive.google.com/file/d/1uKW5ZjKdKar3ZGAXu6u7I1a8B5s-uxH8/view?usp=sharing).
- Download the file and place it in the `finetune_IDRID` folder in the root directory.

### 2. Running the API

- The API is built using Flask.
- Host the model by running:

    ```bash
    python retfound_api.py
    ```

### 3. Accessing the API

- For online tunneling, the current setup uses [ngrok](https://ngrok.com/).
- Currently, the model is hosted locally, and you can use the following endpoint to push your image: [https://623c-2001-8f8-166b-2a54-b5d7-fefa-f504-a420.ngrok-free.app/predict](https://623c-2001-8f8-166b-2a54-b5d7-fefa-f504-a420.ngrok-free.app/predict).
- Use the provided Python code to call the API:

    ```python
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
    url = "https://623c-2001-8f8-166b-2a54-b5d7-fefa-f504-a420.ngrok-free.app/predict"  # Replace this link with your endpoint
    response = requests.post(url, files=files)
    
    end_time = time.time()
    time_taken = end_time - start_time
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        result = response.json()
        print(f'Predicted category: {result["predictions"]}')
        print('Time taken for this inference:', time_taken)
    
    else:
        print(f'Error: {response.text}')
    ```

Please replace `your_image.jpg` with the path to your image file before running the code.
