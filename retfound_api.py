from flask import Flask, request, jsonify
import torch
import models_vit
from util.pos_embed import interpolate_pos_embed
from PIL import Image
import io
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import logging

# Configurable Parameters
MODEL_CHECKPOINT_PATH = 'finetune_IDRiD/checkpoint-best.pth'
NUM_CLASSES = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__)

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO)

# Load the model
model = models_vit.__dict__['vit_large_patch16'](
    num_classes=NUM_CLASSES,
    drop_path_rate=0.2,
    global_pool=True,
)

print('loading model....')
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location='cpu')
checkpoint_model = checkpoint['model']
state_dict = model.state_dict()

# Remove unnecessary keys
for k in ['head.weight', 'head.bias']:
    if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
        logging.warning(f"Removing key {k} from pretrained checkpoint")
        del checkpoint_model[k]

interpolate_pos_embed(model, checkpoint_model)

msg = model.load_state_dict(checkpoint_model, strict=False)

model.to(device)
model.eval()

@app.errorhandler(Exception)
def handle_error(e):
    logging.error(f"Error: {str(e)}")
    return jsonify({'error': 'Internal Server Error'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input validation
        if 'image' not in request.files:
            raise ValueError("No 'image' file in the request.")

        image_data = request.files['image'].read()

        input_data = preprocess_image(image_data)
        input_data = input_data.to(device)

        with torch.no_grad():
            output = model(input_data)

        predictions = postprocess_output(output)

        return jsonify({'predictions': predictions})

    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return jsonify({'error': 'Internal Server Error'}), 500

def build_eval_transform(input_size=224):
    t = [
        transforms.Resize(input_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ]
    return transforms.Compose(t)


def preprocess_image(image_data):
    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    transform = build_eval_transform()
    input_data = transform(image)
    input_data = input_data.unsqueeze(0)  # Add a batch dimension
    return input_data

def postprocess_output(output):
    _, predicted_class = torch.max(output, 1)
    categories = ['noDR', 'mildDR', 'moderateDR', 'severeDR', 'proDR']
    predicted_category = categories[predicted_class.item()]
    
    return predicted_category


if __name__ == '__main__':
    #from waitress import serve
    #serve(app, host='0.0.0.0', port=5002)

    app.run(debug=False, port=5002)

