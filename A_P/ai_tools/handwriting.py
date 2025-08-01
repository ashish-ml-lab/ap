# handwriting.py

from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

# Load the OCR model and processor
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

def convert_handwriting_image(image_file):
    """
    Converts handwritten image to text using TrOCR.
    """
    # Open image from uploaded file
    image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

    # Prepare image for the model
    pixel_values = processor(images=image, return_tensors="pt").pixel_values

    # Generate predicted text
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text
