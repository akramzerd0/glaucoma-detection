# Glaucoma Detection Demo

This is a Gradio app for glaucoma detection from retinal fundus images.

## Classes
- **NRG** = Non-Referable Glaucoma
- **RG** = Referable Glaucoma

## Files to upload to your Hugging Face Space
- `app.py`
- `requirements.txt`
- your trained model file: `eyepacs_airogs_vgg16_best_threshold.h5`

## How prediction works
1. Upload an image
2. The app resizes it to `224x224`
3. It applies VGG16 preprocessing
4. It gets the model probability for **RG**
5. It applies the selected threshold (`0.45`)
6. It returns **NRG** or **RG**

## Important
If your final notebook gave you a different best threshold, update the `THRESHOLD` value in `app.py`.