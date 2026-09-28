# Vehicle Type Classification using ResNet-50

Student project demonstrating vehicle image classification with a pre-trained ResNet-50 ImageNet-1K model.

## Files
- `Vehicle_Type_Classification_ResNet50.ipynb` — complete notebook
- `app.py` — Streamlit web app
- `requirements.txt` — dependencies
- `car.jpg`, `bus.jpg`, `truck.jpg`, `motorcycle.jpg` — sample images

## Run
1. Use Python 3.10 or newer.
2. Install dependencies with `pip install -r requirements.txt`.
3. Start the web app with `streamlit run app.py`.
4. Select one of the included sample images or upload a JPG, PNG, or WEBP vehicle image.
5. On first launch, TorchVision downloads the pretrained weights; later launches reuse the local cache.

To run the notebook instead, open `Vehicle_Type_Classification_ResNet50.ipynb` in Jupyter, VS Code, or Google Colab and run its cells from top to bottom.

## Important technical note
Standard ImageNet ResNet-50 does not directly have custom classes named car/bus/truck/motorcycle. The app and notebook map ImageNet vehicle-related predictions into these groups; their mapped scores are not calibrated confidence values. Four sample images demonstrate the workflow but are not sufficient to establish general accuracy. A dedicated custom classifier should be fine-tuned and evaluated using a labeled vehicle dataset.

## Official model documentation
https://docs.pytorch.org/vision/stable/models/resnet.html
