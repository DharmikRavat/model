# AI Urban Farming Assistant

Plant disease screening, visual explanations, watering suggestions, and a small
plant-care dashboard. The disease classifier supports tomato, potato, and bell
pepper classes. It is a screening aid, not a substitute for agricultural advice.

## What's included

- EfficientNet-B2 disease classifier and two-stage GPU training pipeline
- PlantVillage-style training images and a separate PlantDoc real-world split
- Image quality checks, Grad-CAM overlays, and unsupported-plant screening
- XGBoost watering models with optional Open-Meteo weather
- FastAPI backend and React dashboard
- Classification reports, per-class metrics, and confusion matrices

The current checkout contains the PlantDoc files successfully retrieved by the
download script. Some upstream links returned 404 during the download; see
`dataset_plantdoc/manifest.json` for the exact local contents and
`download_plantdoc.py` to retry against the source.

## Clone and fetch large files

Install [Git](https://git-scm.com/downloads) and
[Git LFS](https://git-lfs.com/), then:

```bash
git lfs install
git clone https://github.com/DharmikRavat/model.git
cd model
git lfs pull
```

Image datasets and model weight files use Git LFS. The raw datasets are
redistributed with the owner's permission. PlantDoc is distributed under
[Creative Commons Attribution 4.0](https://creativecommons.org/licenses/by/4.0/);
its upstream license and citation are included alongside the dataset.

## Python environment

Python 3.12 is recommended. For an NVIDIA GPU, install a PyTorch build matching
your installed CUDA driver first. For CUDA 12.6:

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
```

Confirm GPU availability:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

TensorFlow is used by the legacy EfficientNet-B0 pipeline. TensorFlow GPU
training is not supported by standard native-Windows TensorFlow builds; use
WSL2/Linux or an appropriate TensorFlow GPU setup for that pipeline. The
EfficientNet-B2 PyTorch pipeline is the GPU training path.

## Start the application

Terminal 1, from the project root:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8000
```

Terminal 2:

```bash
cd dashboard
npm ci
npm run dev
```

Open the Vite URL printed in the terminal. The dashboard talks to
`http://127.0.0.1:8000` by default. Set `VITE_API_URL` at dashboard build/run
time when the backend uses a different URL.

API routes include:

- `POST /api/diagnose` and `POST /predict` — multipart image diagnosis, care notes,
  and optional Grad-CAM
- `POST /api/gradcam` and `POST /gradcam` — Grad-CAM overlay
- `POST /api/watering` and `POST /watering` — watering estimate and weather
- `GET|POST /api/plants` and `GET|POST /plants` — plant tracker
- `GET /api/health`, `GET /api/knowledge`, and `GET /api/weather?city=Pune`

Interactive API docs: `http://127.0.0.1:8000/docs`.

## Train and evaluate EfficientNet-B2

The default training run uses the available PlantVillage training subset,
PlantDoc training images, class weights, augmentation, AdamW, cosine scheduling,
and a frozen-head warm-up followed by full fine-tuning. The held-out PlantDoc
test split is not added to training:

```bash
python train_b2.py
python test_unknown.py
```

On Windows, if DataLoader worker processes exceed available memory, run:

```powershell
$env:LOADER_WORKERS = "0"
python train_b2.py
Remove-Item Env:LOADER_WORKERS
```

Training writes `models/plant_disease_b2.pt`, configuration, and reports in
`outputs/`. The current checked-in metrics are prior-run measurements; retrain
and evaluate the exact checkpoint before using those numbers to make deployment
claims. PlantDoc is small and imbalanced, so real-world performance can be
substantially lower than PlantVillage test performance.

## Dataset citations

### PlantDoc

Davinder Singh, Naman Jain, Pranjali Jain, Pratik Kayal, Sudhakar Kumawat, and
Nipun Batra. “PlantDoc: A Dataset for Visual Plant Disease Detection.” ACM
CoDS-COMAD 2020. [Paper](https://arxiv.org/abs/1911.10317) ·
[Source dataset](https://github.com/pratikkayal/PlantDoc-Dataset).

### PlantVillage

Sharada P. Mohanty, David P. Hughes, and Marcel Salathé. “Using Deep Learning
for Image-Based Plant Disease Detection.” *Frontiers in Plant Science*, 2016.
[DOI: 10.3389/fpls.2016.01419](https://doi.org/10.3389/fpls.2016.01419) ·
[Source dataset](https://github.com/spMohanty/PlantVillage-Dataset).

## Limitations

- A high-confidence prediction is not proof of disease; inspect the leaf and
  local conditions before acting.
- Grad-CAM is a model-attention visualization, not a lesion segmentation mask.
- The watering model uses synthetic agronomic training data and estimates when
  a plant may need water; check actual soil moisture before watering.
- Live weather uses the free Open-Meteo service and is optional.
- Disease severity is not predicted because the training labels do not include
  severity annotations.
