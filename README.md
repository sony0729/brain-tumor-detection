# Brain Tumor Detection (VGG16 + FastAPI)

```
Frontend (static HTML)  --POST /predict-->  FastAPI  -->  VGG16  -->  Tumor / No Tumor
```

## Results
VGG16 (ImageNet weights, frozen) + custom head, 253 MRI images, 10 epochs, 90/10 split:

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No Tumor | 0.83 | 1.00 | 0.91 |
| Tumor | 1.00 | 0.88 | 0.93 |

**Test accuracy: 92.3%** (26 held-out images). Small dataset, so treat the numbers as indicative.
A trained model is already included at `backend/model/brain_tumor_vgg16.h5`; step 1 is only needed to retrain.

## 1. Train the model (optional)
Download the dataset (folders `no/` and `yes/`) from Kaggle: *Brain MRI Images for Brain Tumor Detection*.
```bash
cd backend
pip install -r requirements.txt
python train.py --data ./brain_tumor_dataset     # saves model/brain_tumor_vgg16.h5
```

## 2. Run the API
```bash
cd backend
uvicorn main:app --reload        # http://localhost:8000/docs
```

## 3. Open the frontend
Open `frontend/index.html` in a browser (or `python -m http.server -d frontend 5500`).
Upload an MRI image and click **Analyze scan**.

## Deploy
- Frontend: GitHub Pages (Settings → Pages → `/frontend`) — set `API_DEFAULT` in `index.html` to your backend URL.
- Backend: Render / Railway (start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`).

> Educational project only — not a medical diagnosis.
