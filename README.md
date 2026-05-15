# flipPhone-training

Trainiert ML-Modelle zur Erkennung von Skateboard-Tricks anhand von Beschleunigungs- und Gyroskopdaten. Stellt das aktive Modell als REST-API bereit.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# .env mit echten Werten befüllen
```

---

## Environment variables

| Variable | Beschreibung |
|---|---|
| `FLIPPHONE_URL` | URL des Hauptservers (z.B. `https://flipphone.tech`) |
| `FLIPPHONE_API_KEY` | Admin API-Key zum Abrufen des Datasets |

---

## Modelle trainieren

Training wird über die REST-API gesteuert — entweder über das Admin-Panel unter `/admin/models` oder direkt per HTTP.

### Training starten

```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "random_forest", "tricks": ["Kickflip", "Heelflip", "FS Shuvit"]}'
```

`model_type`: `random_forest` | `svm` | `mlp`

`tricks`: Liste der zu trainierenden Tricks. Leer lassen für alle verfügbaren Tricks.

`source_users`: Optional — nur Aufnahmen bestimmter Nutzer verwenden.

### Status prüfen

```bash
curl http://localhost:8000/train/{job_id}
```

### Modell aktivieren

```bash
curl -X POST http://localhost:8000/models/{run_id}/activate
```

Nach der Aktivierung wird das Modell sofort für `/api/predict` genutzt und bleibt beim nächsten Start aktiv.

---

## Daten manuell abrufen

```bash
python fetch_data.py --url https://flipphone.tech --key fp_yourAdminKey
```

Speichert das Dataset als `data/dataset.csv`. Der Trainingsserver ruft die Daten automatisch ab wenn `FLIPPHONE_URL` und `FLIPPHONE_API_KEY` gesetzt sind.

---

## API starten

```bash
python server.py          # Port 8000
PORT=9000 python server.py
```

### Endpunkte

| Methode | Pfad | Beschreibung |
|---|---|---|
| `POST` | `/train` | Training starten |
| `GET` | `/train/{id}` | Trainingsstatus abfragen |
| `GET` | `/models` | Alle Trainingsläufe auflisten |
| `GET` | `/models/{id}` | Details zu einem Lauf |
| `GET` | `/models/{id}/metrics` | Metriken (Accuracy, F1, per-class) |
| `POST` | `/models/{id}/activate` | Modell aktivieren |
| `POST` | `/api/predict` | Trick vorhersagen |
| `POST` | `/batch_embed` | UMAP-Embeddings für Aufnahmen berechnen |
| `GET` | `/tricks` | Verfügbare Tricks abrufen |

### `POST /api/predict`

```json
{
  "samples": [
    {"t": 0, "ax": 1.2, "ay": -0.5, "az": 9.8, "gx": 0.1, "gy": -0.3, "gz": 0.02},
    {"t": 50, "ax": 1.5, "ay": -0.8, "az": 9.5, "gx": 0.5, "gy": -1.2, "gz": 0.1}
  ]
}
```

```json
{
  "trick": "Kickflip",
  "confidence": 0.92,
  "probabilities": {"Kickflip": 0.92, "Heelflip": 0.05, "FS Shuvit": 0.03}
}
```

---

## Projektstruktur

```
server.py           ← FastAPI Prediction + Training API
train.py            ← Feature-Extraktion & Modell-Training
train_nn.py         ← Neuronales Netz Training (experimentell)
augment.py          ← Datenaugmentierung
fetch_data.py       ← Dataset vom Hauptserver abrufen
inspect_outliers.py ← Outlier-Analyse
requirements.txt
data/               ← CSV-Daten (gitignored)
models/             ← Trainierte Modelle (gitignored)
models.db           ← Trainingshistorie (gitignored)
.env.example
.github/workflows/  ← Auto-Deploy auf Hetzner
```

---

## Deploy

Push auf `main` deployt automatisch per GitHub Actions:

1. SSH auf Hetzner: `git fetch` + `git reset --hard` → `pip install`
2. Neustart des `flipphone-training` systemd-Service
