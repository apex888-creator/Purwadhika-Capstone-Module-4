# Purwa YOLO: Electronic Road Pricing (ERP) Simulation with Computer Vision

A Streamlit web app that simulates an **Electronic Road Pricing (ERP)** system for major roads. It detects and counts vehicles (car, van, bus) from images or videos using **YOLO26s**, reads licence plates using **fast-alpr**, and then calculates the ERP charge, including an **odd-even** plate rule.

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [How It Works](#how-it-works)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the App](#running-the-app)
- [ERP Pricing Logic](#erp-pricing-logic)
- [Configuration](#configuration)
- [Testing](#testing)
- [Limitations](#limitations)
- [References](#references)

---

## Background

### Congestion is an economic problem, not just a waste of time

Traffic congestion in large cities carries real economic costs: lost productive time, higher fuel consumption, delayed freight, and worse air quality. In Jakarta, congestion losses are estimated at **around IDR 100 trillion per year** (a 2018 study by the Coordinating Ministry for Economic Affairs, cited by the Jakarta Transportation Office). A World Bank estimate cited in the academic literature puts the figure at about **USD 5.2 billion per year, roughly 5.4% of the city's GDP**.

Building more roads is not enough on its own. Added capacity tends to trigger *induced demand*: wider roads attract more vehicles, and congestion returns. A widely recommended alternative is **Transportation Demand Management (TDM)**, one form of which is putting a price on road space at the times and places where it is scarcest.

### Why ERP

Economically, congestion is an *externality*: every driver who enters a crowded road adds delay for everyone else but does not bear that cost. Road pricing internalises the cost, so road users reconsider whether to make a trip, change their route, or shift the time. ERP lets this mechanism run automatically, without toll booths, and lets the charge vary by vehicle type, time of day, and congestion level.

The Jakarta provincial government has already included ERP in its plan to limit private vehicles, alongside a low emission zone (LEZ), parking management, and limits on vehicle age and number.

### Evidence from other cities

| City (start year) | Measured results | Source |
|---|---|---|
| **Singapore**, ALS (1975) | Traffic volume in the restricted zone fell by about **44%** (other sources report morning traffic falling by up to ~76%, depending on metric and period) | Phang & Toh (2004); FHWA summary |
| **Singapore**, electronic ERP (1998) | Traffic fell by about **13%** and vehicle speeds rose by about **22%** | FHWA (Office of Operations) |
| **London** (2003) | Vehicles entering the zone fell **18%**; congestion inside the zone fell about **30%**; delays fell about 25% and speeds rose about 30% (after the first year) | Leape (2006); USDOT/ITS report (2010) |
| **Stockholm** (trial 2006, permanent 2007) | Traffic across the cordon fell about **22%** (the original target was only 10-15%); stayed in the 18-20% range through 2011; CO₂ in the inner city fell about **14%** | Eliasson et al. (2009); Börjesson et al. (2012) |

**Summary:** there is no single "ERP success rate". Results depend on tariff design, coverage area, and the availability of public transport. Still, in the three reference cities congestion pricing consistently cut traffic in the charged area by roughly **13-30%** and reduced congestion substantially, with effects that lasted for years. Public support in Stockholm also rose from about one third of residents before the trial to around 70% in 2011.

### Where this project fits

A full-scale ERP system needs vehicle detection and classification, licence-plate reading, and automated charge calculation. This project is a **prototype** that shows that pipeline end to end with open-source models running on CPU, and also experiments with an odd-even rule as an additional pricing layer.

---

## Features

- **Vehicle detection and counting** from images (`jpg`, `jpeg`, `png`, `webp`, max 10 MB) or videos (`mp4`, `mov`, `avi`, `mkv`, max 100 MB).
- **Video tracking**, so each vehicle is counted once. A vehicle is counted if it is tracked for at least 3 frames.
- **3 vehicle classes:** `car`, `van`, `bus`.
- **Licence-plate OCR** from the best crops of each vehicle.
- **Odd-even rule:** an extra charge based on the parity of the plate number and the date (WIB time zone).
- **Dynamic pricing:** base rate multiplied by a period factor (Normal / Peak / Off-peak) and a congestion level factor (Free-flowing / Busy / Congested).
- **Multi-currency:** USD, IDR, CNY.
- **Privacy:** plates are masked by default (for example `B ***4 ***`).
- Multi-language UI support (`ui/i18n.py`).
- Ready to run with **Docker** or on **Streamlit Cloud**.

## How It Works

```
Image / Video
      │
      ▼
[1] Vehicle detection (YOLO26s) + tracking (supervision)
      │   → counts per class, congestion level, crops of each vehicle
      ▼
[2] Plate detection (YOLO12n) + OCR (fast-alpr)
      │   → crops of each plate, plate text + confidence per vehicle
      ▼
[3] ERP charge calculation
      │   → base rate × period × congestion × odd-even rule
      ▼
Charge table, summary, and annotated media
```

## Tech Stack

| Component | Technology |
|---|---|
| UI | [Streamlit](https://streamlit.io/) |
| Vehicle detection | [Ultralytics](https://docs.ultralytics.com/) **YOLO26s** (trained on a 3-class dataset: car, van, bus) |
| Tracking | [supervision](https://github.com/roboflow/supervision) |
| Plate detection + OCR | [fast-alpr](https://github.com/ankandrew/fast-alpr) (plate detector + [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr), ONNX/CPU runtime) |
| Image/video processing | OpenCV, Pillow, imageio + imageio-ffmpeg (H.264 output) |
| Deep learning runtime | PyTorch |
| Dependency management | Poetry |
| Deployment | Docker / Docker Compose, Streamlit Cloud |

**Models used**

| Purpose | Model | Location |
|---|---|---|
| Vehicle detection | YOLO26s (fine-tuned, classes: `car`, `van`, `bus`) | `models/best.pt` |
| Plate detector (optional, custom) | YOLOv12n, single class | `models/Detect_Plate_Model.pt` |
| Plate detector (default) | `yolo-v9-t-384-license-plate-end2end` | downloaded automatically by fast-alpr |
| Plate OCR | `cct-s-v2-global-model` | downloaded automatically by fast-plate-ocr |

If `Detect_Plate_Model.pt` exists, it replaces fast-alpr's built-in plate detector; the plate text is still read by fast-plate-ocr.

The training code for the vehicle detection model is in `training_code/vehicle-detection-yolo26s.ipynb`.

## Project Structure

```
.
├── streamlit_app.py            # app entry point
├── pyproject.toml              # dependencies (Poetry), the single source of truth
├── poetry.lock
├── requirements.txt            # dependency list for Streamlit Cloud
├── packages.txt                # system packages for Streamlit Cloud
├── dockerfile
├── docker-compose.yml
├── .streamlit/config.toml      # Streamlit configuration
├── models/
│   ├── best.pt                 # YOLO vehicle model
│   └── Detect_Plate_Model.pt   # custom plate detector (optional)
├── src/purwa_yolo/
│   ├── main.py                 # main app flow
│   ├── app_config.py           # constants & configuration
│   ├── device.py               # device selection (CPU/GPU)
│   ├── core/                   # pure logic (no Streamlit/OpenCV)
│   │   ├── data.py             #   dataclasses for detection & OCR results
│   │   ├── plate_rules.py      #   plate format & odd-even rule
│   │   └── pricing.py          #   ERP charge calculation
│   ├── vision/                 # computer vision layer
│   │   ├── detector.py         #   YOLO detection
│   │   ├── crops.py            #   vehicle crop extraction
│   │   ├── pipelines.py        #   image & video pipelines
│   │   └── plate_ocr.py        #   plate detection + OCR
│   └── ui/                     # UI components
│       ├── gate.py, i18n.py, results.py, sidebar.py, tables.py, theme.py
├── tests/                      # unit tests (pytest)
└── training_code/              # model training notebook
```

## Installation

### Prerequisites

- Python **3.11 - 3.13**
- [Poetry](https://python-poetry.org/) (or pip)
- (Optional) Docker & Docker Compose

### Option A: Poetry (recommended)

```bash
git clone <this-repository-url>
cd <folder-name>
python -m pip install poetry
python -m poetry install
```

### Option B: pip

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

### Option C: Docker

```bash
docker compose up --build
```

### Model files

Make sure the model files are present in the `models/` folder:

- `best.pt` (vehicle detection model)
- `Detect_Plate_Model.pt` (optional)

## Running the App

**Poetry**

```bash
python -m poetry run streamlit run streamlit_app.py
```

**pip / venv**

```bash
streamlit run streamlit_app.py
```

**Docker Compose**

```bash
docker compose up
```

Open `http://localhost:8888`. If the port is already in use, change the left-hand number in `docker-compose.yml`, for example `"8501:8888"`.

Usage flow:

1. Upload an image or video and run **vehicle detection**.
2. Run **plate reading (OCR)**.
3. Set the rates, period, currency, and odd-even date in the sidebar, then view the **ERP charge breakdown**.

## ERP Pricing Logic

```
charge per vehicle = base rate × min(period factor × congestion factor, max multiplier) × odd-even factor
```

**Base rates (default, USD per vehicle)**

| Class | Rate |
|---|---|
| Car | 2.0 |
| Van | 3.0 |
| Bus | 5.0 |

**Period factor (default):** Normal ×1.0 · Peak hours ×1.5 · Off-peak ×0.5

**Congestion factor (default):**

| Level | Vehicle count | Factor |
|---|---|---|
| Free-flowing | < 5 | ×1.0 |
| Busy | ≥ 5 | ×1.25 |
| Congested | ≥ 10 | ×1.5 |

For images, the vehicle count is all vehicles in the image. For videos, it is the highest count in a single frame. The period × congestion factor is capped at **×3.0**.

**Odd-even rule**

- Plate parity is determined by the **last digit of the plate number** (for example `B 1234 XYZ` → 4 → even).
- Date parity is determined by the **day of the month** (WIB time zone, UTC+7).
- If the plate parity **differs** from the date parity, the charge is multiplied by **×1.5** (default). If they match, or the plate is unreadable (confidence below 0.2), the factor is ×1.0.
- The factor is applied per vehicle **after** the cap, so ×1.5 is always exactly ×1.5.

> Rates, factors, and exchange rates are **simulation values** that can be changed in the sidebar. They are not official government tariffs.

## Configuration

The main constants are in `src/purwa_yolo/app_config.py`.

| Setting | Default | Description |
|---|---|---|
| `MAX_IMAGE_MB` | 10 | Image size limit |
| `MAX_VIDEO_MB` | 100 | Video size limit (`server.maxUploadSize` must be ≥ this value) |
| `MAX_VIDEO_SIDE` | 960 px | Longest side of the output video |
| `INFER_SIZES` / `DEFAULT_IMGSZ` | 480 / 640 / 800, default 480 | YOLO input size: smaller = faster, larger = better for small objects |
| `MIN_TRACK_FRAMES` | 3 | Minimum frames for a video vehicle to be counted |
| `CROPS_PER_VEHICLE` | 5 | Largest crops kept per vehicle (for OCR) |
| `MAX_OCR_VEHICLES` | 60 | Vehicles read per OCR run |
| `PLATE_DETECTOR_CONF` | 0.40 | Minimum confidence for the custom plate detector |
| `DEFAULT_PLATE_MIN_CONF` | 0.2 | Below this, a plate counts as unreadable |
| `DEFAULT_PLATE_SURCHARGE` | 1.5 | Odd-even multiplier |
| `UTC_OFFSET_HOURS` | 7 | WIB time zone |

On Windows + conda, `streamlit_app.py` sets `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the OpenMP error (`OMP: Error #15`). The cleaner fix is to leave only one copy of `libiomp5md.dll` in the environment.

## Limitations

- This is a **prototype/simulation**, not a production billing system.
- OCR accuracy depends on resolution, angle, lighting, and plate quality. Unreadable plates are charged at ×1.0.
- The detection dataset has only 3 classes (`car`, `van`, `bus`). Motorcycles, trucks, and other vehicle types are not supported.
- Recognised plate formats follow the Indonesian pattern (letters-digits-letters).
- Real-world ANPR/plate OCR raises privacy and personal-data-protection concerns.

## References

1. Eliasson, J., Hultkrantz, L., Nerhagen, L., & Smidfelt Rosqvist, L. (2009). *The Stockholm congestion-charging trial 2006: Overview of effects.* Transportation Research Part A, 43(3), 240-250.
2. Börjesson, M., Eliasson, J., Hugosson, M. B., & Brundell-Freij, K. (2012). *The Stockholm congestion charges: 5 years on. Effects, acceptability and lessons learnt.* Transport Policy, 20, 1-12.
3. Leape, J. (2006). *The London Congestion Charge.* Journal of Economic Perspectives, 20(4), 157-176.
4. Phang, S.-Y., & Toh, R. S. (2004). *Road congestion pricing in Singapore: 1975 to 2003.* Transportation Journal.
5. U.S. FHWA, Office of Operations. *Congestion Pricing: Examples from around the world* (ops.fhwa.dot.gov/congestionpricing).
6. Susilardi, D., Surjono, & Badriyah, N. *Design of Electronic Road Pricing in Jakarta Based on Willingness and Ability to Pay.* IIETA.
7. ANTARA (2024). *Jakarta aims to complete regulation on limiting private vehicles* (Jakarta congestion losses, Coordinating Ministry for Economic Affairs study 2018).
