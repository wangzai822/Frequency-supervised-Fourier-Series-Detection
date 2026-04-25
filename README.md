<div align="center">

# 🌉 FS-FSD WebGL Damage Intelligence Viewer

**A research-oriented local viewer for SQLite-archived Fourier-based bridge defect records**  
**with native WebGL visualization, schema-adaptive loading, and contour reconstruction**

<br>

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Archive-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![WebGL](https://img.shields.io/badge/WebGL-Native%20Frontend-8A2BE2?style=for-the-badge)
![Fourier](https://img.shields.io/badge/Fourier-Contour%20Native-FF6F61?style=for-the-badge)
![Research Demo](https://img.shields.io/badge/Status-Research%20Demo-orange?style=for-the-badge)

<br><br>

**Contour-native archive visualization for Fourier-based bridge defect intelligence**

</div>

---

## 🖼️ Project Preview

<p align="center">
  <img src="demo/demo.png" width="1000" alt="FS-FSD WebGL Damage Intelligence Viewer Preview">
</p>

<p align="center">
  <em>
    Interactive inspection interface of the FS-FSD WebGL Damage Intelligence Viewer, showing
    image browsing, WebGL-rendered defect contours, defect records, and Fourier coefficient analysis.
  </em>
</p>

---

## 📌 Overview

**FS-FSD WebGL Damage Intelligence Viewer** is a lightweight local visualization system for exploring **Fourier-based structural damage archives** stored in **SQLite** databases.

Unlike conventional viewers that focus only on bounding boxes or raster masks, this project is designed around **contour-native defect records**. It enables users to:

- load an archived FS-FSD-style SQLite database,
- link it with a local image repository,
- reconstruct defect polygons from Fourier descriptors,
- inspect geometric properties interactively,
- and visualize archived damage intelligence in a clean WebGL-based interface.

The system is intended as a **research/demo-grade visualization layer** for the broader **Frequency-Supervised Fourier Series Detection (FS-FSD)** framework.

---

## 📖 Paper in Brief

This viewer is built around the ideas proposed in our research on **Fourier-based arbitrary-shape bridge defect analysis**.

### 🧠 What the paper mainly does

The core goal of the paper is to rethink how bridge defects should be represented in intelligent inspection systems.

Instead of describing defects as:

- **bounding boxes**, which are too coarse for irregular damage regions, or
- **dense raster masks**, which are storage-heavy and not ideal for compact engineering archives,

the paper proposes a **contour-native framework** called **FS-FSD**  
(**Frequency-Supervised Fourier Series Detection**).

In this framework:

- each defect is modeled as a **closed contour**,
- the contour is represented by a **compact Fourier coefficient vector**,
- defects can be **reconstructed**, **stored**, **queried**, and **visualized** as structured engineering objects,
- and heterogeneous outputs can be compared fairly under a **unified polygon-space evaluation protocol**.

### 🎯 Why this matters

For bridge inspection and infrastructure management, the final objective is not only to detect where a defect appears in an image.  
It is also to generate a representation that can support:

- geometric measurement,
- structured archiving,
- compact storage,
- future retrieval,
- and lifecycle maintenance workflows.

FS-FSD therefore treats a defect as a **compact, reusable geometric record**, rather than only a visual prediction.

### 🧩 How this viewer fits in

This viewer acts as the **interactive archive visualization layer** of that idea.

It demonstrates how Fourier-based defect records can be:

- loaded from SQLite archives,
- matched to local inspection images,
- reconstructed into polygons,
- and reviewed through an engineering-friendly WebGL interface.

---

## ✨ Key Features

### 🧠 1. Schema-Adaptive SQLite Loading
- Automatically inspects SQLite schema
- Detects image, defect, and label tables from flexible database layouts
- Supports non-standard archive names beyond fixed table conventions

### 📐 2. Fourier-Aware Defect Reconstruction
- Decodes Fourier coefficients from:
  - BLOB fields
  - JSON strings
  - numeric arrays
- Reconstructs defect polygons from archived coefficients
- Supports fallback reconstruction when full project geometry utilities are unavailable

### 🖼️ 3. Folder-Driven Image Browsing
- If an image repository folder is selected, the viewer uses that folder as the **visible image source**
- Database defect records are matched onto selected local files
- Images without matched defects are still displayed as browseable records

### ⚡ 4. Native WebGL Frontend
- Native WebGL rendering for image overlays and polygon display
- Smooth interactive browsing in the browser
- Scientific-style color palette and research-demo oriented layout

### 🔍 5. Interactive Defect Inspection
- Browse image-level defect records
- Highlight polygons and switch between instances
- Inspect geometric attributes such as:
  - area
  - perimeter
  - orientation
  - elongation
- Review Fourier coefficient spectrum for selected defects

### 🧩 6. Clean Local Deployment
- Fast local startup through FastAPI + Uvicorn
- No external database service required
- Suitable for offline experiments, archive validation, and engineering demonstrations

---

## 🚧 Why This Project Exists

Bridge defect analysis is often reduced to **bounding boxes** or **dense raster masks**, which are not ideal when the final goal is:

- engineering archival,
- compact storage,
- geometry-aware querying,
- and long-term defect reuse across maintenance workflows.

This project supports a different paradigm:

> **Defects as compact, reconstructable contour records rather than only screen-space detection outputs.**

In this system, a defect becomes a reusable geometric object that can be:

- archived efficiently,
- reconstructed on demand,
- rendered interactively,
- and inspected as a structured engineering record.

---

## 🖥️ Interface Highlights

The frontend provides a professional English interface with:

- **SQLite database selection**
- **Image repository selection**
- **Archive initialization**
- **Image search and filtering**
- **Existing-file-only filtering**
- **Image preview with polygon overlays**
- **Selectable line width and polygon sampling density**
- **Label toggling and vertex display**
- **Damage record table**
- **Selected defect preview**
- **Fourier coefficient spectrum visualization**
- **Runtime log panel**

---

## 🗂️ Project Structure

```text
FS-FSD/
├── run_webgl_viewer.py
└── webgl_viewer/
    ├── __init__.py
    ├── fsd_service.py
    ├── server.py
    ├── state.py
    ├── DB/
    │   └── fsd_defects.db
    ├── images/
    │   ├── ...
    │   └── ...
    └── static/
        ├── app.js
        ├── index.html
        └── style.css
```

### 🔧 Core Modules

- **run_webgl_viewer.py**  
  Local launcher for starting the FastAPI/Uvicorn service.

- **webgl_viewer/server.py**  
  FastAPI server that exposes stable API routes for the frontend.

- **webgl_viewer/fsd_service.py**  
  Backend service layer responsible for:
  - schema inspection
  - table auto-detection
  - archive loading
  - image/defect matching
  - Fourier decoding
  - polygon reconstruction
  - JSON-ready data preparation

- **webgl_viewer/state.py**  
  Thread-safe in-memory runtime state for the current viewer session.

- **webgl_viewer/static/**  
  Native HTML/CSS/JavaScript frontend with WebGL-based rendering.

---

## 🏗️ System Architecture

```text
SQLite Archive + Image Repository
              │
              ▼
      fsd_service.py
  (schema detection, decoding,
   matching, reconstruction)
              │
              ▼
         state.py
   (thread-safe session store)
              │
              ▼
         server.py
     (FastAPI API layer)
              │
              ▼
      app.js + WebGL UI
   (interactive browser viewer)
```

---

## 📦 Requirements

### 🐍 Python
- Python **3.9+** recommended

### 📚 Required Packages

```bash
pip install fastapi uvicorn numpy
```

### 🔗 Optional Integration

If your broader project already provides:

- `fsd_geometry.py`
- existing database utilities
- related archive-generation code

the viewer can integrate with them automatically when available.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/wangzai822/Frequency-supervised-Fourier-Series-Detection.git
cd Frequency-supervised-Fourier-Series-Detection
```

### 2. Enter the viewer directory

```bash
cd SQLite/FS-FSD
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn numpy
```

### 4. Launch the local viewer

```bash
python run_webgl_viewer.py
```

### 5. Open your browser

```text
http://127.0.0.1:8000
```

---

## ⌨️ Command-Line Usage

```bash
python run_webgl_viewer.py [OPTIONS]
```

### Common Examples

Run normally:

```bash
python run_webgl_viewer.py
```

Run with auto-reload:

```bash
python run_webgl_viewer.py --reload
```

Run on a custom port:

```bash
python run_webgl_viewer.py --port 8010
```

Automatically find the next free port:

```bash
python run_webgl_viewer.py --auto-port
```

Do not open browser automatically:

```bash
python run_webgl_viewer.py --no-browser
```

---

## 🧭 Typical Workflow

### Step 1 — Select SQLite Database
Choose an archived defect database from the UI or by using the native file picker.

### Step 2 — Select Image Repository
Choose the folder containing the corresponding original inspection images.

### Step 3 — Initialize Archive
The backend will:

- inspect the schema
- auto-detect relevant tables
- load defect records
- build image index
- match image rows and defect rows
- prepare summary statistics

### Step 4 — Browse Images
The viewer displays image cards, image status, and defect counts.

### Step 5 — Inspect Defects
Select an image to:

- view polygon overlays
- click a damage instance
- inspect geometric attributes
- review Fourier coefficient spectrum

---

## 🧪 Data Handling Logic

A key design decision in this viewer is the **folder-driven display mode**.

When an image folder is selected:

- visible images are driven by the selected folder
- only files inside that folder are displayed in the browser
- database records are attached only when they match those files
- images without matched defects are still shown
- defects that do not match the selected folder are not promoted into visible image cards

This behavior is intentional and makes the viewer practical for archive validation against a chosen image repository.

---

## 🔌 API Overview

The frontend communicates with the backend through a stable FastAPI layer.

| Endpoint | Purpose |
|---|---|
| `/api/health` | Health check and runtime status |
| `/api/state` | Current runtime snapshot |
| `/api/schema` | Current schema inspection result |
| `/api/initialize` | Open archive and initialize session |
| `/api/summary` | Return archive summary |
| `/api/images` | List images with optional filters |
| `/api/records` | Return all records for one image |
| `/api/defect/{defect_id}` | Return one defect in detail |
| `/api/image` | Serve resolved local image file |

### Native Pickers

| Endpoint | Purpose |
|---|---|
| `/api/select-db` | Open native file dialog for SQLite database |
| `/api/select-folder` | Open native directory picker for image repository |

---

## 🛠️ Technical Highlights

### Backend
- **FastAPI**
- **Uvicorn**
- **SQLite**
- **NumPy**
- thread-safe in-memory session management

### Frontend
- **HTML**
- **CSS**
- **Vanilla JavaScript**
- **Native WebGL rendering**

### Archive Intelligence
- flexible schema scoring
- fuzzy column detection
- robust table-role identification
- image/defect matching across heterogeneous archives

---

## 🔬 Research Context

This viewer is closely related to our FS-FSD research on:

### **Contour-Native Bridge Defect Analysis and Compact Archiving via Frequency-Supervised Fourier Series**

The broader research motivation is to move from:

- box-native outputs,
- mask-native outputs,

toward:

- **compact contour-native defect records**
- **direct geometric reconstruction**
- **lightweight archival**
- **interactive engineering review**
- **future reuse in infrastructure lifecycle workflows**

This project demonstrates how those ideas can be translated into a practical local archive viewer.

---

## 💡 Design Philosophy

This project is built around four ideas:

1. **Contours are first-class data objects**  
   Defects should be stored and reused as geometry-aware records.

2. **Archives should be practical**  
   Data loading must remain usable even when schemas are imperfect or non-standard.

3. **Visualization should support inspection, not only display**  
   The UI is designed for record review, defect selection, and coefficient-level analysis.

4. **Local deployment matters**  
   The entire system is lightweight enough to run as a local archive viewer.

---

## 🛣️ Future Directions

Potential future extensions include:

- richer archive browser and session export
- multi-database comparison
- defect timeline comparison across inspections
- topology-aware contour review
- tighter integration with BIM / GIS / digital twin workflows
- deployment of precomputed archive summaries for larger datasets

---

## 📚 Citation

If this project supports your research or engineering workflow, please consider citing the accompanying FS-FSD work.

```bibtex
@article{liu2026fsfsd,
  title={Contour-Native Bridge Defect Analysis and Compact Archiving via Frequency-Supervised Fourier Series},
  author={Liu, Jin and Wang, Wang and Pu, Hongxu and Cao, Zhen and Wang, Yasong and Wang, Hu and Qi, Xiaojuan},
  journal={Preprint submitted to Elsevier},
  year={2026}
}
```

> Citation details can be updated after formal publication.

---

## 📝 Notes

- This repository is focused on **local archive visualization**, not cloud deployment.
- The frontend is intentionally fully **English** for professional demo and paper presentation usage.
- If your project already includes a geometry backend such as `fsd_geometry.py`, the viewer will attempt to use it automatically when available.

---

## 📬 Contact

**Author / Maintainer**  
GitHub: [wangzai822](https://github.com/wangzai822)  
Email: **wangw00821@gmail.com**

---

## 🙏 Acknowledgement

This viewer is developed as a practical visualization layer for **Fourier-based arbitrary-shape defect analysis**, with emphasis on:

- compact archive inspection
- contour reconstruction
- engineering-oriented defect review
- lightweight local demonstration

---

<div align="center">

**FS-FSD WebGL Damage Intelligence Viewer**  
*Contour-native archive visualization for bridge defect intelligence* 🚀

</div>
