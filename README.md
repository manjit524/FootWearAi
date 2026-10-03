# 👟 FootwearAI — Multi-Page Style, Care & Exact Match Shopping Platform

An AI-powered multi-page web platform and REST API built with Flask and TensorFlow/Keras (MobileNetV2). Classify footwear styles, find **exact footwear pairs across shopping portals using 1280-dimensional deep visual embeddings**, and browse or reopen **recent footwear searches**.

---

## 🎯 Deep Visual Feature Exact Match Engine (No Color Heuristics)

Rather than relying on inaccurate color-name guessing or crude color-word searches, the platform features a **Deep Visual Feature Matching Engine** that analyzes the physical design, silhouette, sole structure, and upper panelling of the shoe:

1. **1280-Dimensional Convolutional Feature Extraction:**
   - Powered by the `MobileNetV2` deep neural network backbone (`global_average_pooling2d`).
   - Extracts a rich 1280-dimensional feature embedding vector representing the shoe's true visual signature.
2. **Visual Feature Similarity Matching:**
   - Computes visual similarity against a real-world product database across major brands (*Nike, Adidas, Puma, Clarks, Woodland, Timberland, Dr. Martens, Birkenstock, Crocs, Converse, Vans, Bata, etc.*).
   - Ranks the exact matching footwear pair with visual match percentage (e.g. *98.8% Visual Match*).
3. **Multi-Portal 1-Click Buy Links:**
   - Directly links to the exact matching product listings on:
     - **Amazon India**
     - **Myntra**
     - **AJIO**
     - **Flipkart**
     - **Google Shopping**
4. **Google Lens Web-Wide Visual Reverse Search:**
   - 1-click shortcut to launch a visual reverse search on Google Lens to scan the exact shoe SKU across the global web.
5. **Brand Refinement Filter:**
   - Instant dropdown filter to switch between top footwear brands on the fly.

---

## 🕒 Recent Searches & Scan History

Never lose a past analysis or product match! The platform includes a built-in search and scan history manager:

1. **Persistent Local Snapshots:**
   - Every shoe you upload or capture is saved to browser storage with its image thumbnail, timestamp, detected category, and confidence score.
2. **1-Click Full Scan Restoration:**
   - Click **"Open &rarr;"** on any past search to immediately re-populate the complete dashboard: image preview, exact visual shoe pair match, portal buy links (Amazon, Myntra, Flipkart, Google Lens, etc.), style advice, and care tips.
3. **Global Slide-Over Drawer:**
   - Accessible anytime from the top navigation bar via **"🕒 Recent Searches"** (with live badge counter), allowing you to review your past scans from any page.
4. **History Management:**
   - Delete individual scans or click **"Clear All History"** with one click.

---

## 🗂️ Comprehensive Footwear Taxonomy (33+ Categories)

The platform organizes footwear into **7 Macro Departments**:

1. **👔 Formal & Dress Footwear:**
   - Oxford Shoes &bull; Derby Shoes &bull; Monk Strap Shoes &bull; Brogues &bull; Classic Penny & Tassel Loafers
2. **🥾 Boots & Outdoor:**
   - Chelsea Boots &bull; Combat & Tactical Boots &bull; Chukka & Desert Boots &bull; Hiking & Trekking Boots &bull; Work & Safety Boots &bull; Western & Cowboy Boots &bull; Snow & Extreme Winter Boots
3. **👟 Sneakers & Athletic:**
   - Lifestyle & Streetwear Sneakers &bull; Performance Running Shoes &bull; Basketball High-Tops &bull; Soccer & Football Cleats &bull; Canvas Low Slip-Ons &bull; Tennis & Court Shoes
4. **👞 Casual, Mules & Slip-Ons:**
   - Boat & Deck Shoes &bull; Espadrilles &bull; Clogs & Mules &bull; House Slippers & Moccasins
5. **👡 Sandals & Open-Toe:**
   - Slide Sandals & Sliders &bull; Flip Flops & Thongs &bull; Ergonomic Footbed Sandals (Birkenstocks) &bull; Gladiator Statement Sandals
6. **👠 Women's Heels & Flats:**
   - Stilettos & High Heel Pumps &bull; Block Heel Shoes &bull; Platform Heels &bull; Wedge Heels &bull; Ballet Flats & Ballerinas
7. **🪡 Ethnic & Traditional:**
   - Mojaris & Punjabi Juttis &bull; Kolhapuri Chappals

---

## 💻 How to Run

### Method 1: Double-Click Launcher (Windows)
Double-click:
```
run.bat
```
*(or right-click `run.ps1` and select "Run with PowerShell")*

The application will start on `http://127.0.0.1:5000` and open automatically in your default browser!

### Method 2: Manual Terminal Execution
```powershell
cd d:\Shoes
.\venv\Scripts\activate
python app.py
```

---

## 📡 REST API Quickstart

### Exact Match Query API
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"category": "sneakers", "brand": "Nike"}' \
  http://127.0.0.1:5000/api/exact-match
```

### Full Image Prediction with Deep Visual Match
```bash
curl -X POST -F "image=@samples/sample_sneaker.png" http://127.0.0.1:5000/predict
```
