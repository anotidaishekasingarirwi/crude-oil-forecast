# Deployment Guide — Render (Free Tier)

## Prerequisites
- GitHub account
- Render account (render.com — free)
- Your trained model files in a `models/` folder

---

## Step 1 — Create a GitHub Repository

1. Go to https://githupyb.com/new
2. Name it `crude-oil-forecast` (or any name)
3. Set to **Public** (required for Render free tier)
4. Click **Create repository**

---

## Step 2 — Push Your Code

Open a terminal in your project folder and run:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/crude-oil-forecast.git
git push -u origin main
```

> Your `models/` folder is in `.gitignore` — do NOT push model files to GitHub.
> You will upload them directly to Render's disk in Step 5.

---

## Step 3 — Create a Render Web Service

1. Go to https://dashboard.render.com
2. Click **New** → **Web Service**
3. Connect your GitHub account and select your repo
4. Configure:
   - **Name:** `crude-oil-forecast`
   - **Region:** pick closest to you
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn backend.app:app --workers 1 --timeout 120 --bind 0.0.0.0:$PORT`
   - **Plan:** Free

5. Click **Create Web Service**

---

## Step 4 — Add a Persistent Disk (for model files)

1. In your Render service dashboard, go to **Disks** tab
2. Click **Add Disk**
3. Set:
   - **Name:** `models-disk`
   - **Mount Path:** `/opt/render/project/src/models`
   - **Size:** 1 GB (free)
4. Click **Save**

---

## Step 5 — Upload Your Model Files

Since models are large binary files (not in git), upload them via Render Shell:

1. In your Render service, go to **Shell** tab
2. Run:
   ```bash
   ls /opt/render/project/src/models/
   ```
3. Upload each model file using the Render dashboard **Files** section, or use the shell + curl to pull from Google Drive / Dropbox.

### Option A — Upload via Render Shell (if files are online)
```bash
cd /opt/render/project/src/models
# If your files are on Google Drive (use direct download links):
curl -L "YOUR_DIRECT_LINK" -o lstm_model.h5
curl -L "YOUR_DIRECT_LINK" -o cnn_model.h5
curl -L "YOUR_DIRECT_LINK" -o lgb_model.pkl
curl -L "YOUR_DIRECT_LINK" -o ridge_meta_model.pkl
curl -L "YOUR_DIRECT_LINK" -o scaler.pkl
curl -L "YOUR_DIRECT_LINK" -o config.json
curl -L "YOUR_DIRECT_LINK" -o metrics.json
```

### Option B — SCP from your local machine
```bash
# Get your Render SSH key from Settings → SSH Keys
scp -i ~/.ssh/render_key models/* user@your-render-host:/opt/render/project/src/models/
```

---

## Step 6 — Deploy

1. Render auto-deploys on every push to `main`
2. Watch the **Logs** tab — first deploy takes ~5 min (TensorFlow is large)
3. Once live, visit your URL: `https://crude-oil-forecast.onrender.com`

---

## Step 7 — Test

```bash
# Health check
curl https://your-app.onrender.com/api/health

# Live forecast
curl https://your-app.onrender.com/api/forecast/auto
```

---

## Notes

- **Free tier sleeps after 15 min of inactivity** — first request after sleep takes ~30s to wake
- **TF-CPU** is used to keep memory under Render's 512MB free limit
- If you exceed memory, upgrade to Render Starter ($7/mo) or remove TF models and use LightGBM-only mode
- Model files on the disk **persist across deploys** — you only upload once
