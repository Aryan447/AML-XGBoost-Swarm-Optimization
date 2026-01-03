# Vercel Model Setup Guide

## Problem: Deployment Size Limit

Vercel has a **300MB limit** for serverless function deployments. With ML libraries (XGBoost, pandas, numpy, scikit-learn), the deployment can easily exceed this limit.

## Solution: Host Models Externally

Host your model files externally and load them at runtime. This reduces deployment size and allows for easier model updates.

## Option 1: GitHub Releases (Recommended - Free)

### Step 1: Create a GitHub Release

1. Go to your repository on GitHub
2. Click "Releases" → "Create a new release"
3. Tag: `v1.0.0` (or any version)
4. Upload your model files:
   - `best_model_gwo.json`
   - `scaler.pkl`
   - `feature_columns.pkl`
5. Publish the release

### Step 2: Get Direct Download URLs

For each file, get the direct download URL:
- Format: `https://github.com/USERNAME/REPO/releases/download/TAG/FILENAME`
- Example: `https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/download/v1.0.0/best_model_gwo.json`

### Step 3: Set Environment Variables in Vercel

1. Go to Vercel Dashboard → Your Project → Settings → Environment Variables
2. Add:
   ```
   MODEL_URL=https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/download/v1.0.0
   ```
3. Redeploy

## Option 2: Cloud Storage (S3, Google Cloud, etc.)

### Using AWS S3

1. Upload models to S3 bucket
2. Make files publicly accessible (or use signed URLs)
3. Set `MODEL_URL` to your S3 bucket URL:
   ```
   MODEL_URL=https://your-bucket.s3.amazonaws.com/models
   ```

### Using Google Cloud Storage

1. Upload models to GCS bucket
2. Make files publicly accessible
3. Set `MODEL_URL` to your GCS bucket URL:
   ```
   MODEL_URL=https://storage.googleapis.com/your-bucket/models
   ```

## Option 3: Vercel Blob Storage (Pro Feature)

If you have Vercel Pro:

1. Use Vercel Blob Storage API
2. Upload models programmatically
3. Load models using Blob Storage URLs

## Option 4: Reduce Deployment Size

### Use Lighter Dependencies

Create a `requirements-vercel.txt` with minimal dependencies:

```txt
fastapi==0.109.0
mangum>=0.17.0
xgboost==2.0.3
pandas==2.2.0
numpy==1.26.3
scikit-learn==1.4.0
joblib==1.3.2
```

### Exclude Unnecessary Files

Ensure `.vercelignore` excludes:
- Tests
- Documentation
- Development files
- Large data files

## Quick Setup for GitHub Releases

```bash
# 1. Create a release (manually on GitHub or via CLI)
gh release create v1.0.0 models/*.json models/*.pkl

# 2. Set environment variable in Vercel
# MODEL_URL=https://github.com/USERNAME/REPO/releases/download/v1.0.0

# 3. Redeploy
vercel --prod
```

## Testing Model Loading

After setting `MODEL_URL`, test the deployment:

1. Check health endpoint: `https://your-app.vercel.app/health`
2. Should show `"model_status": "ready"` if models loaded successfully
3. Test prediction endpoint with a sample transaction

## Troubleshooting

### Models not loading

- Verify URLs are accessible (try in browser)
- Check Vercel function logs for download errors
- Ensure files are publicly accessible
- Check file names match exactly

### Timeout errors

- First request may be slow (downloading models)
- Consider pre-warming the function
- Use Vercel Pro for longer timeouts (60s vs 10s)

### Size still too large

- Check `vercel.json` - remove unnecessary builds
- Verify `.vercelignore` is working
- Consider using Edge Functions (if applicable)
- Split into multiple smaller functions

## Current Implementation

The `ModelService` now supports:
- ✅ Local file paths: `MODEL_DIR=./models`
- ✅ URLs: `MODEL_URL=https://...`
- ✅ Automatic download and caching
- ✅ Error handling for missing files

## Example Environment Variables

```bash
# For local development
MODEL_DIR=./models

# For Vercel with GitHub Releases
MODEL_URL=https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/download/v1.0.0

# For Vercel with S3
MODEL_URL=https://your-bucket.s3.amazonaws.com/models
```

