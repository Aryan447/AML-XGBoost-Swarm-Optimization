# Quick Fix for Vercel 300MB Limit Error

## Immediate Solution

Your deployment is failing because the package exceeds 300MB. Here's the quickest fix:

### Step 1: Host Models on GitHub Releases

1. **Create a GitHub Release**:
   - Go to: https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/new
   - Tag: `v1.0.0`
   - Title: `Model Files v1.0.0`
   - Upload these files:
     - `models/best_model_gwo.json`
     - `models/scaler.pkl`
     - `models/feature_columns.pkl`
   - Click "Publish release"

2. **Get the Base URL**:
   ```
   https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/download/v1.0.0
   ```

### Step 2: Set Environment Variable in Vercel

1. Go to Vercel Dashboard → Your Project
2. Settings → Environment Variables
3. Add new variable:
   - **Name**: `MODEL_URL`
   - **Value**: `https://github.com/Aryan447/AML-XGBoost-Swarm-Optimization/releases/download/v1.0.0`
   - **Environment**: Production, Preview, Development (select all)
4. Click "Save"

### Step 3: Exclude Models from Deployment

Update `.vercelignore` to exclude models:

```bash
# Add this line to .vercelignore
models/
```

### Step 4: Redeploy

1. Go to Vercel Dashboard → Deployments
2. Click "Redeploy" on the latest deployment
3. Or push a new commit to trigger redeployment

## Verify It Works

After deployment:

1. Check health: `https://your-app.vercel.app/health`
2. Should show: `"model_status": "ready"`
3. Test prediction endpoint

## Alternative: Use Raw GitHub URLs

If releases don't work, use raw GitHub file URLs:

```bash
MODEL_URL=https://raw.githubusercontent.com/Aryan447/AML-XGBoost-Swarm-Optimization/main/models
```

Note: Raw URLs may have rate limits. Releases are preferred.

## Why This Works

- Models are downloaded at runtime (not in deployment package)
- Reduces deployment size by ~280KB
- Allows model updates without redeployment
- Models are cached in serverless function memory

## Still Getting Errors?

If deployment still exceeds 300MB:

1. **Check dependencies**: Some ML libraries are very large
2. **Use minimal requirements**: Consider lighter alternatives
3. **Contact Vercel Support**: They may help with size limits
4. **Upgrade to Pro**: Pro tier has better limits

## Next Steps

After fixing, see [VERCEL_MODEL_SETUP.md](VERCEL_MODEL_SETUP.md) for:
- Better model hosting options
- Caching strategies
- Performance optimization

