# Vercel Deployment Guide

This guide will help you deploy the AML Detection System to Vercel's free hosting platform.

## Quick Start

### Option 1: Deploy via Vercel Dashboard (Recommended)

1. **Prepare your repository**
   - Ensure all model files are in the `models/` directory
   - Commit all changes to Git
   - Push to GitHub, GitLab, or Bitbucket

2. **Deploy on Vercel**
   - Go to [vercel.com](https://vercel.com) and sign in
   - Click "Add New Project"
   - Import your repository
   - Vercel will auto-detect the configuration
   - Click "Deploy"

3. **Wait for deployment**
   - Vercel will build and deploy your application
   - You'll get a URL like `https://your-project.vercel.app`

### Option 2: Deploy via CLI

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

4. **Follow the prompts**
   - Link to existing project or create new
   - Confirm settings
   - Wait for deployment

## Project Structure

Your project is already configured for Vercel:

```
├── api/
│   └── index.py          # Serverless function handler
├── public/
│   └── index.html        # Frontend UI (served automatically)
├── app/                   # Application code
├── models/                # Model files (must be < 50MB each)
├── vercel.json           # Vercel configuration
└── requirements.txt      # Python dependencies
```

## Important Considerations

### Model File Size & Deployment Limit

⚠️ **Important**: Vercel has a **300MB total deployment limit**. With ML libraries (XGBoost, pandas, numpy), deployments often exceed this.

**Solution**: Host models externally and load via URL:

1. **GitHub Releases** (Recommended - Free)
   - Upload models to a GitHub release
   - Set `MODEL_URL` environment variable
   - See [VERCEL_MODEL_SETUP.md](VERCEL_MODEL_SETUP.md) for details

2. **Cloud Storage** (S3, GCS, etc.)
   - Upload models to cloud storage
   - Set `MODEL_URL` to bucket URL

3. **Vercel Blob Storage** (Pro feature)

**Quick Fix**:
```bash
# Set in Vercel Dashboard → Environment Variables
MODEL_URL=https://github.com/USERNAME/REPO/releases/download/v1.0.0
```

### Environment Variables

Set these in Vercel Dashboard → Settings → Environment Variables:

- `MODEL_DIR`: Path to models (default: `./models`)

### Cold Starts

Serverless functions have cold starts. The first request after inactivity may take 2-5 seconds.

### Function Timeout

- **Free tier**: 10 seconds
- **Pro tier**: 60 seconds

Ensure your predictions complete within the timeout limit.

## Testing Your Deployment

After deployment, test these endpoints:

1. **Frontend**: `https://your-project.vercel.app/`
2. **Health Check**: `https://your-project.vercel.app/health`
3. **API Docs**: `https://your-project.vercel.app/docs`
4. **Prediction**: `https://your-project.vercel.app/api/v1/predict`

## Troubleshooting

### Model files not found

- Ensure `models/` directory is committed to Git
- Check file paths in Vercel logs
- Verify `MODEL_DIR` environment variable

### Import errors

- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility (3.9)

### Timeout errors

- Optimize model loading (use singleton pattern - already implemented)
- Consider upgrading to Pro tier for longer timeouts
- Cache model in global scope

### Build failures

- Check Vercel build logs
- Verify `vercel.json` configuration
- Ensure `api/index.py` exists and is correct

## Custom Domain

1. Go to Vercel Dashboard → Project Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

## Monitoring

- View logs in Vercel Dashboard → Deployments → [Your Deployment] → Logs
- Monitor function execution time and errors
- Set up alerts for failures

## Updating Your Deployment

Every push to your main branch automatically triggers a new deployment.

For manual deployment:
```bash
vercel --prod
```

## Support

- [Vercel Documentation](https://vercel.com/docs)
- [Vercel Python Runtime](https://vercel.com/docs/concepts/functions/serverless-functions/runtimes/python)
- [FastAPI on Vercel](https://vercel.com/guides/deploying-fastapi-with-vercel)

