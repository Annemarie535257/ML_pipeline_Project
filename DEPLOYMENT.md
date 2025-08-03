# Deployment Guide for Plant Disease Classification System

## Overview
This project consists of two services that need to be deployed separately on Render:

1. **API Server** (`api_server.py`) - Backend service
2. **Streamlit App** (`UI/streamlitapp.py`) - Frontend service

## Deployment Steps

### Step 1: Deploy the API Server

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `plant-disease-api`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python api_server.py`
   - **Plan**: Free

5. Add Environment Variables:
   - `PYTHON_VERSION`: `3.11.9`

6. Click "Create Web Service"

### Step 2: Deploy the Streamlit App

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:
   - **Name**: `plant-disease-app`
   - **Environment**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run UI/streamlitapp.py --server.port $PORT --server.address 0.0.0.0`
   - **Plan**: Free

5. Add Environment Variables:
   - `PYTHON_VERSION`: `3.11.9`
   - `API_BASE_URL`: `https://your-api-service-name.onrender.com` (replace with your actual API service URL)

6. Click "Create Web Service"

### Step 3: Update API URL

After both services are deployed:

1. Get your API service URL from the Render dashboard
2. Go to your Streamlit app service settings
3. Update the `API_BASE_URL` environment variable with your actual API service URL
4. Redeploy the Streamlit app

## Alternative: Using render.yaml

You can also deploy both services using the `render.yaml` file:

1. Push the `render.yaml` file to your repository
2. Go to Render Dashboard
3. Click "New +" → "Blueprint"
4. Connect your repository
5. Render will automatically create both services

## Troubleshooting

### Common Issues:

1. **API Server Not Found**: Make sure the `API_BASE_URL` environment variable is set correctly in your Streamlit app service.

2. **Model Files Missing**: Ensure all model files (`*.h5`, `preprocessor.pkl`) are in the `models/` directory and committed to your repository.

3. **Port Issues**: The API server runs on port 5000, and Streamlit uses the `$PORT` environment variable provided by Render.

4. **Memory Issues**: If you encounter memory issues, consider upgrading to a paid plan or optimizing the model size.

### Health Checks:

- API Server: `https://your-api-service.onrender.com/health`
- Streamlit App: `https://your-streamlit-service.onrender.com`

## Local Development

For local development, you can still use the auto-start functionality:

```bash
cd UI
streamlit run streamlitapp.py
```

This will automatically start the API server in the background for local development only. 