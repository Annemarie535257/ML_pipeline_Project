import requests
import json

def test_api():
    """Test the API endpoints"""
    base_url = "http://localhost:5000"
    
    print("Testing API endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"Health endpoint status: {response.status_code}")
        if response.status_code == 200:
            health_data = response.json()
            print(f"Health data: {json.dumps(health_data, indent=2)}")
        else:
            print(f"Health response: {response.text}")
    except Exception as e:
        print(f"Health endpoint error: {e}")
    
    # Test model info endpoint
    try:
        response = requests.get(f"{base_url}/model/info", timeout=10)
        print(f"Model info endpoint status: {response.status_code}")
        if response.status_code == 200:
            model_data = response.json()
            print(f"Model loaded: {'error' not in model_data}")
        else:
            print(f"Model info response: {response.text}")
    except Exception as e:
        print(f"Model info endpoint error: {e}")

if __name__ == "__main__":
    test_api() 