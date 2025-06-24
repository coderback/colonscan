#!/usr/bin/env python
import requests
import os

# Test the slide inference endpoint
def test_slide_inference():
    url = "http://histopathology:8001/infer/slide"
    
    # Check if we have a test slide file
    test_slide = "/app/media/slides/TCGA-AA-3527-01A-01-TS1.612d33b3-569d-4ea0-9dd2-f08a1ba61707.svs"
    
    if not os.path.exists(test_slide):
        print(f"Test slide not found: {test_slide}")
        return
    
    print(f"Testing slide inference with file: {test_slide}")
    print(f"File size: {os.path.getsize(test_slide) / (1024*1024):.2f} MB")
    
    try:
        with open(test_slide, 'rb') as f:
            files = {"file": (os.path.basename(test_slide), f, "application/octet-stream")}
            params = {"patch_size": 224, "overlap": 0.5, "include_heatmap": True}
            
            print("Sending request to histopathology service...")
            response = requests.post(url, files=files, params=params, timeout=60)
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                data = response.json()
                print(f"Success! Summary: {data.get('summary', 'No summary')}")
            else:
                print(f"Error response: {response.text}")
                
    except requests.exceptions.Timeout:
        print("Request timed out after 60 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    test_slide_inference() 