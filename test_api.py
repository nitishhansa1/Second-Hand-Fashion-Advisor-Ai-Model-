from fastapi.testclient import TestClient
from main import app, classes
import json

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200
    print("GET / -> HTML Response, preview:", response.text[:50])

def test_categories():
    response = client.get("/categories")
    assert response.status_code == 200
    print("GET /categories ->", {k: v for i, (k, v) in enumerate(response.json()['categories'].items()) if i < 5}, '... (truncated)')

def test_predict():
    import os
    # Find a test image
    test_img_path = "archive/categorized_images/A/0.jpg" # Dummy path, let's use glob
    import glob
    test_imgs = glob.glob("archive/categorized_images/*/*.jpg") + glob.glob("archive/images/*.jpg")
    if not test_imgs:
        print("No test images found.")
        return
    test_img_path = test_imgs[0]

    with open(test_img_path, "rb") as f:
        file_bytes = f.read()

    response = client.post(
        "/predict",
        files={"file": ("test_img.jpg", file_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 200
    print(f"POST /predict ->", response.json())

if __name__ == "__main__":
    print("Testing Home Endpoint:")
    test_home()
    print("\nTesting Categories Endpoint:")
    test_categories()
    print("\nTesting Predict Endpoint:")
    test_predict()
    print("\nAll tests passed successfully.")
