import sys
from sagemaker.sklearn.model import SKLearnPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# 1. MUST match the static name in deploy.py
endpoint_name = "iris-production-api" 

print(f"🧪 Testing live endpoint: {endpoint_name}...")

try:
    # 2. Add the JSON translators
    predictor = SKLearnPredictor(
        endpoint_name=endpoint_name,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    # 3. Send fake Setosa flower data
    sample_data = [[5.1, 3.5, 1.4, 0.2]]
    print("📤 Sending data to the model...")
    prediction = predictor.predict(sample_data)
    
    # 4. Verify the response
    print(f"✅ Success! Prediction received: {prediction}")
    
    # Check if the prediction actually contains data
    if len(prediction) > 0:
        print("🎉 Pipeline Integration Test Passed!")
        sys.exit(0) # Exit 0 tells GitHub Actions: "PASS (Green Check)"
    else:
        print("❌ Error: The prediction was empty!")
        sys.exit(1) # Exit 1 tells GitHub Actions: "FAIL (Red X)"

except Exception as e:
    print(f"❌ Fatal Test Error: {e}")
    sys.exit(1) # Exit 1 tells GitHub Actions: "FAIL (Red X)"