import sys
import boto3
import sagemaker

from sagemaker.sklearn.model import SKLearnPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# Create AWS session with explicit region
boto_session = boto3.Session(region_name="eu-north-1")

sagemaker_session = sagemaker.Session(
    boto_session=boto_session
)

endpoint_name = "iris-production-api"

print(f"🧪 Testing live endpoint: {endpoint_name}...")

try:
    predictor = SKLearnPredictor(
        endpoint_name=endpoint_name,
        sagemaker_session=sagemaker_session,
        serializer=JSONSerializer(),
        deserializer=JSONDeserializer()
    )

    sample_data = [[5.1, 3.5, 1.4, 0.2]]

    print("📤 Sending data to the model...")

    prediction = predictor.predict(sample_data)

    print(f"✅ Success! Prediction received: {prediction}")

    if len(prediction) > 0:
        print("🎉 Pipeline Integration Test Passed!")
        sys.exit(0)
    else:
        print("❌ Error: Empty prediction!")
        sys.exit(1)

except Exception as e:
    print(f"❌ Fatal Test Error: {e}")
    sys.exit(1)