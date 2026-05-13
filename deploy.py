import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time
import boto3

role="arn:aws:iam::477010600979:role/service-role/AmazonSageMaker-ExecutionRole-20260511T140775"

boto_session = boto3.Session(region_name="eu-north-1")
session = sagemaker.Session(boto_session=boto_session)


endpoint_name = "iris-production-api"

model = SKLearnModel(
    model_data="s3://muhammed-sagemaker-models-123/model.tar.gz",
    role=role,
    entry_point="inference.py",  
    framework_version="1.2-1",
    sagemaker_session=session
)

# 🔍 Check if endpoint exists
sm_client = boto3.client("sagemaker", region_name="eu-north-1")

try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_exists = True
except:
    endpoint_exists = False

# 🚀 Deploy logic
if endpoint_exists:
    print("🔄 Updating existing endpoint...")
    predictor = model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        update_endpoint=True
    )
else:
    print("🚀 Creating new endpoint...")
    predictor = model.deploy(
        instance_type="ml.m5.large",
        initial_instance_count=1,
        endpoint_name=endpoint_name
    )

print("✅ Deployment complete")