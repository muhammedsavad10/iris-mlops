import boto3

runtime = boto3.client(
    "sagemaker-runtime",
    region_name="eu-north-1"
)

endpoint_name = "iris-endpoint--1778601925"

payload = "5.1,3.5,1.4,0.2"

response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="text/csv",
    Body=payload
)

result = response["Body"].read().decode()

print(result)