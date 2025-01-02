import os
import base64
from google.cloud import aiplatform
from google.api_core.exceptions import GoogleAPICallError, InvalidArgument

def upload_model_sample(
    project: str,
    location: str,
    display_name: str,
    serving_container_image_uri: str,
    artifact_uri: str,
    endpoint_display_name: str,
):

    # Decode the Base64 encoded Google credentials and write them to a file
    try:
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google-credentials.json'
        with open('google-credentials.json', 'w') as file:
            file.write(base64.b64decode(os.environ['GOOGLE_CREDENTIALS']).decode())
    except KeyError:
        print("GOOGLE_CREDENTIALS environment variable is not set.")
        return

    try:
        aiplatform.init(project=project, location=location)

        # Upload the model
        model = aiplatform.Model.upload(
            display_name=display_name,
            artifact_uri=artifact_uri,
            serving_container_image_uri=serving_container_image_uri,
        )

        model.wait()  # Wait until the model is uploaded successfully

        # Create an Endpoint
        endpoint = aiplatform.Endpoint.create(
            display_name=endpoint_display_name,
            project=project,
            location=location,
        )

        # Deploy the Model to the Endpoint
        endpoint.deploy(
            model=model,
            deployed_model_display_name=display_name,
            traffic_percentage=100,
            sync=True
        )

        print(f"Model deployed successfully: {model.display_name}")
        print(f"Model resource name: {model.resource_name}")
        print(f"Endpoint resource name: {endpoint.resource_name}")
        
        return model

    except GoogleAPICallError as e:
        print(f"API call failed: {e}")
    except InvalidArgument as e:
        print(f"Invalid argument error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


upload_model_sample(
    project="gitlab-vertexai",
    location="us-central1",
    display_name="credit-card-fraud-detection",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-2:latest",
    artifact_uri="gs://gitlab-vertexai-bucket/models/",
    endpoint_display_name="vertex-ai-endpoint",  
)
