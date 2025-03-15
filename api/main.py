import sys
import os
from typing import List, Any
import time
from functools import partial
import numpy as np

from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from underthesea import word_tokenize

import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

from utils import client

# Parse environment variables
#
model_name    = os.getenv("MODEL_NAME")
model_version = os.getenv("MODEL_VERSION", "")
batch_size    = int(os.getenv("BATCH_SIZE", 1))
#
url           = os.getenv("TRITON_URL", "localhost:8000")
protocol      = os.getenv("PROTOCOL", "HTTP")
verbose       = os.getenv("VERBOSE", "False").lower() in ("true", "1", "t")
async_set     = os.getenv("ASYNC_SET", "False").lower() in ("true", "1", "t")


try:
    if protocol.lower() == "grpc":
        # Create gRPC client for communicating with the server
        triton_client = grpcclient.InferenceServerClient(
            url=url, verbose=verbose
        )
    else:
        # Specify large enough concurrency to handle the number of requests.
        concurrency = 20 if async_set else 1
        triton_client = httpclient.InferenceServerClient(
            url=url, verbose=verbose, concurrency=concurrency
        )
except Exception as e:
    print("client creation failed: " + str(e))
    sys.exit(1)

try:
    model_metadata = triton_client.get_model_metadata(
        model_name=model_name, model_version=model_version
    )
    model_config = triton_client.get_model_config(
        model_name=model_name, model_version=model_version
    )
except InferenceServerException as e:
    print("failed to retrieve model metadata: " + str(e))
    sys.exit(1)

if protocol.lower() == "grpc":
    model_config = model_config.config
else:
    model_metadata, model_config = client.convert_http_metadata_config(
        model_metadata, model_config
    )

# parsing information of model
max_batch_size, input_name, output_name, format, dtype = client.parse_model(
    model_metadata, model_config
)

supports_batching = max_batch_size > 0
if not supports_batching and batch_size != 1:
    print("ERROR: This model doesn't support batching.")
    sys.exit(1)


class ListStr(BaseModel):
    texts: List[str]

############
# FastAPI
############


app = FastAPI()

@app.get("/")
def root():
    return {"Hello": "World"}

@app.post("/viencoder")
async def viencoder(textRequest: ListStr) -> JSONResponse:

    # Word-segment the input texts
    texts = textRequest.texts
    text_responses = await preprocessing(texts)
    print(text_responses)
    text_obj = np.array(text_responses, dtype="object")

    # Generate the request
    inputs, outputs = requestGenerator(
        text_obj, input_name, output_name, dtype
    )
    # Perform inference
    try:
        start_time = time.time()

        if protocol.lower() == "grpc":
            user_data = client.UserData()
            embeddings = triton_client.async_infer(
                model_name,
                inputs,
                partial(client.completion_callback, user_data),
                model_version=model_version,
                outputs=outputs,
            )
        else:
            async_request = triton_client.async_infer(
                model_name,
                inputs,
                model_version=model_version,
                outputs=outputs,
            )
    except InferenceServerException as e:
        return {"Error": "Inference failed with error: " + str(e)}

    # Collect results from the ongoing async requests
    if protocol.lower() == "grpc":
        (embeddings, error) = user_data._completed_requests.get()
        if error is not None:
            return {"Error": "Inference failed with error: " + str(error)}
    else:
        # HTTP
        embeddings = async_request.get_result()

    # Process the results    
    end_time = time.time()
    print("Process time: ", end_time - start_time)
    print("new")

    return JSONResponse({
        "embeddings": embeddings.as_numpy(output_name).tolist(),
    })


@app.post("/word-segment")
async def preprocessing(texts: List[str]) -> List[str]:
    return [word_tokenize(sentence, format="text") for sentence in texts]


###################
# Helper functions
###################

def requestGenerator(text_obj, input_name, output_name, dtype):
    # define protocol
    if protocol.lower() == "grpc":
        client = grpcclient
    else:
        client = httpclient

    # Set the input and output data
    inputs = [client.InferInput(input_name, text_obj.shape, dtype)]
    inputs[0].set_data_from_numpy(text_obj)
    outputs = [client.InferRequestedOutput(output_name)]
    return inputs, outputs