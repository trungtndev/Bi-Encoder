# Vietnamese BiEncdoer

This repostory is used for deploying model [bkai-foundation-models/vietnamese-bi-encoder](https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder) by using Triton Inference Server and FastAPI. 

## 1. Prepare
### Export model
Currently model is exported at path `/models/viencoder/1` using optimum of huggingface:
```
optimum-cli export onnx --model "bkai-foundation-models/vietnamese-bi-encoder" models/viencoder/1
```

### With Docker Compose
```
docker pull heronq02/viencoder:cuda11.8-final
docker pull nvcr.io/nvidia/tritonserver:24.07-py3
docker pull nvcr.io/nvidia/tritonserver:24.07-py3-sdk
docker-compose up
```

## 2. Running Docker with two images
```
docker-compose up
```
This command will start two images: 
- tritonserver: using file `Dockerfile.triton` to build more required packages as well as deloying docker, currently image version is `24.07` . 
- viencoder: using `Dockerfile` to create image for client. After being initialized, we can access API through `http://127.0.0.1:7999/docs`

> [!WARNING]  
> All of image are require GPU device, all of configuration can be viewed on file `docker-compose.yaml`

## 3. Use API
FastAPI will deploy API at port 7999. This is a sample of request to server:

```
curl -X 'POST' 'http://127.0.0.1:7999/viencoder' \
     -H 'Content-Type: application/json' \
     -d '{"texts": [
         "Đây là một đoạn văn mẫu gửi đến server FastAPI.", 
         "Here is another text to process."
     ]
     }'
```
The output will be:
```
{
    "embeddings": [
            [...],
            [...],
        [
}   
```

## 4. Benchmark

Open and acess to the container of client and run the following command:
```
docker run --rm -it --net=host nvcr.io/nvidia/tritonserver:24.07-py3-sdk
```
The CMD will be look like:
```
root@nttrung:/workspace#
```

Mearure the performance of the model by using `perf_analyzer`:
```
perf_analyzer -m ensemble_model --shape TEXT:8 --concurrency-range 1:4 --collect-metrics
```
The output will be:
```
*** Measurement Settings ***
  Batch size: 1
  Service Kind: TRITON
  Using "time_windows" mode for stabilization
  Stabilizing using average latency and throughput
  Measurement window: 5000 msec
  Latency limit: 0 msec
  Concurrency limit: 4 concurrent requests
  Using synchronous calls for inference
...
Request concurrency: 1
...
Request concurrency: 2
...
Request concurrency: 3
...
Request concurrency: 4
...

Inferences/Second vs. Client Average Batch Latency
Concurrency: 1, throughput: 4.33197 infer/sec, latency 230834 usec
Concurrency: 2, throughput: 4.05339 infer/sec, latency 490687 usec
Concurrency: 3, throughput: 4.33268 infer/sec, latency 703616 usec
Concurrency: 4, throughput: 4.05413 infer/sec, latency 963780 usec
```





 




# Bi-Encoder
