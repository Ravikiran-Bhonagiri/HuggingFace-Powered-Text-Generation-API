
# Text Generation API using Hugging Face Transformers

## Overview
This project provides an API for generating text using a pre-trained Hugging Face transformer model. The API is built using **FastAPI** and leverages **environment variables** for configuration management, making it flexible and easy to deploy.

The API uses the `microsoft/Phi-3-mini-4k-instruct` model for text generation by default. However, you can easily change the model and other settings using a configuration file.

---

## Table of Contents
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the API](#running-the-api)
- [Testing the API](#testing-the-api)
- [Endpoints](#endpoints)
- [Environment Variables](#environment-variables)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)

---

## Project Structure
```
text_generation_api/
├── main.py           # FastAPI application code
├── config.py         # Configuration management script
├── config.json       # Configuration file for model and settings
├── requirements.txt  # Python dependencies
└── README.md         # Project documentation
```

---

## Prerequisites
Ensure you have the following installed:
- Python 3.8 or higher
- `pip` (Python package manager)

---

## Installation


1. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Configuration

### `config.json`
Before running the API, configure the `config.json` file as needed:

```json
{
  "model_name": "microsoft/Phi-3-mini-4k-instruct",
  "device": "cuda",
  "torch_dtype": "auto",
  "trust_remote_code": true,
  "default_temperature": 0.7,
  "default_top_p": 0.9,
  "default_max_length": 50,
  "min_temperature": 0.0,
  "max_temperature": 2.0,
  "min_top_p": 0.1,
  "max_top_p": 1.0,
  "min_max_length": 1,
  "max_max_length": 1000
}
```

### Environment Variables
Set the `CONFIG_FILE` environment variable to specify the configuration file:

```bash
export CONFIG_FILE=config.json
```

On Windows:
```powershell
set CONFIG_FILE=config.json
```

---

## Running the API

To start the FastAPI server, run:

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- The server will be accessible at [http://localhost:8000](http://localhost:8000).
- The `--reload` flag enables auto-reloading for development.

---

## Testing the API

### 1. Using `curl`
```bash
curl -X POST "http://localhost:8000/generate" -H "Content-Type: application/json" -d '{"prompt": "Tell me a joke about chickens.", "temperature": 0.7, "top_p": 0.8, "max_length": 50}'
```

### 2. Using Swagger UI
Open [http://localhost:8000/docs](http://localhost:8000/docs) to access the interactive Swagger documentation.

### Example Response
```json
{
  "generated_text": "Why did the chicken join a band? Because it had the drumsticks!"
}
```

---

## Endpoints

### `GET /`
- **Description**: Health check for the API.
- **Response**:
  ```json
  {"message": "Welcome to the Text Generation API!"}
  ```

### `POST /generate`
- **Description**: Generate text based on the given prompt and parameters.
- **Request Body**:
  ```json
  {
    "prompt": "Write a funny joke",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_length": 50
  }
  ```
- **Response**:
  ```json
  {
    "generated_text": "Why did the chicken cross the road? To prove it wasn’t a chicken!"
  }
  ```

---

## Environment Variables

| Variable       | Description                              | Default Value   |
|----------------|------------------------------------------|-----------------|
| `CONFIG_FILE`  | Path to the configuration file           | `config.json`   |

---

## Deployment

### Deploying to Heroku
1. Create a `Procfile`:
   ```plaintext
   web: uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
   ```

2. Push to Heroku:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   heroku create
   git push heroku master
   ```

### Deploying to AWS EC2
1. SSH into your EC2 instance.
2. Clone the repository and install dependencies.
3. Run the API using `uvicorn`.

---

## Troubleshooting

### Common Issues

1. **Model Loading Errors**:
   - Ensure that the model name in `config.json` is correct.
   - Make sure `torch` and `transformers` libraries are installed.

2. **CUDA Device Errors**:
   - If you don’t have a GPU, set `"device": "cpu"` in `config.json`.

3. **Configuration Errors**:
   - Check if the `CONFIG_FILE` environment variable is set correctly.
   - Ensure that `config.json` is properly formatted.

---

## Contributing
Feel free to open issues or pull requests if you find bugs or want to add new features.

## License
This project is licensed under the MIT License.

## Author
Developed by [Ravikiran Bhonagiri](https://github.com/Ravikiran-Bhonagiri).
