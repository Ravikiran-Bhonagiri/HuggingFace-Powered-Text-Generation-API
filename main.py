import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import Optional
from config import config
import logging

# Initialize FastAPI app
app = FastAPI()

# Set up logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Function to load the model and tokenizer with proper error handling
def load_model_and_tokenizer():
    """
    Loads the pre-trained model and tokenizer from Hugging Face using the configuration.
    Raises a RuntimeError if loading fails.
    """
    try:
        logging.info("Loading model and tokenizer...")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            device_map=config["device"],
            torch_dtype=config["torch_dtype"],
            trust_remote_code=config["trust_remote_code"]
        )
        tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
        logging.info("Model and tokenizer loaded successfully.")
        return model, tokenizer
    except FileNotFoundError as fnf_error:
        logging.error(f"Model files not found: {fnf_error}")
        raise RuntimeError("Model files are missing. Please check the model path.")
    except ValueError as val_error:
        logging.error(f"Value error while loading model: {val_error}")
        raise RuntimeError("Invalid model configuration.")
    except Exception as e:
        logging.error(f"Unexpected error during model loading: {e}")
        raise RuntimeError(f"Error loading model or tokenizer: {e}")

# Load the model and tokenizer when the application starts
try:
    model, tokenizer = load_model_and_tokenizer()
except RuntimeError as e:
    logging.critical(f"Failed to start the API due to model loading issue: {e}")
    raise HTTPException(status_code=500, detail="Model loading failed.")

# Define the request schema using Pydantic
class TextGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=1000, description="Input text prompt")
    temperature: Optional[float] = Field(
        default=config["default_temperature"],
        ge=config["min_temperature"],
        le=config["max_temperature"],
        description="Controls randomness of the output (0.0 to 2.0)"
    )
    top_p: Optional[float] = Field(
        default=config["default_top_p"],
        ge=config["min_top_p"],
        le=config["max_top_p"],
        description="Controls diversity via nucleus sampling (0.1 to 1.0)"
    )
    max_length: Optional[int] = Field(
        default=config["default_max_length"],
        ge=config["min_max_length"],
        le=config["max_max_length"],
        description="Maximum length of generated text"
    )

def generate_text(prompt: str, temperature: float, top_p: float, max_length: int) -> str:
    """
    Generate text using the pre-trained model.
    This function handles tokenization, model inference, and decoding.
    """
    try:
        # Validate CUDA availability if device is set to 'cuda'
        if config["device"] == "cuda" and not torch.cuda.is_available():
            logging.warning("CUDA device not available, falling back to CPU.")
            raise RuntimeError("CUDA device not available. Please set 'device' to 'cpu' in config.")

        # Tokenize the input prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(config["device"])

        # Check if input tokenization was successful
        if inputs["input_ids"].nelement() == 0:
            raise ValueError("The input prompt could not be tokenized. Please provide a valid prompt.")

        # Generate text using the model with error handling for CUDA and inference issues
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True
            )

        # Decode the generated tokens into a human-readable format
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        if not generated_text:
            raise ValueError("Generated text is empty. Adjust the input prompt or parameters.")
        
        return generated_text

    except torch.cuda.CudaError as cuda_err:
        logging.error(f"CUDA error: {cuda_err}")
        raise RuntimeError("CUDA error occurred. Please check your GPU setup.")
    except ValueError as val_err:
        logging.error(f"Value error during text generation: {val_err}")
        raise RuntimeError(f"Invalid input or parameters: {val_err}")
    except Exception as e:
        logging.error(f"Unexpected error during text generation: {e}")
        raise RuntimeError(f"Error generating text: {e}")

# Define the API endpoint for health check
@app.get("/")
def root():
    return {"message": "Welcome to the Text Generation API!"}

# Define the API endpoint for text generation
@app.post("/generate")
def generate(request: TextGenerationRequest):
    """
    Endpoint to generate text based on user input.
    Handles input validation and text generation.
    """
    try:
        logging.info(f"Received request with prompt: {request.prompt}")
        # Call the text generation function
        generated_text = generate_text(
            request.prompt, request.temperature, request.top_p, request.max_length
        )
        return {"generated_text": generated_text}

    except ValueError as ve:
        logging.error(f"Validation error: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except RuntimeError as re:
        logging.error(f"Runtime error: {re}")
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred during text generation.")

# Custom error handler for 404 errors
@app.exception_handler(404)
async def not_found_error(request: Request, exc):
    logging.warning("404 error - Resource not found.")
    return {"detail": "The requested resource was not found."}

# Custom error handler for 500 errors
@app.exception_handler(500)
async def internal_server_error(request: Request, exc):
    logging.error(f"500 error - Internal server error: {exc}")
    return {"detail": "Internal server error. Please try again later."}
