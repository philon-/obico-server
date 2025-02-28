# Standalone ml_api 

This is a modified ml_api implementation based on the original by Obico.

Notably, many of the parameters have been moved into ml_api_standalone.env, and a new method to analyze images already present on the filesystem is introduced.

## Usage

There is a sample `docker-compose.yaml` that can be used as a start when setting up your docker service.

### Environment variables
The most relevant variables are listed below, for a full list see ml_api_standalone.env
```
# Port to listen to
# ML_API_PORT=3333

# Host to listen to
# ML_API_HOST=0.0.0.0

# If we want to read from an image file instead of the ?img= argument, provide this variable.
ML_INPUT_IMAGE=/app/image.jpg

# If we want to render the boxes we can do that to this filename
ML_OUTPUT_IMAGE=/app/output.jpg

# Add confidence to output file? 
ML_OUTPUT_TEXT=True

# NMS threshold
# ML_NMS=0.45

# DET threshold
# ML_THRESH=0.08
```