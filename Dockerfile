# Stage 1: Use an official Python runtime as a parent image.
# Using 'slim' results in a smaller, more secure final image.
FROM python:3.11-slim

# Stage 2: Set the working directory inside the container.
# This is where our application code will live.
WORKDIR /app

# Stage 3: Copy the requirements file first.
# This leverages Docker's layer caching. The dependencies will only be re-installed
# if the requirements.txt file changes, making subsequent builds much faster.
COPY requirements.txt .

# Stage 4: Install the Python dependencies.
# --no-cache-dir ensures that pip doesn't store the download cache,
# keeping the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy the application code and model files into the container.
# This includes our server script and the directory containing the trained model.
COPY server.py .
COPY ./model ./model

# Stage 6: Expose the port the app runs on.
# This informs Docker that the container listens on port 8000.
EXPOSE 8000

# Stage 7: Define the command to run the application.
# This command starts the Uvicorn server.
# We use --host 0.0.0.0 to make the server accessible from outside the container.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

