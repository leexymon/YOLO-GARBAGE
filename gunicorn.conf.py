import multiprocessing

# Gunicorn configuration for memory-constrained environments (e.g., Render Free Tier 512MB)
# Binding
bind = "0.0.0.0:10000"

# Workers
# Limit to 1 worker to avoid copying the models into multiple memory spaces.
workers = 1

# Threads
# Allow concurrent requests to be handled efficiently using threads.
threads = 4

# Timeout
# Increase timeout since models running on CPU might take a bit longer for large images.
timeout = 120

# Worker class
worker_class = 'gthread'

# Memory Management Options
# Preload app allows models to load before forks, but with 1 worker it just helps startup time
preload_app = True
max_requests = 100
max_requests_jitter = 10
