from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from .endpoints import router

app = FastAPI(
    title='Iris Classification API',
    description='A simple API for classifying iris flowers',
    version='1.0.0'
)

# Setup templates
templates = Jinja2Templates(directory="src/api/templates")

# Mount the router with a prefix
app.include_router(router, prefix="/api/v1")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("welcome.html", {"request": request})