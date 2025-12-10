# Start the FastAPI server

uvicorn main:app --reload

api endpoint at http://localhost:8000/api/voice-query/

UI can be accessed at http://localhost:8000/

# Changes of this version vs. original voice chat bot:

1. UI change from separate gradio app to html and javascript mounted to FastAPI app

2. API change:
  - adding prompt for llm to return json for special cases
  - function call routing if response returned from llm is certain json format
  - return a JSONResponse instead of a FileResponse, so more information is returned and response audio file is no longer directly sent back but stored on server with an accessible URI

3. Dislay user question and bot answer in text on the right

4. Missing the live record functionality as that is built-in support by gradio Audio component, but with html/javascript will need to implement separately