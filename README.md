How to run:
In server/src, run
```bash
uvicorn server.src.main:app --reload
```

In web, run
```bash
npm i
npm run start
```

example test logs in example_test_logs.txt

Changes made:
* Updated arxiv response
* Add logging for test logs
* Add frontend UI
* Change to use ollama cloud model instead
* Added prompts for LLM to use tools
* Perform TTS instead of just using text queries
* Changed tool usage format to fit with ollama's tool_calls