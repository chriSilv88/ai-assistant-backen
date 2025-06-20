# assistant-backend-fastapi

## Quick Start

Install dependencies  
Make sure [Poetry](https://python-poetry.org/) is installed.

```bash
poetry install
```

Start the server  
By default, the `LLMOrchestrator` will be used.  
You can switch between different handlers by setting the `POE_HANDLER` environment variable.  
Available options: `default_chat`, `conversation`, `conversation_retrieval`.  
See `__main__.py` for more info.

```bash
make start
```

Send a sample request  
You can use this `curl` command to test the server locally:

```bash
curl -X 'POST' \
  'http://0.0.0.0:8080/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "version": "1.0",
    "type": "query",
    "query": [
      {
        "message_id": "1",
        "role": "system",
        "content": "You are a helpful assistant.",
        "content_type": "text/markdown",
        "timestamp": 1678299819427621,
        "feedback": []
      },
      {
        "message_id": "2",
        "role": "user",
        "content": "What is the capital of Nepal?",
        "content_type": "text/markdown",
        "timestamp": 1678299819427621,
        "feedback": []
      }
    ],
    "user_id": "u-1234abcd5678efgh",
    "conversation_id": "c-jklm9012nopq3456",
    "message_id": "2"
  }' -N
```
