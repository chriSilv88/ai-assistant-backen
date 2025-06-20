.PHONY: start format try

## Format code with black and isort
format:
	poetry run black .
	poetry run isort .

## Start the FastAPI server
start:
	poetry run python -m langchain_template_poe_fastapi

## Test the API with a sample cURL query
try:
	curl -X POST http://0.0.0.0:8080/ \
		-H "accept: application/json" \
		-H "Content-Type: application/json" \
		-d '{ \
			"version": "1.0", \
			"type": "query", \
			"query": [ \
				{ \
					"message_id": "1", \
					"role": "system", \
					"content": "You are a helpful assistant.", \
					"content_type": "text/markdown", \
					"timestamp": 1678299819427621, \
					"feedback": [] \
				}, \
				{ \
					"message_id": "2", \
					"role": "user", \
					"content": "What is the capital of Nepal?", \
					"content_type": "text/markdown", \
					"timestamp": 1678299819427621, \
					"feedback": [] \
				} \
			], \
			"user_id": "u-1234abcd5678efgh", \
			"conversation_id": "c-jklm9012nopq3456", \
			"message_id": "2" \
		}' -N
