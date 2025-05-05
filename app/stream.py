def stream_response(response):
    for chunk in response:
        if chunk.text:
            yield chunk.text