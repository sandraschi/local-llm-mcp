"""AWS Bedrock adapter — sigv4 signing, Converse API."""

import json
from typing import Any

from llm_mcp.gateway.base import BaseLLMAdapter, register_provider


@register_provider("bedrock")
class BedrockAdapter(BaseLLMAdapter):
    provider = "bedrock"
    base_url = ""
    api_key_env = ""

    def get_api_key(self, headers: dict[str, str]) -> str:
        return ""

    def map_params(self, body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages", [])
        system = ""
        bedrock_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system = msg.get("content", "")
            else:
                role = "assistant" if msg["role"] == "assistant" else "user"
                bedrock_messages.append({
                    "role": role,
                    "content": [{"text": msg.get("content", "")}],
                })

        mapped = {
            "modelId": body.get("model", ""),
            "messages": bedrock_messages,
            "inferenceConfig": {
                "maxTokens": body.get("max_tokens", 4096),
            },
        }
        if system:
            mapped["system"] = [{"text": system}]
        if "temperature" in body:
            mapped["inferenceConfig"]["temperature"] = body["temperature"]
        if "top_p" in body:
            mapped["inferenceConfig"]["topP"] = body["top_p"]
        return mapped

    def transform_response(self, resp_data: dict[str, Any], model: str) -> dict[str, Any]:
        output = resp_data.get("output", {})
        message = output.get("message", {})
        content = ""
        for c in message.get("content", []):
            if c.get("text"):
                content += c["text"]

        usage = resp_data.get("usage", {})
        return self._openai_chunk(
            model=model,
            choice={
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": output.get("stopReason", "stop"),
                "logprobs": None,
            },
            usage={
                "prompt_tokens": usage.get("inputTokens", 0),
                "completion_tokens": usage.get("outputTokens", 0),
                "total_tokens": usage.get("inputTokens", 0) + usage.get("outputTokens", 0),
            },
        )

    async def complete(self, body: dict[str, Any], headers: dict[str, Any]) -> dict[str, Any]:
        import os
        import aws4py  # pip install aws4py
        from aws4py import AWSRequest

        mapped = self.map_params(body)
        region = os.getenv("AWS_REGION", "us-east-1")
        host = f"bedrock-runtime.{region}.amazonaws.com"
        url = f"https://{host}/model/{mapped['modelId']}/converse"

        payload = json.dumps(mapped).encode()
        aws_req = AWSRequest(
            method="POST",
            url=url,
            data=payload,
            headers={"Content-Type": "application/json", "Host": host},
            service="bedrock",
            region=region,
        )
        signed = aws_req.sign()
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(url, data=payload, headers=dict(signed.headers))
            resp.raise_for_status()
            data = resp.json()

        return self.transform_response(data, body.get("model", ""))
