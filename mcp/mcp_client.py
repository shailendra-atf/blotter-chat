import asyncio
import json
import httpx

BASE_URL = "http://127.0.0.1:8090"

async def call_mcp_api():
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Step 1: Open SSE connection to retrieve the session message URL
        async with client.stream("GET", f"{BASE_URL}/sse") as response:
            message_endpoint = None
            
            # Read lines from stream to capture the session URI
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    # SSE event payload contains the target endpoint for POST messages
                    message_endpoint = line.replace("data:", "").strip()
                    print(f"[Connected] Message Endpoint URI: {message_endpoint}")
                    break

            if not message_endpoint:
                raise RuntimeError("Failed to establish SSE connection or retrieve message endpoint.")

            target_post_url = f"{BASE_URL}{message_endpoint}" if message_endpoint.startswith("/") else message_endpoint

            # Step 2: Send JSON-RPC Initialization Handshake
            init_payload = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "python-api-client", "version": "1.0.0"}
                }
            }
            await client.post(target_post_url, json=init_payload)

            # Send initialized notification
            await client.post(target_post_url, json={
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            })

            # Step 3: Call an MCP Tool (e.g., execute_read_query)
            tool_call_payload = {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "find_relevant_tables",
                    "arguments": {
                        "user_intent": "fetch user profiles and financial revenue"
                    }
                }
            }
            
            post_response = await client.post(target_post_url, json=tool_call_payload)
            print(f"[Tool Call Request Sent] HTTP Status: {post_response.status_code}")

            # Step 4: Listen on the SSE Stream for the JSON-RPC execution result
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    data_str = line.replace("data:", "").strip()
                    data_json = json.loads(data_str)
                    
                    # Look for the response matching our request ID (id: 2)
                    if data_json.get("id") == 2:
                        print("\n=== MCP Tool Result Received ===")
                        print(json.dumps(data_json, indent=2))
                        break

if __name__ == "__main__":
    asyncio.run(call_mcp_api())