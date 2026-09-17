# how to run
docker compose up -d
"# blotter_chat" 

<!-- how to run manually in virtual env -->
uv run python -m mcp_server.mcp_web_app
uv run uvicorn src_mcp.app:app --port 9999
docker run -d -p 8050:8080 --add-host=host.docker.internal:host-gateway -v open-webui-data:/app/backend/data --name open-webui --restart always ghcr.io/open-webui/open-webui:main