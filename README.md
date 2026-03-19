<div align="center">
  
# Open MCP Client

![CopilotKit-Banner](https://github.com/user-attachments/assets/8167c845-0381-45d9-ad1c-83f995d48290)
</div>

https://github.com/user-attachments/assets/364b6705-14d4-4e6d-bea7-fb9f12664fab


# Getting Started

## Set Up Environment Variables 
Create a `.env` file at the root of your project:



```sh
touch .env
```

Add the following to `.env`:


```sh
LANGSMITH_API_KEY=lsv2_...
OPENAI_API_KEY=sk-...
```

Next, navigate to the `agent` folder and create another `.env` file:



```sh
cd agent
touch .env
```

Add the following inside `agent/.env`:

```sh
OPENAI_API_KEY=sk-...
LANGSMITH_API_KEY=lsv2_...
OPENROUTER_API_KEY=sk-or-...  # Get from https://openrouter.ai/keys
```

## Set Up Poetry:

Poetry manages dependencies for the agent service. Install it with:


```sh
pip install poetry
```

Verify the installation by running:


```sh
poetry --version
```

## Development

For easier debugging, run the `frontend` and `agent` in separate terminals:


```bash
# Terminal 1 - Frontend
pnpm run dev-frontend

# Terminal 2 - Agent
pnpm run dev-agent
```

Alternatively, launch both services together:


```bash
pnpm run dev
```

Visit [http://localhost:3000](http://localhost:3000) in your browser to view the application.


## Architecture

The codebase is organized into two primary components:


- **Frontend** -  Handles the user interface.
- **Agent** - Manages the core functionality.

## License
Distributed under the MIT License. See LICENSE for more info.
