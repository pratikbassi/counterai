# CounterAI Backend

Rails API backend for CounterAI. Serves the API used by the frontend.

## Architecture

Monolithic Rails app: REST API, ActiveRecord models, and business logic in one codebase. Frontend is a separate React app that talks to this API.

## Run

```bash
rails s
```

Server runs by default at `http://localhost:3000`.
