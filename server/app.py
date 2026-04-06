from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

from .leadqualenv_environment import LeadQualOpenEnv
from .models import LeadQualActionModel, LeadQualObservationModel

os.environ.setdefault(
    "LEADQUALENV_USE_LLM",
    "1" if os.getenv("GROQ_API_KEY") else "0"
)

HTTP_ENV = LeadQualOpenEnv()

MAX_CONCURRENT = int(os.getenv("LEADQUALENV_MAX_CONCURRENT", "4"))

app = create_app(
    lambda: LeadQualOpenEnv(),
    LeadQualActionModel,
    LeadQualObservationModel,
    env_name="leadqualenv",
    max_concurrent_envs=MAX_CONCURRENT,
)


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
