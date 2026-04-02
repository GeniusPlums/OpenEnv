from __future__ import annotations

from openenv.core.env_server.http_server import create_app

from .leadqualenv_environment import LeadQualOpenEnv
from .models import LeadQualActionModel, LeadQualObservationModel


HTTP_ENV = LeadQualOpenEnv()


app = create_app(
    lambda: HTTP_ENV,
    LeadQualActionModel,
    LeadQualObservationModel,
    env_name="leadqualenv",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
