import uvicorn

from subspace.app import create_app
from subspace.settings import SubspaceSettings


def main():
    settings = SubspaceSettings()
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
