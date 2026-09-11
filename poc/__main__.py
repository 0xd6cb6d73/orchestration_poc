import uvicorn


def main() -> None:
    uvicorn.run("poc.api.app:app", host="0.0.0.0", port=8000, reload=False, workers=1)


if __name__ == "__main__":
    main()

