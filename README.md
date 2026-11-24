# satisrag

Asistente conversacional para jugadores nuevos de Satisfactory. Es un proyecto personal de práctica para llevar un sistema RAG de cero a producción.

El diseño completo está en el [plan maestro](Satisfactory-rag-assistant-master-plan.md).

## Puesta en marcha

Requisito: [uv](https://docs.astral.sh/uv/). La versión de Python la instala uv.

```bash
uv sync
uv run pre-commit install
```

## Controles de calidad

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy
uv run pytest
```

Pre-commit ejecuta los tres primeros en cada commit. El CI ejecuta los cuatro en cada PR.

## Evaluación

```bash
uv run python -m eval.run_eval --dry-run
uv run python -m eval.run_eval --retriever modulo:objeto --note "qué cambió"
```

El primero valida el dataset y cuenta las preguntas. El segundo mide un retriever y añade una entrada a `docs/eval-log.md`; con `--no-log` no la escribe.

## Datos del juego

Los datos del juego no se distribuyen en este repositorio. Copia `en-US.json` y `es-ES.json` desde `CommunityResources/Docs/`, en la carpeta de instalación de Satisfactory, a `data/raw/`.

## Convenciones

- Una rama por historia: `feat/US0-3-1-bootstrap-project`.
- Conventional Commits con el ID de la historia: `feat(eval): add schema [US0.2.1]`.
- Código y commits en inglés; documentación en español.
