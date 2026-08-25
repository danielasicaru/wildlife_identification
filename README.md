# Wildlife Identification

Computer vision systems for automated wildlife monitoring from camera trap imagery — dataset
analysis, augmentation strategy, model training and evaluation, and production-shaped inference
serving.

## Projects

### [camera-trap-classifier](projects/camera-trap-classifier)

Species identification pipeline for camera trap images deployed across multiple national park
sites: dataset characterization and quality analysis, an augmentation strategy grounded in
real dataset gaps, animal localization, a controlled multi-backbone classifier comparison,
evaluation, and a production-shaped FastAPI inference service.

Start there for the full write-up — problem, approach, setup instructions, and findings.

## Stack

Python, PyTorch, MegaDetector, FastAPI, MLflow, Docker.

## Repository conventions

- `projects/<name>/` — one self-contained, independently runnable deliverable per project, each
  documented as problem -> approach -> results -> tradeoffs in its own README.
- Prefer small, focused files over large ones that do too much.
- Follow existing patterns in a project before introducing new ones.
- Commit messages, pull request descriptions, code comments, and README content are written in
  first person, as the repository author.
