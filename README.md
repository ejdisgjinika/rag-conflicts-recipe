# Contextual Knowledge Conflicts in RAG Systems with a Recipe Dataset

Experiment runner and processed recipe data for the
[associated paper](https://doi.org/10.1007/978-3-032-40774-0_24).

## Setup

Requires **Python 3.12**, [uv](https://docs.astral.sh/uv/) and
[Ollama](https://ollama.com/). Run commands from the repository root.

```bash
uv sync
uv run python scripts/setup_nltk.py
```

Start Ollama in a separate terminal if it is not already running:

```bash
ollama serve
```

## Run

Run the twelve-model [configs/paper.yaml](configs/paper.yaml):

```bash
uv run python main.py --output-dir results/full
```

This configuration runs **400 cases per model**. Select `cpu` or `cuda` for
embeddings through `models.device` in the configuration file.

Select models and experiments in the YAML configuration:

| `experiments` value | Operation |
| --- | --- |
| `rag` | Generate with oracle ingredient context and score against recipe variants |
| `llm_only` | Generate without context and score against recipe variants |
| `rag_vs_llm` | Compare the responses generated with and without context |

Run `uv run python main.py --help` for command-line options.

## Data and outputs

Input CSVs are included in `data/`. See [data/README.md](data/README.md) for the
schema and source. Data paths in the configuration are relative to the working directory.

The runner writes JSON files under `--output-dir`:

| File | Contents |
| --- | --- |
| `<model>/<experiment>/results_<timestamp>.json` | Queries, responses, references and ingredient-set Jaccard scores |
| `<model>/rag/retrieval_trace_<timestamp>.json` | Oracle context supplied for each case |
| `summary_<timestamp>.json` | Results across all configured models |
| `provenance_<timestamp>.json` | Input hashes, environment and evaluation settings |

Generated outputs are ignored by Git.

## Citation

```bibtex
@inproceedings{gjinika2027contextual,
  author    = {Gjinika, Ejdis and
                   Putelli, Luca and
                   Chiari, Mattia and
                   Pagani, Alessandro and
                   Serina, Ivan and
                   Gerevini, Alfonso Emilio},
  title     = {Contextual Knowledge Conflicts in {RAG} Systems with a Recipe Dataset},
  booktitle = {{AIxIA} 2026 -- Advances in Artificial Intelligence},
  editor    = {Bistarelli, S. and Maratea, M. and Santini, F.},
  series    = {Lecture Notes in Computer Science},
  volume    = {17131},
  year      = {2027},
  publisher = {Springer},
  address   = {Cham},
  doi       = {10.1007/978-3-032-40774-0_24},
  url       = {https://doi.org/10.1007/978-3-032-40774-0_24}
}
```

Citation metadata is also available in [CITATION.cff](CITATION.cff).

## Licence

Code: [MIT](LICENSE). Data: [DATA_LICENSE.md](DATA_LICENSE.md).
