# OS_AI_Frontier
This project consists of a multi-step analytical pipeline that combines structural topic modeling and LLM categorization (using Llama 3.1 70B).

.
├── README.md
├── requirements.txt        Python packages
├── .gitignore
├── data/
│   └── dataor.xlsx         review data (see "Data" below)
├── R/
│   ├── newOS.R          structural topic model (STM), K = 40
│   ├── diagOS.R                correlations between codes and topics
│   └── metaparam.R   comparison of topic modelling and categorizations
├── python/
│   ├── classifier.py           LLM classification, codes c1-c6
│   ├── classifierBOK.py         LLM classification, codes k1-k6
│   ├── metapipeline.py      three-agent interpretation of the STM results
│   ├── MetaDiag.py     three-agent interpretation of the correlations
│   └── trainingRoBERTa.py        three RoBERTa rating models
└── outputs/
    ├── input.txt           STM report: FREX words and covariate effects
    ├── output.txt          three-agent report on input.txt
    ├── diag.txt            correlation summary (codes and topics)
    └── diagout.txt         three-agent report on diag.txt

    # OS_AI_Frontier

<!-- Replace this line with your project description. -->


## Requirements

**R** (4.2 or later recommended). The scripts install missing packages on the
first run: `stm`, `readxl`, `openxlsx`, `stringi`, `tm`, `SnowballC`, `dplyr`,
`tidyr`, `ggplot2`, `viridis`, `scales`, `patchwork`.

**Python** (3.9 or later recommended):

```
pip install -r requirements.txt
```

`train_roberta.py` downloads `roberta-base` on first use and runs much faster on
a GPU. It was written for a specific version of `transformers`; if you see an
error about `evaluation_strategy` or `tokenizer`, install the version listed in
`requirements.txt`.


**Always run every script from the project root folder** (the folder that
contains `data/`, `R/`, `python/` and `outputs/`). All paths are relative to it.

In R or RStudio, set the working directory to the project root, then:

r
source("R/stm_analysis.R")


In a terminal, from the project root:


python python/classifier.py


### Order of the analysis

| Step | Script | Reads | Writes |
|------|--------|-------|--------|
| 1 | `R/newOS.R` | `data/dataor.xlsx` | topic columns and a `words` sheet in `data/dataor.xlsx`; `outputs/input.txt` and `outputs/input.xlsx` |
| 2 | `python/classifier.py` | `data/dataor.xlsx` | columns `c1`-`c6` in `data/dataor.xlsx` |
| 3 | `python/classifierBOK.py` | `data/dataor.xlsx` | columns `k1`-`k6` in `data/dataor.xlsx` |
| 4 | `R/diagOS.R` | `data/dataor.xlsx` | `outputs/summary_correlations.txt` (the file `outputs/diag.txt` in this repository is this file, renamed) |
| 5 | `python/metapipeline.py` | `outputs/input.txt` | `outputs/output.txt` and `outputs/out.json` |
| 6 | `python/MetaDiag.py` | `outputs/diag.txt` | `outputs/diagout.txt` and `outputs/diagout.json` |
| 7 | `R/MetaParam.R` | `data/dataor.xlsx` | plots and CSV files in `outputs/Analysis_Results/` |
| 8 | `python/trainingRoBERTa.py` | `data/dataor.xlsx` | models and predictions in `outputs/roberta_out_version3/` |

Notes:

- Scripts 1-3 **modify `data/dataor.xlsx` in place**. Keep a copy of the
  original file before running them. `stm_analysis.R` also saves
  `data/dataor_backup.xlsx` the first time it runs.
- Step 6 reads `outputs/diag.txt`. If you rerun step 4, copy or rename its
  output `outputs/summary_correlations.txt` to `outputs/diag.txt`, or pass the
  path directly:
  `python python/three_agent_diag.py outputs/summary_correlations.txt`
- Step 7 needs the columns `i1`-`i6`, `cfit` and `kfit`. (cfit and kfit indicate a value of zero when a classifier did not capture any concept)
- You can run the steps in a different order only if the input columns and
  files listed above already exist.
- Only the four text reports in `outputs/` are stored in this repository.
  The other files in the table (Excel, JSON, plots, models) are created when
  you run the scripts.

## Language model access (steps 2, 3, 5, 6)

These scripts send requests to a Llama 3.1 70B Instruct model served through an
OpenAI-compatible (vLLM) chat completions endpoint. The model used in the study
ran on an institutional computing cluster that is not publicly available. To
rerun these steps you need your own access to any compatible endpoint serving a
Llama 3.1 70B Instruct model. Results may differ with other models or settings.

Set these environment variables before running (never write keys into the
scripts and never upload them):


macOS / Linux:

```
export LLM_URL="http://127.0.0.1:8000/v1/chat/completions"
export LLM_API_KEY="your_key_here"
```

Windows (Command Prompt, then reopen the window):

```
setx LLM_URL "http://127.0.0.1:8000/v1/chat/completions"
setx LLM_API_KEY "your_key_here"
```

If you cannot access the model, you can still reproduce steps 1, 4 and 7 from
`data/dataor.xlsx` (which already contains the labels), and read the original
results in `outputs/`.

## Outputs

`outputs/` contains the final text reports of the original run:

- `input.txt`: topics with their FREX words and the effects of score and period
  on topic prevalence (from `stm_analysis.R`).
- `output.txt`: Ip, If and Ma agent interpretation of `input.txt`.
- `diag.txt`: correlations among the codes c1-c6 and between the codes and the
  topics (from `diagOS.R`).
- `diagout.txt`: Ip, If and Ma agent interpretation of `diag.txt`.

## Reproducibility notes

- STM and the train/validation/test split use fixed seeds (12345 and 42).
- LLM outputs can vary between runs and servers even at low temperature.
- The classification scripts resume from existing labels. Set `RERUN_ALL = True`
  in the script to relabel everything.

## Data



## License and citation

Chalmers Tekniska Högskola (c)
