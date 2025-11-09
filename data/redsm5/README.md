---
license: apache-2.0
task_categories:
  - text-classification
  - text-generation
language:
  - en
tags:
  - dsm5
  - depression
  - reddit
  - clinical-nlp
  - rationale
  - explainability
pretty_name: ReDSM5
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: annotations
        path: "redsm5_annotations.csv"
      - split: posts
        path: "redsm5_posts.csv"
---

# ⚕️💬 ReDSM5: A Reddit Dataset for DSM-5 Depression Detection

> ℹ️ **Looking for a quick preview?**  
> A fully paraphrased, anonymized sample with 25 entries is [publicly available on the Hugging Face Hub](https://huggingface.co/datasets/irlab-udc/redsm5-sample) — no user agreement required!

## 🚦 Access Conditions

This dataset is gated. To obtain access, please complete the access request form at [ReDSM5 Agreement Form](https://www.irlab.org/ReDSM5_agreement.odt) and submit it via email to [eliseo.bao@udc.es](mailto:eliseo.bao@udc.es). Your request will be reviewed and you’ll receive approval or further instructions by email.

## 📝 Dataset Summary

**ReDSM5** is a corpus of 1,484 Reddit posts, each *sentence-level annotated* for presence or absence of the nine DSM-5 major depressive episode symptoms (or `SPECIAL_CASE` for expert discrimination cases), along with a clinical rationale from a licensed psychologist.

A **public paraphrased sample** is available for inspection and prototyping at [irlab-udc/redsm5-sample](https://huggingface.co/datasets/irlab-udc/redsm5-sample).  
This sample includes 25 entries, has been completely paraphrased to protect anonymity, and can be accessed without any agreement.

Unlike other datasets, **every annotation** includes:
- **`sentence_text`**: The relevant sentence from a Reddit post.
- **`DSM5_symptom`**: One of:
  - `DEPRESSED_MOOD`
  - `ANHEDONIA`
  - `APPETITE_CHANGE`
  - `SLEEP_ISSUES`
  - `PSYCHOMOTOR`
  - `FATIGUE`
  - `WORTHLESSNESS`
  - `COGNITIVE_ISSUES`
  - `SUICIDAL_THOUGHTS`
  - `SPECIAL_CASE` (for non-DSM-5 clinical/positive discriminations)
- **`status`**:  
  - `1` = relevant evidence for the symptom  
  - `0` = explicit negative (clinician-annotated absence of symptom)
- **`explanation`**: Short clinical rationale.
- **`post_id`, `sentence_id`**: For reference and grouping.

There is also a **full post file** with the original post text.

### 📁 Files

- `redsm5_posts.csv`  
  - Columns: `post_id`, `text` (cleaned full post)
- `redsm5_annotations.csv`  
  - Columns: `post_id`, `sentence_id`, `sentence_text`, `DSM5_symptom`, `status`, `explanation`

#### Example annotation row:

| post_id      | sentence_id        | sentence_text                                    | DSM5_symptom   | status | explanation                                                           |
|--------------|-------------------|--------------------------------------------------|---------------|--------|-----------------------------------------------------------------------|
| s_3019_239   | s_3019_239_1      | However my sex drive went crazy...               | SPECIAL_CASE   | 1      | The above statement allows us to affirm that the person...            |
| s_427_110    | s_427_110_1       | I'm feeling pretty proud of myself...            | SPECIAL_CASE   | 1      | The person who writes this post is manifesting positive feelings...   |
| s_221_12     | s_221_12_5        | I have trouble sleeping every night              | SLEEP_ISSUES   | 1      | This statement shows persistent sleep issues matching DSM-5 criteria. |

## 📊 Dataset Statistics

- **Total posts**: 1,484  
- **Total number of expert explanations**: 1,547  
- **Average explanations per post**: 1.04  
- **Average number of symptoms per post (status=1 only)**: 1.39  
- **Number of hard negatives (posts with no symptoms, status=1)**: 392  
- **Average post length (in words)**: 294.7  
- **Min / Max post length (in words)**: 2 / 6,990  

#### Symptom Distribution (posts with `status=1`):

| Symptom                | Posts tagged |
|------------------------|-------------:|
| DEPRESSED_MOOD         | 328          |
| ANHEDONIA              | 124          |
| APPETITE_CHANGE        | 44           |
| SLEEP_ISSUES           | 102          |
| PSYCHOMOTOR            | 35           |
| FATIGUE                | 124          |
| WORTHLESSNESS          | 311          |
| COGNITIVE_ISSUES       | 59           |
| SUICIDAL_THOUGHTS      | 165          |
| **SPECIAL_CASE**       | 92           |

## 📦 File Format

### `redsm5_full_posts.csv`
| post_id    | text                                                                 |
|------------|----------------------------------------------------------------------|
| s_1001_1   | I feel tired every day. I can't concentrate and sleep is a mess...   |

### `redsm5_annotations_long.csv`
| post_id    | sentence_id      | sentence_text             | DSM5_symptom   | status | explanation                                |
|------------|------------------|---------------------------|----------------|--------|---------------------------------------------|
| s_1001_1   | s_1001_1_1       | I feel tired every day.   | FATIGUE        | 1      | Indicates chronic fatigue.                  |
| s_1001_1   | s_1001_1_3       | I can't concentrate...    | COGNITIVE_ISSUES | 1    | Persistent cognitive issues.                |

## 💡 Notes
- Each row in `redsm5_annotations_long.csv` is a unique `(post, sentence, symptom)` expert-annotated evidence.
- For multi-label tasks, group by `post_id`.
- For post-level text, use `redsm5_full_posts.csv`.

## 📝 Citation

This paper has been accepted as a Resource Paper at **CIKM 2025**. The official conference proceedings will be available soon. In the meantime, you can read the preprint on [arXiv](https://www.arxiv.org/abs/2508.03399):

```bibtex
@misc{bao2025redsm5,
  title        = {ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author       = {Eliseo Bao and Anxo Pérez and Javier Parapar},
  year         = {2025},
  eprint       = {2508.03399},
  archivePrefix= {arXiv},
  primaryClass = {cs.CL},
  url          = {https://arxiv.org/abs/2508.03399},
  note         = {Accepted at CIKM 2025}
}
```

## 📬 Contact

For questions, please reach out via email: `eliseo.bao@udc.es`