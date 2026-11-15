# Data

## Source

Both datasets come unchanged from the IndoNLU benchmark repository
(https://github.com/IndoNLP/indonlu, `dataset/` folder), which is released under the
Apache License 2.0. A copy of that license is in `LICENSE-IndoNLU-Apache-2.0.txt`.

| Folder | Original path in IndoNLU | Original resource |
|---|---|---|
| `raw/smsa/` | `dataset/smsa_doc-sentiment-prosa/{train,valid,test}_preprocess.tsv` | Purwarianti & Crisdayanti (2019) |
| `raw/emot/` | `dataset/emot_emotion-twitter/{train,valid,test}_preprocess.csv` | Saputri, Mahendra & Adriani (2018) |

Downloaded on 2026-11-10. The masked-label test files from IndoNLU are not included because
the labelled test files are available.

## Modifications

`processed/` contains the same rows in the same order, converted to CSV with columns
`id,text,label`. The only changes are: (1) an added `id` column (`<task>-<split>-<row index>`),
and (2) the EmoT text column renamed from `tweet` to `text`. No text was cleaned, filtered,
or relabelled. Notes from verification (notebook 00): SmSA train contains 67 exact duplicate
rows and EmoT train 6, as in the original release; one EmoT test text also appears in EmoT train.

## Integrity

`SHA256SUMS` lists checksums for every file; notebook 00 verifies them.
