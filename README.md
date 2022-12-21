# edos
Explainable Detection of Online Sexism

Competition link: https://codalab.lisn.upsaclay.fr/competitions/7124#learn_the_details

# Setup
Mac M1: [M1](./envs/m1.md)

# Runs
* Bertweet: `!PYTHONPATH=. python src/runner/classify.py --device cuda --config bertweet.json`
* UnifiedQA: `!PYTHONPATH=. python src/runner/classify.py --device cuda --config unifiedQA.json`
* Generate submission: `!PYTHONPATH=. python src/runner/generate_submission.py --device cuda --config <config>.json`

