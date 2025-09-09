# EHR_FM

```
conda deactivate 
conda create --name MEDS python=3.12 
conda activate MEDS 
pip install -e .[jupyter] 

```

```
python3 -m src.tokenizer.run_tokenization
python3 -m src.train
```