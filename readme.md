# Estimating Discrete Latent Variable Models Using Amortized Variational Inference

This repository contains all our code used to estimate **LCA**, **GDINA**, and **Mixture IRT** models using *Amortized Variational Inference*.

You can easily fit these models to your own data using the `fit_model.py` file.

---

## 🔧 Installation

Before fitting any models, make sure all dependencies are installed:

```bash
pip3 install -r requirements.txt
```

---

## 🚀 Running the Models

You can fit one of the following models from the command line:

### ✅ Latent Class Analysis (LCA)

```bash
python3 fit_model.py LCA [path_to_data] [number_of_classes]
```

### ✅ Generalized DINA Model (GDINA)

```bash
python3 fit_model.py GDINA [path_to_data] [number_of_attributes] [path_to_q_matrix]
```

### ✅ Mixture Item Response Theory Model (Mixture IRT)

```bash
python3 fit_model.py MIXIRT [path_to_data] [number_of_classes] [path_to_q_matrix]
```
---

#### 📌 Example

```bash
python3 fit_model.py MIXIRT ./data/NPI.csv 7 ./Qmatrices/QmatrixNPI.csv
```

## 📁 Output

All parameter estimates, including EAP latent variable estimates, will automatically be saved to:

```
./results/estimates/
```

---

## ⚙️ Configuration

By default, model fitting uses the configurations specified in:

```
./configs/fitconfig.yml
```

You can change settings like:

- Learning rate
- Batch size
- Number of importance-weighted samples
- Maximum/minimum epochs
- Early stopping criteria
- Etc

Just open `fitconfig.yml` and adjust the values to your needs.

---

