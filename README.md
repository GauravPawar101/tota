#              TOTA
# 🌐 English-Hindi Neural Machine Translation (NMT) 

This project implements an English ↔ Hindi translator using a Transformer-based sequence-to-sequence model (such as `mBART`, `T5`, or a custom Transformer). It supports training from scratch or fine-tuning a pre-trained model.

---

## 🚀 Features

- 🔤 Translation from English to Hindi 
- 🧠 Transformer-based sequence-to-sequence model
- 📦 HuggingFace `transformers` integration
- 💾 Supports custom datasets (parallel sentence pairs)

---

---

## 🧰 Requirements

Install dependencies with:

```bash
pip install transformers pytorch

Main packages:

transformers

accelerate (optional for multi-GPU)

🔧 Configuration
Edit parameters in train.py or use command-line args:

bash
Copy
Edit
--model_name_or_path facebook/mbart-large-50-many-to-many-mmt
--source_lang en_XX
--target_lang hi_IN
--train_file data/train.csv
--validation_file data/valid.csv
--output_dir models/translator
--per_device_train_batch_size 8
--num_train_epochs 5
🏁 Training
Fine-tune a pre-trained model:
bash
Copy
Edit
python scripts/train.py \
  --model_name_or_path facebook/mbart-large-50-many-to-many-mmt \
  --source_lang en_XX \
  --target_lang hi_IN \
  --train_file data/train.csv \
  --validation_file data/valid.csv \
  --output_dir models/translator
Training from scratch (not recommended unless you have >10M samples):
bash
Copy
Edit
python scripts/train.py --do_train --config train_config.yaml
🔍 Evaluation
bash
Copy
Edit
python scripts/evaluate.py --model_dir models/translator --test_file data/test.csv
Outputs BLEU score and sample predictions.

🧪 Inference
bash
Copy
Edit
python scripts/translate.py --model_dir models/translator --input "How are you?"
Expected Output:

Copy
Edit
तुम कैसे हो?
📊 Example Dataset Format (CSV)
csv
Copy
Edit
source,target
Hello,नमस्ते
I am learning.,मैं सीख रहा हूँ।
