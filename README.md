# Llama-3.1-8B Fine-Tuning with Unsloth + TRL
PubMedQA-Ko 데이터셋 기반 LoRA Instruction Fine-Tuning 프로젝트

## 1. 프로젝트 개요 (Project Overview)

이 프로젝트는 Meta-Llama-3.1-8B-Instruct 모델을 Unsloth(4bit LoRA) 및 Hugging Face TRL(SFTTrainer) 기반으로 파인튜닝하여 한글 의료 질의응답 데이터셋(PubMedQA-test-Ko)에 특화된 서술 답변(Long Answer)을 생성하도록 학습(fine-tuning)하는 프로젝트입니다.

## 2. 데이터셋 (Dataset)

 1) Name: ChuGyouk/PubMedQA-test-Ko

 2) Type: 한국어 번역 버전의 PubMedQA

 3) Structure:
 
  (1) QUESTION :	의학적 질문
  
  (2) CONTEXTS : 관련 논문 요약(Abstract)
  
  (3) LONG_ANSWER : 논문 내용 기반 서술형 답변

## 3. 주요구성

 1) Base model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit

 2) Framework: PyTorch + Hugging Face Transformers

 3) Task: Text Generation (질문 + 문맥 → 답변 생성)

 4) Architecture:

  (1) Input: QUESTION + CONTEXTS

  (2) Output: LONG_ANSWER (자연어 문장)

  (3) Fine-tuning scope: 상위 7개 Transformer layer + LM head

 5) 학습 방법 : LoRA (Low-Rank Adaptation) + Supervised Fine-Tuning (SFT)
 
``` python
   # meta tensor materialize
   model = FastLanguageModel.for_training(model)

   # LoRA 설정
   model = FastLanguageModel.get_peft_model(
       model,
       r=16,
       target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
       lora_alpha=16,
       lora_dropout=0,
       bias="none",
       use_gradient_checkpointing="unsloth",
       random_state=3407,)
   model = model.to("cuda")
```

 6) 환경 설정 
  ``` python
   !pip install -U "unsloth>=0.8.8" "trl>=0.9.4" "transformers>=4.44.0" "accelerate>=0.33.0" "bitsandbytes>=0.43.1" "datasets" "scikit-learn"
   !pip uninstall -y peft && pip install "peft>=0.11.1"
```
☞ 하위층은 언어 일반 지식 유지, 상위층만 의학 도메인에 맞게 미세조정

## 4. 학습 설정(Training Configuration)
 1) Epochs : 3
 
 2) Learning rate :	1e-5
 
 3) Optimizer :	AdamW
 
 4) Batch size : 1
 
 5) Precision : bfloat16
 
 6) Framework	PyTorch 2.x / Transformers 4.40+

## 5. WorkFlow 

 1) 모델 로드 및 LoRA 구성
 
 2) 데이터 전처리 (Alpaca 포맷 변환)
 
 3) 학습 설정 및 Fine-Tuning
    
  ```python 
   from trl import SFTTrainer
   from transformers import TrainingArguments
   
   training_args = TrainingArguments(
       output_dir="outputs",
       per_device_train_batch_size=3,
       gradient_accumulation_steps=4,
       warmup_steps=5,
       max_steps=60,
       learning_rate=2e-4,
       fp16=True,
       eval_strategy="steps",
       eval_steps=10,
       report_to="none",
       remove_unused_columns=False)
   
   trainer = SFTTrainer(
       model=model,
       tokenizer=tokenizer,
       train_dataset=train_ds,
       eval_dataset=valid_ds,
       dataset_text_field="text",
       max_seq_length=512,
       dataset_num_proc=2,
       packing=False,
       args=training_args)
   
   trainer_stats = trainer.train()
 ```

 4) 모델 저장
```python 
  # LoRA 어댑터 저장
model.save_pretrained("lora_model_llama3")
tokenizer.save_pretrained("lora_model_llama3")
```

 5) 테스트 평가 및 추론
 ```python 
   # test_ds에서 text 컬럼만 사용
   test_ds = test_ds.select_columns(["text"])
   trainer.args.remove_unused_columns = False
   
   eval_results = trainer.evaluate(test_ds)
   print("\n📊 Test Evaluation Results:")
   print(eval_results)
   
   # 샘플 추론
   prompt = "폐암 환자의 통증 관리를 위해 가장 중요한 평가 항목은 무엇인가요?"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
   outputs = model.generate(**inputs, max_new_tokens=200)
   print(tokenizer.decode(outputs[0], skip_special_tokens=True))
 ```
 6) 분석 & 시각화
 ```python 
 # 1. 데이터 로드
from datasets import load_dataset
ds = load_dataset("ChuGyouk/PubMedQA-test-Ko")

# 2. 학습 실행
python train_pubmedqa_exaone.py 

## training 
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
num_epochs = 3
total_steps = len(train_loader) * num_epochs
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=int(0.1 * total_steps),
    num_training_steps=total_steps
)

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

```
## 6. Summary & Research Plan
본 데이터는 한국어 의학 텍스트 기반 질의응답(QA) 데이터셋을 EXAONE-3.5-2.4B를 PubMedQA-Ko에 특화시켜 “의료 데이터 + AI + 품질 관리” 교차점에서 임상연구 데이터 분석 및 응답 발현 시스템을 구현하였습니다.
향후 DCT(분산형 임상연구), eCRF, 임상문서 생성 및 검증 AI 시스템으로 개발로의 확장을 다음과 같이 제언합니다. 

 ### 1) DCT 분산형 임상 연구를 위한 증상 모니터링: 환자 자가 보고용 triage 분류 
   (1) 단계 
      - 증상 모니터링 자동 triage
        : 환자 자가 보고 데이터 기반 이상 증상 자동 분류
      - AE 노트 - triage + 코딩 Proof of Concept
        : CDASH/SDTM 표준 기준에 기반하여 의학 용어 매핑 및 품질 검증 
      - 규정 RAG + 검증 + 문서생성 
        : 규정에 준수하여 환자 증상 모니터링 및 보고, EHR 작성 후, 자동 검증 시스템 구축 
        
** Author ** 
양 소 라 | RN, BSN, MSN

**참고 문헌**
딥러닝 전이학습(Transfer Learning)과 파인튜닝(Fine Tuning) → https://hi-ai0913.tistory.com/32
GPT-2를 이용한 챗봇 파인튜닝 → https://wikidocs.net/217620
Hu et al., LoRA: Low-Rank Adaptation of Large Language Models (ICLR 2022)
Ouyang et al., InstructGPT: Training language models to follow instructions with human feedback (2022)
Unsloth Docs — Efficient LoRA Training
Hugging Face TRL Docs
