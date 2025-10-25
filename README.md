# Llama-3.1-8B Fine-Tuning with Unsloth + TRL
	PubMedQA-Ko 데이터셋 기반 LoRA Instruction Fine-Tuning 프로젝트

## 1. 프로젝트 개요 (Project Overview)

이 프로젝트는 Meta-Llama-3.1-8B-Instruct (4bit, via Unsloth) 모델을 한국어 의학 QA 데이터셋인 PubMedQA-test-Ko로 파인튜닝(fine-tuning)하여 의료 관련 QA 품질 향상과 도메인 적응을 실험 및 검증한 모델링입니다.


## 2. 데이터셋 (Dataset)

	  1) Name: ChuGyouk/PubMedQA-test-Ko
	
	  2) Type: 한국어 번역 버전 PubMedQA
	
	  3) Structure:
	 
		  (1) QUESTION :	의학적 질문
		  
		  (2) CONTEXTS : 논문 요약(Abstract)
		  
		  (3) LONG_ANSWER : 논문 내용 기반 서술형 답변
	
	  4) test dataset을 저장 후, train/validation 분할 

## 3. 모델 구성

	  1) Base model : unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
	
	  2) Framework : PyTorch + TRL (SFTTrainer) + Unsloth
	
	  3) Task : Text Generation (질문 + 문맥 → 답변 생성)
	
	  4) LoRA 설정 : r=16, α=16, dropout=0
	
	  5) Fine-tuning Layer : Q/K/V/Proj 계열 Layer
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
	  6) Architecture :
	
		  (1) Input: QUESTION + CONTEXTS
		
		  (2) Output: LONG_ANSWER (자연어 문장)


## 4. 학습 설정 (Training Configuration)

	  1) Epochs : 2
	 
	  2) Learning rate : 2e-4
	 
	  3) Optimizer : AdamW 
	 
	  4) Scheduler : Linear Warmup
	 
	  5) Batch size : 3 (gradient_accum 4 → total 12)
	 
	  6) Max steps : 60
	 
	  7) Precision : bf16 (지원 시)
	 
	  8) Stability : Triton, CUDA Graph 비활성화

## 5. WorkFlow 

	  1) 데이터/모델 로드 및 LoRA 구성
	
```python
  from datasets import load_dataset
  ds = load_dataset("ChuGyouk/PubMedQA-test-Ko")
  df = ds["test"].to_pandas()
  df["input"] = "질문: " + df["QUESTION"]
  df["output"] = df["LONG_ANSWER"]

  from unsloth import FastLanguageModel
  model, tokenizer = FastLanguageModel.from_pretrained(
  "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
	      max_seq_length=512,
	      dtype=None,
	      load_in_4bit=True,
	      device_map=None)
	
   model = FastLanguageModel.for_training(model)
```
	
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
   FastLanguageModel.for_inference(model)
   model.save_pretrained("lora_model_llama3")
   tokenizer.save_pretrained("lora_model_llama3")
   model.save_pretrained_merged("lora_model_llama3_merged", tokenizer, save_method="merged_16bit")
```
	
	  5) 검증 및 테스트
	 
```python 
   from unsloth import FastLanguageModel
   import torch
   
   model, tokenizer = FastLanguageModel.from_pretrained(
			  "lora_model_llama3_merged",
    		  max_seq_length=512,
		      dtype=torch.float16,
		      load_in_4bit=False,
		      device_map=None)
		   
   prompt = "배변 시 핸드폰 사용 습관이 직장암에 미치는 영향은?"
   inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
   with torch.no_grad():
   outputs = model.generate(**inputs, max_new_tokens=128)
   print("💬 모델 응답:", tokenizer.decode(outputs[0], skip_special_tokens=True))
``` 
   
## 6. 결과
	  1) 정량적/정성적 성능 평가
		  (1) Validation loss 1.24 → Test loss 1.21로 안정 수렴
		
		  (2) 정성적 품질 : Instruction 기반 응답의 일관성 향상, 의료용어 표현 자연스러움
		  
		  (3) 결과 특징 : PubMedQA-Ko 데이터에서 임상 질문 이해력 향상(의학적 정확도와 문체 일관성 향상)
		
		  (4) Error 사례 : 긴 문장에서 문단 전환이 부자연스러움 → max_length=512 제한 영향(용량문제로 제한 함)
	
	  
	  2) 테스트 결과
          (1) 의학적 문체 및 논문식 서술 문체로 자연스러운 응답 생성
![img:Resul Answer](URL:https://github.com/seirah-yang/LLM_finetuning/blob/main/LLM_answer.png)

## 7. Summary & Research Plan

본 연구는 한국어 의학 텍스트 기반 질의응답(QA) 데이터셋인 PubMedQA-Ko를 활용하여 Meta-Llama-3.1-8B-Instruct (LoRA 4bit) 모델을 파인튜닝함으로써, “의료 데이터 + AI + 품질 관리”의 교차점에서 임상연구 데이터 분석 및 응답 발현 시스템을 구현하였습니다.
	
이를 통해 모델은 의학적 질문의 이해력, 문체의 일관성, 의료용어의 자연스러운 사용 능력을 향상시켰으며, 향후 분산형 임상연구(DCT), eCRF 데이터 품질검증, 임상문서 생성 및 자동검증 AI 시스템으로의 확장하는 것을 제언합니다. 

	1) DCT 분산형 임상 연구를 위한 증상 모니터링: 환자 자가 보고용 triage 분류 
	  	(1) 단계 
	    
		    ① 증상 모니터링 자동 triage
		    
		      : 환자 자가 보고 데이터 기반 이상 증상 자동 분류
		    
		    ② AE 노트 - triage + 코딩 Proof of Concept
		    
		      : CDASH/SDTM 표준 기준에 기반하여 의학 용어 매핑 및 품질 검증 
		    
		    ③ 규정 RAG + 검증 + 문서생성 
		    
		      : 규정에 준수하여 환자 증상 모니터링 및 보고, EHR 작성 후, 자동 검증 시스템 구축

  	2) 확장 및 활용 방안
		(1) 의료 QA 자동화 
	     	- 임상연구 간호사(CRC/CRA) 및 연구자 대상 의료 질의응답 시스템 구축
	     	- PubMedQA Fine-tuning 결과 기반 임상시험 프로토콜, SAE 보고서 작성 및 질의응답 자동화
	
		(2) DCT 기반 모니터링
	   		- 환자 자가 보고형 증상 데이터 triage 시스템 구축
	      	- 환자가 모바일 또는 웨어러블을 통해 보고한 증상을 자동 분류
			- 이상 증상(AE) 발생 시 즉시 알림 및 의료진 검토 지원
	   	
		(3) eCRF 연계
	   		- CDASH/SDTM 표준 포맷 매핑 및 AI 기반 질의 생성
	    	- 환자 보고 데이터나 모니터링 로그 eCRF 구조로 변환
			- EHR의 이상치 또는 결측치 자동 탐지 및 질의 자동 생성
	    
		(4) 문서 품질관리
	        - 행정/R&D 문서 자동 검증 및 참고문헌 링크 제시 시스템
	        - 규정 기반 RAG과 NLI 기반 검증기를 연계
			- 문서의 포맷·규정·정합성을 평가 및 참조 문헌 자동 링크
	
	    (5) 도메인 확장
			- Oncology, Pharmacovigilance, CDISC 표준 기반 학습 확장
	      	- 암환자 증상 데이터, 약물 부작용 모니터링
			- 의학 통계 분석 등 다양한 임상 데이터 도메인 확장 가능
        
### Author  
	** 양 소 라 | RN, BSN, MSN **
	
	Clinical Data Science Researcher
	
	AI Developer (End-to-End Clinical AI Bootcamp, AlphaCo)
	
	Domain Focus: Clinical Data Management & Digital Medicine
					- DCT
					- CDISC/CDASH
					- AI for eCRF & NLP-based Document Automation

### 참고문헌 
	 • GPT-2를 이용한 챗봇 파인튜닝. https://wikidocs.net/217620
	 
	 • Hugging Face TRL Docs
		
	 • Jin et al. (2019). PubMedQA: A Dataset for Biomedical Research Question Answering. EMNLP 2019.
		
	 • LGAI Research (2024). EXAONE 3.5 Multilingual Large Language Model.
	 
	 • Unsloth Docs — Efficient LoRA Training
	 
	 • 딥러닝 전이학습(Transfer Learning)과 파인튜닝(Fine Tuning) → https://hi-ai0913.tistory.com/32
