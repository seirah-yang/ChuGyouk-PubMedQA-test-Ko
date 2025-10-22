# ChuGyouk-PubMedQA-test-Ko
LLM Fine-Tuning with EXAONE 3.5 (2.4B-Instruct)

## 1. 프로젝트 개요 (Project Overview)

이 프로젝트는 LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct 모델을 활용하여
PubMedQA-test-Ko 데이터셋의 의학적 질문(Question) 과 *논문 문맥(Context)*을 입력으로 받아
요약형 서술 답변(Long Answer) 을 생성하도록 학습(fine-tuning)하는 프로젝트입니다.

이는 “의료 데이터 + AI + 품질 관리” 교차점에서
임상연구 데이터 품질 검증, 보고서 자동화, 질의응답(QA) 시스템 등으로 확장 가능한 Proof of Concept (PoC) 연구입니다.

## 2. 데이터셋 (Dataset)

 1) Name: ChuGyouk/PubMedQA-test-Ko

 2) Type: 한국어 번역 버전의 PubMedQA

 3) Structure:
  (1)QUESTION :	의학적 질문
  (2)CONTEXTS : 관련 논문 요약(Abstract)
  (3) LONG_ANSWER : 논문 내용 기반 서술형 답변

## 3. 모델 및 구조 (Model & Architecture)

 1) Base model: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct

 2) Framework: PyTorch + Hugging Face Transformers

 3) Task: Text Generation (질문 + 문맥 → 답변 생성)

 4) Architecture:

  (1) Input: QUESTION + CONTEXTS

  (2) Output: LONG_ANSWER (자연어 문장)

  (3) Fine-tuning scope: 상위 4개 Transformer layer + LM head

  (4) Optimizer: AdamW (lr=1e-5)


  ``` python
  for name, param in model.named_parameters():
    if "transformer" in name or "model" in name:
        if any(k in name for k in ["layers.28", "layers.29", "layers.30", "layers.31"]):
            param.requires_grad = True   
        else:
            param.requires_grad = False  
    else:
        param.requires_grad = True      

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
양 소 라 | RN, BSN, MSN, 
**참고 문헌**
Jin et al. (2019), PubMedQA: A Dataset for Biomedical Research Question Answering, EMNLP 2019
LGAI-EXAONE (2024), EXAONE 3.5: Multilingual Large Language Model → https://huggingface.co/LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct
딥러닝 전이학습(Transfer Learning)과 파인튜닝(Fine Tuning) → https://hi-ai0913.tistory.com/32
GPT-2를 이용한 챗봇 파인튜닝 → https://wikidocs.net/217620
Hugging Face Docs — https://huggingface.co/docs/transformers
Dataset — https://huggingface.co/datasets/ChuGyouk/PubMedQA-test-Ko
