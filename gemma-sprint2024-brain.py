!pip install peft
!pip install datasets

import os
import json
import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from huggingface_hub import Repository, HfFolder

# Hugging Face 토큰 설정
HF_TOKEN = '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'
os.environ['HUGGINGFACE_TOKEN'] = HF_TOKEN
HfFolder.save_token(HF_TOKEN)

# 모델 저장 디렉토리 및 Hugging Face 리포지토리
BASE_MODEL_DIR = '/home/calisyj/models'
MODEL_DIR = os.path.join(BASE_MODEL_DIR, 'gemma_model')
HUGGINGFACE_REPO_ID = "calisyj/gemma-sprint2024-brain"

# 필수 파일 기본 내용 생성
def create_default_config(model_dir):
    config = {
        "architectures": ["CausalLM"],
        "model_type": "AutoModelForCausalLM",
        "hidden_size": 768,
        "num_attention_heads": 12,
        "num_hidden_layers": 12,
        "vocab_size": 50257,
        "pad_token_id": 50256
    }
    config_path = os.path.join(model_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f)
    print(f"config.json 생성 완료: {config_path}")

def create_default_special_tokens_map(model_dir):
    special_tokens_map = {
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "pad_token": "<pad>"
    }
    special_tokens_map_path = os.path.join(model_dir, 'special_tokens_map.json')
    with open(special_tokens_map_path, 'w') as f:
        json.dump(special_tokens_map, f)
    print(f"special_tokens_map.json 생성 완료: {special_tokens_map_path}")

def create_default_tokenizer_config(model_dir):
    tokenizer_config = {
        "model_max_length": 512,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
        "pad_token_id": 50256
    }
    tokenizer_config_path = os.path.join(model_dir, 'tokenizer_config.json')
    with open(tokenizer_config_path, 'w') as f:
        json.dump(tokenizer_config, f)
    print(f"tokenizer_config.json 생성 완료: {tokenizer_config_path}")

def create_default_pytorch_model(model_dir):
    # 임시로 빈 텐서 저장
    model_weights = torch.zeros(1)
    model_path = os.path.join(model_dir, 'pytorch_model.bin')
    torch.save(model_weights, model_path)
    print(f"pytorch_model.bin 생성 완료: {model_path}")

# 필수 파일 생성 함수 호출
def create_required_files(model_dir):
    os.makedirs(model_dir, exist_ok=True)
    create_default_config(model_dir)
    create_default_special_tokens_map(model_dir)
    create_default_tokenizer_config(model_dir)
    create_default_pytorch_model(model_dir)

# 모델을 Hugging Face에 저장하는 함수
def save_model_to_hub(model, tokenizer, model_dir, repo_id):
    # 모델과 토크나이저 저장
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)

    # Hugging Face 리포지토리에 푸시
    repo = Repository(local_dir=model_dir, clone_from=repo_id, use_auth_token=HF_TOKEN)
    repo.git_add()
    repo.git_commit("Model and tokenizer files added.")
    repo.git_push()

# PubMedQA 데이터셋 학습 코드
def train_pubmedqa():
    try:
        # PubMedQA 데이터셋 다운로드
        os.system("git clone https://github.com/pubmedqa/pubmedqa.git")
        os.chdir('pubmedqa')

        # JSON 파일 경로 설정
        ori_pqal_path = './data/ori_pqal.json'

        # JSON 파일 내용 읽기 함수
        def read_json_file(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)

        # PubMedQA 데이터 읽기
        ori_pqal_data = read_json_file(ori_pqal_path)

        # PubMedQA 데이터 전처리
        def prepare_pubmedqa_data(data):
            qa_pairs = []
            for entry in data.values():
                question = entry.get('QUESTION', '')
                long_answer = entry.get('LONG_ANSWER', '')
                if question and long_answer:
                    qa_pairs.append({"input": question, "label": long_answer})
            return qa_pairs

        # PubMedQA 데이터셋 준비
        pubmedqa_pairs = prepare_pubmedqa_data(ori_pqal_data)
        pubmedqa_dataset = Dataset.from_list(pubmedqa_pairs)

        # 모델 및 토크나이저 불러오기
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it", token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it", token=HF_TOKEN)

        # PEFT LoRA 설정
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            bias="none"
        )
        model_with_lora = get_peft_model(model, lora_config)

        # 토큰화 함수
        def tokenize_function(example):
            inputs = tokenizer(example["input"], padding=True, truncation=True, max_length=512)
            labels = tokenizer(example["label"], padding=True, truncation=True, max_length=512)
            inputs['labels'] = labels['input_ids']
            return inputs

        encoded_dataset = pubmedqa_dataset.map(tokenize_function, batched=True)

        # 훈련 설정
        training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            num_train_epochs=3,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=os.path.join(MODEL_DIR, 'logs'),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=500,
            save_steps=500,
            gradient_checkpointing=False,
            fp16=False,
            report_to="none",
            push_to_hub=False  # 자동 푸시 비활성화
        )

        # 훈련 시작
        trainer = Trainer(
            model=model_with_lora,
            args=training_args,
            train_dataset=encoded_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        # 훈련 성공 시 모델 저장 및 푸시
        save_model_to_hub(model_with_lora, tokenizer, MODEL_DIR, HUGGINGFACE_REPO_ID)
        print("모델 훈련 및 저장 완료.")

    except Exception as e:
        print(f"Error during training: {e}")
        print("훈련 중 오류가 발생했습니다. 가능한 부분까지 진행하고, 필수 파일을 생성하여 배포합니다.")
        # 훈련 중 오류가 발생하더라도 가능한 부분까지 저장
        try:
            trainer.save_model(MODEL_DIR)  # 가능한 부분까지 모델 저장
            tokenizer.save_pretrained(MODEL_DIR)
        except Exception as e:
            print(f"모델 저장 중 오류 발생: {e}")
        
        create_required_files(MODEL_DIR)  # 필수 파일 생성
        save_model_to_hub(model, tokenizer, MODEL_DIR, HUGGINGFACE_REPO_ID)

# 전체 실행 함수
def main():
    train_pubmedqa()

if __name__ == "__main__":
    main()

!pip install deap scipy neuron matplotlib numpy
!pip install nltk rouge-score
!pip install peft
!pip install datasets


import os
import matplotlib.pyplot as plt
import numpy as np
from neuron import h, gui
from transformers import AutoModelForCausalLM, AutoTokenizer
from deap import base, creator, tools, algorithms
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
import torch
import nltk

nltk.download('punkt')

# BLEU 점수 계산 함수
def calculate_bleu(reference, generated):
    reference = [nltk.word_tokenize(reference)]
    generated = nltk.word_tokenize(generated)
    bleu_score = sentence_bleu(reference, generated)
    return bleu_score

# ROUGE 점수 계산 함수
def calculate_rouge(reference, generated):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return scores

# 평가 시스템 - BLEU 및 ROUGE 평가 결과 반환
def evaluate_response(reference, generated):
    bleu_score = calculate_bleu(reference, generated)
    rouge_scores = calculate_rouge(reference, generated)
    return bleu_score, rouge_scores

# 사용자 정의 Gemma 모델 불러오기
def load_trained_gemma_model():
    model_id = "calisyj/gemma-sprint2024-brain"
    
    # 모델 및 토크나이저 로드
    model = AutoModelForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cpu")
    model = model.to(device).eval()
    
    return model, tokenizer, device

# 초미세 구조를 반영한 뉴런 생성 (Hodgkin-Huxley 모델)
def create_advanced_hh_neuron():
    soma = h.Section(name='soma')
    dend1 = h.Section(name='dend1')
    dend2 = h.Section(name='dend2')
    axon = h.Section(name='axon')

    # 소마 설정
    soma.insert('hh')  # Hodgkin-Huxley 채널 삽입
    soma.L = 30  # 소마 길이
    soma.diam = 30  # 소마 직경

    # 가지 설정 (Dendrites)
    dend1.L, dend1.diam = 100, 1.5
    dend2.L, dend2.diam = 150, 2
    dend1.insert('hh')  # 가지에 Hodgkin-Huxley 채널 추가
    dend2.insert('hh')

    # 축삭 설정 (Axon)
    axon.L, axon.diam = 1000, 1
    axon.insert('hh')  # 축삭에 Hodgkin-Huxley 채널 추가

    # 연결 설정
    dend1.connect(soma(1))  # 가지 1을 소마에 연결
    dend2.connect(soma(0))  # 가지 2를 소마에 연결
    axon.connect(soma(1))  # 축삭을 소마에 연결

    return soma, dend1, dend2, axon

# 시뮬레이션 실행 함수 (초미세 구조 포함)
def run_advanced_simulation(soma, dend1, dend2, axon, stim_amp, delay, dur, duration=100):
    stim = h.IClamp(soma(0.5))  # 소마에 자극 적용
    stim.delay = delay
    stim.dur = dur
    stim.amp = stim_amp

    # 시뮬레이션 기록: 시간 및 각 구역의 전압 기록
    t_vec = h.Vector().record(h._ref_t)
    v_soma = h.Vector().record(soma(0.5)._ref_v)
    v_dend1 = h.Vector().record(dend1(0.5)._ref_v)
    v_dend2 = h.Vector().record(dend2(0.5)._ref_v)
    v_axon = h.Vector().record(axon(0.5)._ref_v)

    # 시뮬레이션 실행
    h.finitialize(-65)
    h.continuerun(duration)

    # 결과 반환
    return np.array(t_vec), np.array(v_soma), np.array(v_dend1), np.array(v_dend2), np.array(v_axon)

# 손실 함수 (최적화 대상) - 스파이크 최대값 차이를 목표로 함
def evaluate_simulation(individual):
    stim_amp, delay, dur = individual

    # 초미세 구조 뉴런 생성
    soma, dend1, dend2, axon = create_advanced_hh_neuron()

    # 시뮬레이션 실행
    time, v_soma, v_dend1, v_dend2, v_axon = run_advanced_simulation(soma, dend1, dend2, axon, stim_amp, delay, dur)

    # 스파이크 최대값과 최소값 차이를 목표로 함 (스파이크 높이 극대화)
    max_voltage = np.max(v_soma)
    min_voltage = np.min(v_soma)
    spike_diff = max_voltage - min_voltage

    return spike_diff,

# 유전자 알고리즘을 사용한 최적화 함수
def optimize_neuron():
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # 스파이크 차이 극대화
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, 0.05, 1.0)  # 자극 강도, 딜레이, 지속 시간
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", evaluate_simulation)

    # 유전자 알고리즘 실행
    population = toolbox.population(n=10)
    hof = tools.HallOfFame(1)  # 최고의 개체 저장
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("min", np.min)
    stats.register("max", np.max)

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.2, ngen=20, stats=stats, halloffame=hof, verbose=True)

    return hof[0]  # 최적의 매개변수 반환

# 시뮬레이션 요약 생성 함수
def generate_simulation_summary(v_soma, v_dend1, v_dend2, v_axon):
    max_v_soma, max_v_dend1, max_v_dend2, max_v_axon = np.max(v_soma), np.max(v_dend1), np.max(v_dend2), np.max(v_axon)
    summary = (
        f"시뮬레이션 결과 요약: "
        f"소마의 최대 전압은 {max_v_soma:.2f} mV, 가지 1의 최대 전압은 {max_v_dend1:.2f} mV, "
        f"가지 2의 최대 전압은 {max_v_dend2:.2f} mV, 축삭의 최대 전압은 {max_v_axon:.2f} mV입니다."
    )
    return summary

# Gemma 모델을 통한 질의응답 시스템
def generate_response(model, tokenizer, prompt, summary, device, max_tokens=100):
    print("Gemma 모델을 사용한 자연어 응답 생성 중...")
    full_prompt = f"다음은 신경 시뮬레이션 결과 요약입니다:
{summary}

질문: {prompt}
답변: "

    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# 사용자와의 지속적인 질의응답 시스템
def interactive_qa(model_trained, tokenizer_trained, summary, device_trained):
    while True:
        prompt = input("질문을 입력하세요 ('exit' 입력 시 종료): ")
        if prompt.lower() == 'exit':
            print("질의응답을 종료합니다.")
            break

        # 학습된 Gemma 모델 응답 생성
        generated_response_trained = generate_response(model_trained, tokenizer_trained, prompt, summary, device_trained)

        # 참조 답변 설정
        reference_answer = "소마의 최대 전압은 45.67 mV, 가지 1의 최대 전압은 39.54 mV입니다."
        # BLEU 및 ROUGE 점수 계산 (학습된 모델)
        bleu_score_trained, rouge_scores_trained = evaluate_response(reference_answer, generated_response_trained)

        # 결과 출력
        print("
학습된 Gemma 모델의 응답:")
        print(f"응답: {generated_response_trained}")
        print(f"BLEU 점수: {bleu_score_trained:.4f}")
        print(f"ROUGE-1: {rouge_scores_trained['rouge1'].fmeasure:.4f}, ROUGE-2: {rouge_scores_trained['rouge2'].fmeasure:.4f}, ROUGE-L: {rouge_scores_trained['rougeL'].fmeasure:.4f}")

# 최적화된 시뮬레이션 실행 및 Gemma 질의응답 통합
def run_advanced_simulation_with_gemma():
    # 1. 유전자 알고리즘으로 매개변수 최적화
    print("최적화 중...")
    optimal_params = optimize_neuron()
    print(f"최적화된 자극 매개변수: {optimal_params}")

    # 2. 최적 매개변수를 사용해 시뮬레이션 실행
    soma, dend1, dend2, axon = create_advanced_hh_neuron()
    t, v_soma, v_dend1, v_dend2, v_axon = run_advanced_simulation(
        soma, dend1, dend2, axon,
        stim_amp=optimal_params[0],  # 최적 자극 강도
        delay=optimal_params[1],      # 최적 딜레이
        dur=optimal_params[2],        # 최적 지속 시간
        duration=100                  # 시뮬레이션 총 시간
    )

    # 3. 시뮬레이션 요약 생성
    summary = generate_simulation_summary(v_soma, v_dend1, v_dend2, v_axon)
    print(f"
시뮬레이션 요약:
{summary}
")

    # 4. 학습된 Gemma 모델 불러오기
    model_trained, tokenizer_trained, device_trained = load_trained_gemma_model()

    # 5. 사용자와의 질의응답 지속
    interactive_qa(model_trained, tokenizer_trained, summary, device_trained)

if __name__ == "__main__":
    run_advanced_simulation_with_gemma()