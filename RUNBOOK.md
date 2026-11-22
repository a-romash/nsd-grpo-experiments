# Runbook: запуск на 2× A100 (Selectel, прерываемый)

Сервер: 2× A100 40GB, 16 vCPU, 64 ГБ RAM, 128 ГБ NVMe, 191 ₽/ч.

## Как устроен пайплайн

```
GPU1: trl vllm-serve ── генерирует ответы (8 на задачу) ──┐
                                                          ▼
GPU0: python -m src.experiment --stage train  ── GRPO (+NSD) на LoRA в bf16
       │  после каждого шага LoRA вливается в веса и отправляется в vLLM
       └─ чекпоинт каждые 25 шагов → results/runs/<profile>/<run>/checkpoint-N
GPU0: python -m src.experiment --stage eval   ── vLLM, весь MATH-500: greedy + Avg@8
```

- Для каждого запуска vLLM перезапускается, так что каждый метод стартует с базовой Qwen3-1.7B.
- Все методы используют одни и те же гиперпараметры из `configs/common.yaml`. Размер батча, число шагов и размер eval задаёт профиль (`configs/profiles/`):

| Профиль | Задач/шаг × генераций | Шагов | Задач увидит | Оценка времени, 3 метода |
|---|---|---|---|---|
| `short` | 16 × 8 | 150 | ~2.4K | ~6–8 ч |
| `full` | 32 × 8 | 250 | ~8K (~1 эпоха) | ~15–25 ч |
| `smoke` | накладывается на short/full: 10 шагов, eval на 50 задачах | | | ~20–30 мин |

Оценки времени грубые. Точнее их пересчитает смоук-тест (см. ниже).

## 1. Установка

```bash
nvidia-smi                       # обе карты видны? версия драйвера >= 550 (лучше 570+)
df -h                            # где лежит диск, переживает ли он прерывание (см. п. 5)

cd /data && git clone <repo> nsd_grpo_starter && cd nsd_grpo_starter   # или scp
python3.11 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt  # torch приедет вместе с vLLM, отдельно не ставить
export HF_HOME=/data/hf_cache    # добавьте в ~/.bashrc
python scripts/check_server.py
python -m pytest -q
```

Если pip не может разрешить зависимости между `trl[vllm]`, `transformers` и `torch`, уберите пин `transformers` и поставьте ту версию, которую тянет vLLM. Наш код использует только стабильные API.

## 2. Смоук-тест (обязательно перед долгим запуском)

```bash
tmux new -s smoke
scripts/smoke.sh                 # проверяет профиль short; BASE_PROFILE=full scripts/smoke.sh для full
```

Что делает скрипт:
1. Обучает `grpo_nsd_all` с **теми же размерами батча и длинами**, что в боевом профиле, но только 10 шагов. Берётся NSD-all, потому что это самый дорогой режим, и оценка времени получается сверху.
2. На шаге 7 **имитирует прерывание**: процесс выходит, vLLM гасится. Последний чекпоинт при этом сделан на шаге 5.
3. Запускает ту же команду ещё раз. Обучение должно продолжиться с `checkpoint-5` и дойти до 10.
4. Делает быстрый eval на 50 задачах и печатает отчёт.

**Что смотреть в отчёте:**

| Что | Норма | Если не так |
|---|---|---|
| `Median step` | ~30–60 с для short | на этом числе строится прогноз времени и денег, который печатается ниже |
| `Peak VRAM` | < 36 GiB | OOM или впритык → `micro_batch_size: 4` в `configs/common.yaml` |
| `loss`, `kl`, `nsd/loss` | конечные числа | NaN/inf → не запускать долгий прогон, разбираться |
| `reward` | 0.3–0.8 | ~0: сломан verifier или формат ответа; ~1: задачи слишком лёгкие, сигнала нет |
| `frac_reward_zero_std` | < 0.5 | доля групп, где все 8 ответов одинаковы. Такие группы GRPO ничего не дают |
| `completions/clipped_ratio` | < 0.1 | много обрезанных ответов → увеличить `max_completion_tokens` |
| `nsd/active_rows` | > 0 | для режима zero смотрите это в боевом прогоне: если ~0, zero фактически работает как обычный GRPO |
| `Resume` | `OK, resumed from .../checkpoint-5` | если нет, прерывание сотрёт прогресс, долгий запуск не начинать |
| Eval | accuracy адекватная, truncated мал | проверьте глазами пару ответов в `predictions.jsonl` |

Логи лежат в `results/logs/short+smoke/`, артефакты в `results/runs/short+smoke/`.

## 3. Боевой запуск

```bash
tmux new -s run
PROFILE=short SEEDS="1" scripts/run_all.sh
# потом при желании:
PROFILE=full SEEDS="1 2 3" scripts/run_all.sh
```

Скрипт делает так:
- baseline eval, затем для каждого seed: `grpo` → `grpo_nsd_all` → `grpo_nsd_zero`, каждый с train и eval;
- в конце пишет `results/runs/<profile>/summary.md`, `accuracy.png`, `training_curves.png`.

Отдельный метод: `EXPERIMENTS="grpo_nsd_zero" SEEDS="2" scripts/run_all.sh`.

## 4. После прерывания

Просто запустите **ту же команду** ещё раз:
- готовые train/eval пропускаются (по файлам `train_metrics.json` / `eval.json`);
- недообученный запуск продолжается с последнего чекпоинта;
- потеряется максимум 25 шагов.

Чтобы после перезагрузки всё продолжалось само, можно добавить в `crontab -e`:

```
@reboot cd /data/nsd_grpo_starter && . .venv/bin/activate && HF_HOME=/data/hf_cache PROFILE=short SEEDS="1" scripts/run_all.sh >> results/logs/cron.log 2>&1
```

## 5. Диск

Нужно ~30 ГБ:
- venv ~10 ГБ;
- HF-кэш ~4 ГБ;
- временная слитая модель на время eval ~3.5 ГБ (удаляется после);
- чекпоинты LoRA с оптимизатором: сотни МБ на запуск.

**Проверьте у Selectel**, сохраняется ли 128 ГБ NVMe при прерывании. Если нет, `HF_HOME`, репозиторий с `.venv` и `results/` должны лежать на отдельном сетевом томе.

## 6. Время в отчёте

- В `train_metrics.json` поле `train_seconds` — это сумма времени уникальных шагов: генерация, обучение, NSD-forward. Повторно пройденные после прерывания шаги и загрузка модели в него не входят, так что методы можно сравнивать по времени честно.
- `peak_mem_gb` — пик памяти на обучающей GPU.

## Если что-то упало

- **vLLM не стартует:** смотрите `results/logs/<profile>/*_vllm.log`. Часто причина в версии драйвера или в занятом порте: `pkill -f vllm-serve`.
- **OOM в обучении:** `micro_batch_size: 4` в `configs/common.yaml`. На эффективный батч это не влияет: gradient accumulation пересчитается сам. Ещё можно уменьшить `nsd_forward_batch_size` до 2.
- **OOM в eval:** уменьшите `vllm_gpu_memory_utilization` или `eval_max_new_tokens`.
- **Тренер ругается на аргумент `GRPOConfig`:** значит, поставилась версия TRL не 0.24.0. Проверьте `pip show trl`.
