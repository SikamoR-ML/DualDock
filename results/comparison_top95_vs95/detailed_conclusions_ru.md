# Подробные выводы по сравнению top-95 vs top-95

## Контекст сравнения

- Сравнение выполнено на равных выборках: `top-95` лучших молекул из single-run и `top-95` из dual-run.
- Источник для сравнения: базовые ранжированные результаты (`results/ranked_ligands.jsonl`), без дополнительного пост-сэмплинга dual.
- Это позволяет убрать перекос по размеру выборки и сравнивать одинаковое число кандидатов.

## Ключевые численные итоги

- `Single base total_reward`: mean `0.3399`, median `0.3090`, max `0.6757`.
- `Dual base total_reward`: mean `0.0101`, median `0.0000`, max `0.0806`.
- Разница по total reward (single - dual): `+0.3299`, 95% CI `0.3066..0.3554`.
- Mann-Whitney по total reward: `p = 8.579e-34` (различие статистически значимое).
- Для `targetA` также выраженный разрыв:
  - single mean `0.3399` vs dual mean `0.1284`,
  - `p = 4.237e-26`.
- В dual положительная селективность (`targetA - offTargetB > 0`) только у `33/95` (34.7%).

## Интерпретация графиков

## `01_total_reward_hist_compare.png`

- Гистограмма показывает, что распределение single смещено в область высоких reward.
- Dual сосредоточен около нуля; это означает, что большая часть кандидатов в dual не проходит по селективности и получает низкий итоговый reward.

## `02_targetA_ecdf_compare.png`

- Даже по целевой мишени `targetA` single превосходит dual на top-95.
- Кривая dual показывает более высокую долю молекул с низким `targetA_score`.

## `03_dual_target_vs_offtarget_scatter.png`

- Точки dual часто лежат около или ниже диагонали `targetA = offTargetB`.
- Это прямой индикатор проблемы селективности: off-target score сопоставим с targetA или выше.

## `04_dual_selectivity_margin_hist.png`

- Распределение селективного маржина сдвинуто в отрицательную область.
- Средний маржин `-0.0206`, медиана `-0.0097`: в среднем dual-кандидаты пока не селективны в пользу targetA.

## `05_topk_reward_compare.png`

- По Top-10/25/50 single стабильно выше dual.
- Практически это означает, что список лидов из single на этом этапе качественнее по оптимизируемой метрике.

## `06_reinvent_step_score_compare.png`

- Dual-run показывает низкий уровень reward по шагам и слабую динамику роста.
- Это согласуется с общей картиной: задача dual существенно сложнее и требует большего compute-бюджета.

## Научно корректный вывод

1. При equal-size сравнении `95 vs 95` single-run статистически и practically значительно превосходит dual-run по `total_reward` и `targetA_score`.
2. Основная причина отставания dual — селективность: для большинства молекул off-target сигнал слишком велик, что обнуляет или сильно уменьшает итоговый reward.
3. Текущий dual-run корректно трактовать как диагностический baseline, а не как финальную оценку качества dual-режима.
4. Для финального вывода о dual-модели нужен более длинный и ресурсный запуск с большим числом RL-шагов и полноценно завершенным post-sampling.

## Формулировка для слайда

`Equal-size top-95 analysis shows a clear and statistically robust advantage of single-target optimization over the current dual-selectivity run; dual performance is currently limited by selectivity constraints and compute budget.`
