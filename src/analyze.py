
import os
import csv
import random
import statistics as pystat  
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import basic_stats as my

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "tips.csv")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")


def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)
    for row in data:
        row["tip_pct"] = str(float(row["tip"]) / float(row["total_bill"]) * 100)
    return data


def get_values(data, key):
    return [float(row[key]) for row in data]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    data = load_data(DATA_PATH)

    report = []
    def log(msg=""):
        print(msg)
        report.append(str(msg))

    # 1. Базовая статистика
    log("\n1. Базовая статистика")
    stats_csv_rows = []
    for feature in ["total_bill", "tip", "tip_pct"]:
        m = my.mean(data, feature)
        v = my.variance(data, feature)
        s = my.std(data, feature)
        med = my.median(data, feature)
        modes, mode_count = my.mode(data, feature)

        log(f"{feature}: mean={m:.3f}  var={v:.3f}  std={s:.3f}  median={med:.3f}  "
            f"mode={modes} (частота={mode_count})")
        
        stats_csv_rows.append({
            "feature": feature, "mean": round(m, 4), "variance": round(v, 4),
            "std": round(s, 4), "median": round(med, 4),
        })

    # проверка
    check_mean = pystat.mean(get_values(data, "total_bill"))
    log(f"\nСверка со stdlib (total_bill): ручной mean={my.mean(data,'total_bill'):.6f}  "
        f"stdlib mean={check_mean:.6f}")

    # 2. Закон распределения категориальных признаков
    log("\n2. Закон распределения категориальных признаков")
    for key in ["day", "smoker", "sex", "time"]:
        dist = my.distribution_table(data, key)
        pretty = ", ".join(f"{k}={v:.3f}" for k, v in dist.items())
        log(f"P({key}): {pretty}")

    # 3. Условная вероятность / проверка независимости
    log("\n3. Условная вероятность")
    is_smoker = lambda r: r["smoker"] == "Yes"
    is_saturday = lambda r: r["day"] == "Sat"
    p_smoker = my.empirical_probability(data, is_smoker)
    p_smoker_sat = my.conditional_probability(data, is_smoker, is_saturday)
    log(f"P(smoker=Yes) = {p_smoker:.3f}")
    log(f"P(smoker=Yes | day=Sat) = {p_smoker_sat:.3f}")
    diff = abs(p_smoker - p_smoker_sat)
    verdict = "похоже на НЕЗАВИСИМЫЕ события" if diff < 0.03 else "разница заметна -> события ЗАВИСИМЫ"
    log(f"|P(A) - P(A|B)| = {diff:.3f} -> {verdict}")

    # 4. Ковариация и корреляция
    log("\n4. Ковариация и корреляция (total_bill, tip)")
    cov_xy = my.cov(data, "total_bill", "tip")
    cor_xy = my.cor(data, "total_bill", "tip")
    log(f"Cov = {cov_xy:.4f}   Corr = {cor_xy:.4f}")

    # 5. Z-оценка / поиск выбросов
    log("\n5. Выбросы по Z-оценке (|Z| > 2.5)")
    m_bill, s_bill = my.mean(data, "total_bill"), my.std(data, "total_bill")
    outliers = []
    for row in data:
        z = (float(row["total_bill"]) - m_bill) / s_bill
        if abs(z) > 2.5:
            outliers.append((float(row["total_bill"]), round(z, 2)))
    log(f"Найдено выбросов: {len(outliers)}")
    for bill, z in outliers:
        log(f"  total_bill={bill:.2f}  Z={z}")

    # 6. Центральная предельная теорема (симуляция)
    log("\n6. Демонстрация ЦПТ (1000 выборок по n=30, с возвращением)")
    rng = random.Random(67)
    sample_means = [my.mean(rng.choices(data, k=30), "total_bill") for _ in range(1000)]
    m_means = sum(sample_means) / len(sample_means)
    var_means = sum((x - m_means) ** 2 for x in sample_means) / (len(sample_means) - 1)
    std_means = var_means ** 0.5
    log(f"Среднее выборочных средних = {m_means:.3f}  (истинное E(X) = {m_bill:.3f})")
    log(f"Std выборочных средних = {std_means:.3f}  "
        f"(теоретическое sigma/sqrt(30) = {s_bill / 30 ** 0.5:.3f})")

    # --- сохранение отчёта и таблицы ---
    with open(os.path.join(RESULTS_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(report))

    with open(os.path.join(RESULTS_DIR, "summary_stats.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["feature", "mean", "variance", "std", "median"])
        writer.writeheader()
        writer.writerows(stats_csv_rows)

    # --- графики ---
    total_bill_vals = get_values(data, "total_bill")
    tip_vals = get_values(data, "tip")

    plt.figure(figsize=(7, 4.5))
    plt.hist(total_bill_vals, bins=25, color="#2a78d6", edgecolor="white")
    plt.axvline(m_bill, color="red", linestyle="--", label=f"среднее = {m_bill:.2f}")
    plt.title("Распределение total_bill")
    plt.xlabel("total_bill, $")
    plt.ylabel("частота")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "hist_total_bill.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.scatter(total_bill_vals, tip_vals, alpha=0.6, color="#1baf7a")
    plt.title(f"total_bill vs tip (corr = {cor_xy:.2f})")
    plt.xlabel("total_bill, $")
    plt.ylabel("tip, $")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "scatter_bill_tip.png"), dpi=120)
    plt.close()

    day_dist = my.distribution_table(data, "day")
    plt.figure(figsize=(7, 4.5))
    plt.bar(list(day_dist.keys()), list(day_dist.values()), color="#7F77DD")
    plt.title("P(day) - эмпирическое распределение по дням")
    plt.ylabel("вероятность")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "bar_day_distribution.png"), dpi=120)
    plt.close()

    plt.figure(figsize=(7, 4.5))
    plt.hist(sample_means, bins=30, color="#e0a72e", edgecolor="white")
    plt.axvline(m_means, color="red", linestyle="--", label=f"среднее средних = {m_means:.2f}")
    plt.title("ЦПТ: распределение выборочных средних")
    plt.xlabel("выборочное среднее total_bill")
    plt.ylabel("частота")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "clt_demo.png"), dpi=120)
    plt.close()

    log(f"\nСохранено: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
