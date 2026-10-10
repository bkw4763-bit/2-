# 3·4단계 결과 그림. results/policy 의 CSV만 읽으므로 cpi.py, optimize.py --sensitivity --replay 를 먼저 실행합니다.
#   cpi.png      : 3단계 ① 반영률 경로  ② 상한제가 낮춘 월별 물가(유종별 누적)와 정부 설명 0.6%p 대조
#   optimize.png : 4단계 ① 11차 권고와 현재 격차  ② 216개 첫 행동의 물가-재정 맞바꿈  ③ 경로별 24주 재정 부담 분포
#   checks.png   : 4단계 검증 ① 민감도(조건별 권고)  ② 과거 결정 재현(실제 조정 vs 권고)
# 유종 색은 1단계 그림(counterfactual.png)과 같게 유종에 고정합니다. 실행: python policy/plots.py
import logging

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # 없는 글꼴을 건너뛸 때 나오는 경고 숨김
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C

COLORS = {"gasoline": "#2a78d6", "diesel": "#eb6834", "kerosene": "#1baf7a"}
MARKERS = {"gasoline": "o", "diesel": "s", "kerosene": "^"}       # 색 말고도 모양으로 유종 구분
INK, INK_2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OUT = C.OUT_DIR




# 1. 그림 공통 서식(1단계 그림과 같음). 한글 글꼴은 윈도·맥·리눅스 순서로 찾습니다.
def apply_style():
    plt.rcParams.update({
        "font.family": ["Malgun Gothic", "AppleGothic", "NanumGothic", "WenQuanYi Zen Hei", "sans-serif"],
        "axes.unicode_minus": False, "font.size": 9,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10,
        "axes.titlelocation": "left", "axes.titleweight": "bold",
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "lines.linewidth": 2,
    })


def read(name, **kw):
    return pd.read_csv(OUT / name, encoding="utf-8-sig", **kw)


# 2. 3단계 그림.
def plot_cpi():
    pt = read("cpi_pass_through.csv", index_col=0)
    effect = read("cpi_policy_effect.csv", index_col=0)
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.4]})

    # ① 반영률: 상한을 바꾼 뒤 판매가가 몇 %까지 따라가나. 끝점에 유종 이름과 4주 값을 직접 적습니다.
    weeks = [1, 2, 3, 4]
    for f in C.FUELS:
        y = pt.loc[C.KO[f], ["1주", "2주", "3주", "4주"]].to_numpy(float)
        a.plot(weeks, y, color=COLORS[f], marker=MARKERS[f], markersize=6, label=C.KO[f])
        nudge = {"gasoline": -7, "diesel": 7, "kerosene": 0}[f]          # 휘발유·경유 끝점이 붙어 있어 위아래로 벌림
        a.annotate(f"{C.KO[f]} {y[-1]:.0f}%", (4, y[-1]), xytext=(6, nudge), textcoords="offset points",
                   va="center", color=INK, fontsize=9)
    a.set_xticks(weeks, ["1주", "2주", "3주", "4주"])
    a.set_xlim(0.8, 4.9)
    a.set_ylim(0, 100)
    a.set_ylabel("판매가 변화 ÷ 상한 변화 (%)")
    a.set_title("① 반영률: 상한 조정 뒤 판매가가 따라가는 정도")
    a.legend(loc="lower right", fontsize=8)
    a.text(0, -0.16, "2차(+210원)·7차(−150원) 실제 반응 평균. 3주는 2·4주 중간값", transform=a.transAxes, color=MUTED, fontsize=8)

    # ② 상한제가 낮춘 물가(%p): 유종별 기여를 쌓은 막대. 3~8월 평균을 정부 설명과 나란히 적습니다.
    months = [str(m)[-2:].lstrip("0") + "월" for m in effect.index]
    x = np.arange(len(months))
    bottom = np.zeros(len(x))
    w = C.CPI_WEIGHTS
    for f in C.FUELS:
        part = (w[f] / 1000 * effect[f"{f}_retail_cut_pct"]).to_numpy()
        b.bar(x, part, bottom=bottom, width=0.62, color=COLORS[f], edgecolor=SURFACE, linewidth=2, label=C.KO[f])
        bottom += part
    for i, v in enumerate(bottom):
        b.annotate(f"{v:.2f}", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", color=INK, fontsize=8)
    avg = effect.loc["2026-03":"2026-08", "cpi_cut_pp"].mean()
    b.hlines(0.6, -0.4, 5.4, color=INK_2, linestyle="--", linewidth=1.2)
    b.annotate("정부 설명 3~8월 평균 0.6%p", (5.4, 0.6), xytext=(0, 4), textcoords="offset points", ha="right", color=INK_2, fontsize=8)
    b.set_xticks(x, months)
    b.set_ylabel("물가 인하 효과 (%p)")
    b.set_ylim(0, bottom.max() * 1.18)
    b.set_title(f"② 상한제가 낮춘 소비자물가: 3~8월 평균 {avg:.2f}%p (정부 0.6%p)")
    b.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(OUT / "cpi.png", dpi=160)
    plt.close(fig)


# 3. 4단계 그림.
def plot_optimize():
    state = read("optimize_state.csv").set_index("fuel")
    actions = read("optimize_actions.csv")
    paths = read("optimize_paths.csv", index_col=0)
    limit = float(state["cpi_limit"].iloc[0])
    fig = plt.figure(figsize=(13, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1.25, 1.1])
    a, b, c = (fig.add_subplot(gs[0, i]) for i in range(3))

    # ① 11차 권고: 현재 격차(연한 막대) 안에서 권고 조정 폭(진한 막대)이 얼마인지.
    y = np.arange(3)[::-1]
    for yi, f in zip(y, C.FUELS):
        gap, act = state.loc[f, "gap"], state.loc[f, "action"]
        a.barh(yi, gap, height=0.55, color=COLORS[f], alpha=0.22)
        a.barh(yi, act, height=0.55, color=COLORS[f])
        a.annotate(f"{act:+.0f}원", (act, yi), xytext=(4, 0), textcoords="offset points", va="center", color=INK, fontweight="bold")
        a.annotate(f"격차 {gap:.0f}원", (gap, yi), xytext=(-4, 0), textcoords="offset points", va="center", ha="right", color=INK_2, fontsize=8)
    a.set_yticks(y, [C.KO[f] for f in C.FUELS])
    a.set_xlabel("원/리터")
    a.grid(axis="y", visible=False)
    a.set_title("① 11차 권고 조정 폭 (연한 막대 = 현재 격차)")

    # ② 216개 첫 행동: 물가 기여도(가로) vs 24주 평균 재정 부담(세로). 한도 안 = 진한 점, 권고 = 별.
    ok = actions["한도 충족"].astype(str) == "True"
    cost = actions["24주 평균 부담(억 원)"] / 1e4
    b.scatter(actions.loc[~ok, "물가 기여도(%p)"], cost[~ok], s=14, color=AXIS, label="한도 초과·인하 조건 위반")
    b.scatter(actions.loc[ok, "물가 기여도(%p)"], cost[ok], s=16, color=INK_2, label="허용되는 행동")
    best = actions.iloc[0]
    b.scatter(best["물가 기여도(%p)"], cost.iloc[0], s=140, marker="*", color="#e34948", edgecolor=SURFACE, linewidth=1.5, zorder=5, label="권고")
    freeze = actions[(actions["휘발유"] == 0) & (actions["경유"] == 0) & (actions["등유"] == 0)].iloc[0]
    b.scatter(freeze["물가 기여도(%p)"], freeze["24주 평균 부담(억 원)"] / 1e4, s=60, marker="D", color=SURFACE, edgecolor=INK, zorder=5, label="동결")
    b.axvline(limit, color=INK_2, linestyle="--", linewidth=1.2)
    b.annotate(f"월 물가 한도 {limit:.3f}%p", (limit, cost.min()), xytext=(4, 0), textcoords="offset points", color=INK_2, fontsize=8, va="bottom")
    b.set_xlabel("이번 달 물가 기여도 (%p)")
    b.set_ylabel("24주 평균 재정 부담 (조 원)")
    b.set_title("② 첫 행동 216개: 물가를 더 올릴수록 재정 부담 감소")
    b.legend(loc="lower left", fontsize=8)

    # ③ 경로 500개의 24주 재정 부담 분포: 권고 vs 동결. 겹치므로 선(계단)으로 그립니다.
    bins = np.linspace(0, np.quantile(paths.to_numpy(), 0.99) / 1e4, 30)
    for col, color, ls in [("동결", MUTED, "--"), ("권고", "#e34948", "-")]:
        v = paths[col] / 1e4
        c.hist(v, bins=bins, histtype="step", color=color, linestyle=ls, linewidth=2, label=f"{col} (평균 {v.mean():.2f}조)")
        c.axvline(v.mean(), color=color, linestyle=ls, linewidth=1)
    c.set_xlabel("24주 재정 부담 (조 원)")
    c.set_ylabel("경로 수")
    c.set_title("③ 국제가 경로 500개의 재정 부담 분포")
    c.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "optimize.png", dpi=160)
    plt.close(fig)


# 4. 4단계 검증 그림.
def plot_checks():
    sens = read("optimize_sensitivity.csv")
    rep = read("optimize_replay.csv")
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})

    # ① 민감도: 조건(행)마다 유종별 권고 조정 폭(점). 기본 권고는 맨 위.
    y = np.arange(len(sens))[::-1]
    offsets = {"gasoline": 0.18, "diesel": 0, "kerosene": -0.18}
    for f in C.FUELS:
        a.scatter(sens[C.KO[f]], y + offsets[f], color=COLORS[f], marker=MARKERS[f], s=55, label=C.KO[f], zorder=3)
    a.set_yticks(y, sens["조건"])
    a.set_xticks(C.ACTIONS)
    a.set_xlabel("11차 권고 조정 폭 (원)")
    a.axhspan(y[0] - 0.4, y[0] + 0.4, color=GRID, zorder=0)
    a.set_title("① 민감도: 조건을 바꿔도 경유·등유를 더 많이 올림")
    a.legend(loc="lower left", fontsize=8, ncol=3, bbox_to_anchor=(0, -0.28))

    # ② 과거 결정 재현: 차수마다 실제 조정(회색 막대, 3유종 같은 폭)과 같은 규칙의 권고(유종별 점).
    x = rep["차수"].to_numpy()
    actual = rep["실제"].str.split("/").str[0].astype(float)
    b.bar(x, actual, width=0.6, color=AXIS, label="실제 결정 (3유종 같은 폭)")
    rec = rep["권고"].str.split("/", expand=True).astype(float)
    for i, f in enumerate(C.FUELS):
        b.scatter(x + (i - 1) * 0.18, rec[i], color=COLORS[f], marker=MARKERS[f], s=40, zorder=3, label=f"권고 {C.KO[f]}")
    for xi, v in zip(x, actual):
        if v:
            b.annotate(f"{v:+.0f}", (xi, v), xytext=(0, 4 if v > 0 else -12), textcoords="offset points", ha="center", color=INK, fontsize=8)
    b.axhline(0, color=AXIS, linewidth=1)
    b.set_xticks(x, [f"{r}차" for r in x])
    b.set_ylabel("조정 폭 (원)")
    b.set_title(f"② 과거 결정 재현: 방향 일치 {int(rep['방향 일치'].astype(str).eq('True').sum())}/{len(rep)} (정부는 물가를 더 무겁게 봄)")
    b.legend(loc="lower left", fontsize=8, ncol=4, bbox_to_anchor=(0, -0.28))
    fig.tight_layout()
    fig.savefig(OUT / "checks.png", dpi=160)
    plt.close(fig)


# 5. 한 장 요약(발표용). 질문 하나에 그림 하나, 제목이 곧 결론입니다.
#   ① 정부가 지금 리터당 얼마를 메워 주고 있고, 11차에 얼마를 올리나
#   ② 같은 100원을 올려도 물가가 오르는 정도가 유종마다 다르다 → 휘발유는 조금, 등유는 많이 올리는 이유
#   ③ 권고대로 하면 정부 부담이 얼마나 줄고, 물가는 얼마나 오르나
def plot_summary():
    st = read("optimize_state.csv").set_index("fuel")
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1, 0.9]})
    y = np.arange(3)[::-1]
    names = [C.KO[f] for f in C.FUELS]

    for yi, f in zip(y, C.FUELS):
        gap, act = st.loc[f, "gap"], st.loc[f, "action"]
        a.barh(yi, gap, height=0.6, color=COLORS[f], alpha=0.22)
        a.barh(yi, act, height=0.6, color=COLORS[f])
        a.annotate(f"{act:+.0f}원 올림", (act, yi), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=11, fontweight="bold")
        a.annotate(f"정부가 메워 주는 돈 {gap:.0f}원", (gap, yi), xytext=(-5, 0), textcoords="offset points", va="center", ha="right", color=INK_2, fontsize=9)
    a.set_yticks(y, names, fontsize=11)
    a.set_xlabel("리터당 원")
    a.grid(axis="y", visible=False)
    a.set_title("① 11차 권고: 휘발유 +30원, 경유 +90원, 등유 +120원")

    per100 = st["cpi_per_100won"]
    for yi, f in zip(y, C.FUELS):
        b.barh(yi, per100[f], height=0.6, color=COLORS[f])
        b.annotate(f"{per100[f]:.3f}%p", (per100[f], yi), xytext=(5, 0), textcoords="offset points", va="center", color=INK, fontsize=11)
    b.set_yticks(y, names, fontsize=11)
    b.set_xlim(0, per100.max() * 1.35)
    b.set_xlabel("상한을 100원 올렸을 때 소비자물가 상승 (%p)")
    b.grid(axis="y", visible=False)
    b.set_title("② 같은 100원이라도 휘발유가 물가를 가장 많이 올림")

    freeze, rec = st["cost24_freeze"].iloc[0] / 1e12, st["cost24_rec"].iloc[0] / 1e12
    bars = c.bar(["그대로 두면\n(동결)", "권고대로 하면"], [freeze, rec], width=0.55, color=[AXIS, "#e34948"])
    for bar, v in zip(bars, [freeze, rec]):
        c.annotate(f"{v:.2f}조 원", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 4), textcoords="offset points",
                   ha="center", color=INK, fontsize=11, fontweight="bold")
    c.set_ylim(0, freeze * 1.2)
    c.set_ylabel("앞으로 24주 정부 부담 (조 원)")
    c.tick_params(axis="x", labelsize=10)
    c.grid(axis="x", visible=False)
    saving = (freeze - rec) * 1e4                                       # 억 원
    c.set_title(f"③ 정부 부담 약 {saving:,.0f}억 원 감소, 물가는 +{st['cpi_rec'].iloc[0]:.2f}%p")
    fig.tight_layout()
    fig.savefig(OUT / "summary.png", dpi=160)
    plt.close(fig)


def main():
    apply_style()
    plot_summary()
    plot_cpi()
    plot_optimize()
    plot_checks()
    print(f"저장: {OUT / 'summary.png'} (발표용 요약), {OUT / 'cpi.png'}, {OUT / 'optimize.png'}, {OUT / 'checks.png'}")


if __name__ == "__main__":
    main()
