"""Build the assessed report from frozen experimental artifacts.

Run from the repository root after `python -m jet_rul reproduce`.
The generated Markdown companion contains the same prose/tables for editing.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET
from html import escape
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak

ROOT = Path(__file__).resolve().parents[1]
A = ROOT/'artifacts'
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True, exist_ok=True)
selection = json.loads((A/'selection.json').read_text())
perf = json.loads((A/'performance.json').read_text())
intervals = json.loads((A/'bootstrap_intervals.json').read_text())
env = json.loads((A/'environment.json').read_text())
support = json.loads((A/'support_audit.json').read_text())
metrics = pd.read_csv(A/'test_metrics.csv').set_index('model')
cv = pd.read_csv(A/'cv_results.csv')
pred = pd.read_csv(A/'test_predictions.csv')
ablation = pd.read_csv(A/'ablation.csv')
seeds = pd.read_csv(A/'seed_sensitivity.csv')
robust = pd.read_csv(A/'robustness.csv')
chosen = selection['selected']
m = metrics.loc[chosen]
test_count = int(ET.parse(A/'test-results.xml').getroot().find('testsuite').attrib['tests'])
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyReport', fontName='Helvetica', fontSize=10.4, leading=14.3, spaceAfter=9))
styles.add(ParagraphStyle(name='CaptionReport', fontName='Helvetica', fontSize=8.4, leading=11, textColor=colors.HexColor('#46566c'), spaceAfter=10))
styles.add(ParagraphStyle(name='TableReport', fontName='Helvetica', fontSize=8.2, leading=10.7))
styles.add(ParagraphStyle(name='TitleReport', fontName='Helvetica-Bold', fontSize=25, leading=29, textColor=colors.HexColor('#12334d'), spaceAfter=16))
styles['Heading1'].textColor = colors.HexColor('#12334d')
styles['Heading1'].fontSize = 18
styles['Heading1'].leading = 22
styles['Heading2'].fontSize = 12
styles['Heading2'].leading = 16
story, md = [], []


def p(text, style='BodyReport'):
    story.append(Paragraph(text, styles[style]))
    md.append(text.replace('<b>', '**').replace('</b>', '**'))


def heading(text, level=1):
    p(text, 'Heading1' if level == 1 else 'Heading2')


def table(headers, rows, widths=None):
    data = [[Paragraph(escape(str(c)), styles['TableReport']) for c in row] for row in [headers]+rows]
    t = Table(data, colWidths=widths, hAlign='LEFT', repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND', (0,0),(-1,0), colors.HexColor('#e4edf3')),
                           ('VALIGN', (0,0),(-1,-1),'TOP'), ('TOPPADDING',(0,0),(-1,-1),6),
                           ('BOTTOMPADDING',(0,0),(-1,-1),6),
                           ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#829bad')),
                           ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#dce3e8'))]))
    story.extend([t, Spacer(1,10)])
    md.extend(['| '+' | '.join(map(str, headers))+' |', '| '+' | '.join(['---']*len(headers))+' |'])
    md.extend('| '+' | '.join(map(str, row))+' |' for row in rows)


def figure(name, caption, height=158):
    from PIL import Image as PILImage
    path = A/'figures'/f'{name}.png'
    w,h = PILImage.open(path).size
    width = 485
    story.append(Image(str(path), width=width, height=width*h/w))
    p(caption, 'CaptionReport')
    md.append(f'![{caption}](../../artifacts/figures/{name}.png)')


def page():
    story.append(PageBreak())
    md.append('\n---\n')


def ci(values):
    return f'[{values[0]:.2f}, {values[1]:.2f}]'


p('Turbofan remaining useful life<br/>from causal sensor histories', 'TitleReport')
p('<b>Chua Shanwei</b> &nbsp; 35393343<br/>MMA3001: Numerical Methods and Machine Learning<br/>NASA C-MAPSS FD001', 'BodyReport')
p('Technical report - nine main pages followed by references. Prepared with substantial AI assistance, disclosed in Section 7. Student review and personal reflection remain necessary before submission.', 'CaptionReport')
heading('1. Engineering Problem')
p('A turbofan maintenance planner needs to estimate how many operational cycles remain before a degradation threshold is reached. Scheduling too late risks operating a degraded engine, while scheduling too early wastes useful life. This project estimates a continuous remaining useful life (RUL) from observed age and six temperature, pressure and speed channels. It evaluates whether nonlinear regression provides a defensible benefit over simpler alternatives.')
p('NASA C-MAPSS simulates run-to-failure engine degradation [1,2]. FD001 contains one operating condition and one HPC degradation mode. The 100 training engines provide complete histories; 100 separate test histories stop before failure. A maintenance review would receive an RUL point estimate and input-support warnings, rather than an automatic retirement instruction. A cycle is the benchmark time unit and is not converted into hours or flights.')
figure('engineering_data', 'Figure 1. Training-only evidence: different temperature histories and failure ages motivate sensor-based estimation rather than age alone.')
p('<b>Scope.</b> Only FD001 is supported. The model assumes future degradation resembles this simulated population. Other operating conditions, failure modes, maintenance interventions and real fleet deployment are outside the validated scope. Sensor correlations are physically plausible indicators of gas-path degradation, but the regressor does not solve conservation equations or identify a unique wear mechanism.')
p(f'<b>Main result.</b> The predeclared selection rule chose {chosen}. On 100 official endpoints it obtained MAE {m.mae:.2f} and RMSE {m.rmse:.2f} cycles. These are measured benchmark errors, not a certified maintenance margin.')
page()

heading('2. Inputs & Outputs')
p('The parser accepts a headerless, whitespace-delimited TXT file with exactly 26 numeric columns: engine ID, cycle, three settings and sensors s1-s21. It sorts by engine and cycle. For prediction, supply one engine with at least 30 contiguous observed rows and retain its absolute cycle numbers. A trailing history may start after cycle 1. Table 1 specifies all primary predictors and outputs; the other source fields are parsed for schema checks.')
table(['Input / symbol', 'Physical meaning', 'Unit', 'Training range / domain'], [
 ['ID','Engine identifier (excluded from predictors)','none','Positive integer'],
 ['t','Observed operational age','cycles','Positive integer'],
 ['s2 / T24','LPC outlet total temperature','deg R','641.21-644.53'],
 ['s3 / T30','HPC outlet total temperature','deg R','1571.04-1616.91'],
 ['s4 / T50','LPT outlet total temperature','deg R','1382.25-1441.49'],
 ['s7 / P30','HPC outlet total pressure','psia','549.85-556.06'],
 ['s9 / Nc','Physical core speed','rpm','9021.73-9244.59'],
 ['s11 / Ps30','HPC outlet static pressure','psia','46.85-48.53'],
 ['setting_1..3','Operating metadata, excluded','Undeclared','Finite; FD001 only'],
 ['Other 15 sensors','Unused source measurements','Native units','Finite, excluded from model']
], [90,178,60,157])
p('Table 1. Native sensor units follow the source paper [2]. Numerical training extrema describe empirical support, not physical validity boundaries. Source settings lack an explicit unit declaration in the supplied README, so no interpretation is invented.', 'CaptionReport')
table(['Output field', 'Meaning and operational use'], [
 ['predicted_rul_cycles','max(0, raw prediction), a nonnegative estimate for maintenance review'],
 ['raw_rul_prediction','Unconstrained finite estimate, retained to expose clipping'],
 ['engine_id, cycle','Endpoint identity and observed age for traceability'],
 ['input_status, flags','Empirical sensor-envelope excursions and negative-output clipping'],
 ['model_version','Configuration fingerprint linking the estimate to experimental settings']
], [150,335])
p('<b>Invalid inputs.</b> Missing/nonfinite measurements, wrong column count, nonnumeric fields, duplicate engine-cycle pairs, cycle gaps, nonpositive primary sensors and insufficient history raise an explicit error. No imputation or silent row deletion is used. Soft range warnings retain valid but unfamiliar measurements. Unsupported subsets are rejected when declared; the caller remains responsible for declaring the correct regime.')
p('<b>Provenance.</b> FD001 has 20,631 training rows and 13,096 test rows. SHA-256 hashes freeze the supplied files. The NASA catalog permits public access but leaves the specific license unspecified [1]. Raw data stay outside public Git history; instructions and a checksum manifest make acquisition auditable.')
page()

heading('3. Computational Solution')
p('The workflow validates input, forms causal features, partitions engines, tunes models, freezes selection, refits on all training engines, and evaluates official endpoints. It records each step as inspectable CSV/JSON evidence. Model bundles support repeated prediction from the same validated input contract.')
heading('3.1 Target and features', 2)
p('For training engine i with verified failure cycle T_i, the target is RUL(i,t) = T_i - t. Failure is zero remaining cycles. The target is uncapped, preserving its engineering meaning. Official test histories are censored: their separate RUL file supplies truth at the last observed cycle. Using the final test cycle as failure would create incorrect zero labels.')
p('At cycle t, each selected sensor contributes its current value, trailing mean and least-squares slope over W observed samples. Together with age, this gives 19 features. Candidate W values are 10, 20 and 30 cycles, but every comparison starts at cycle 30. No observation after t enters its feature vector.')
table(['Quantity', 'Definition'], [
 ['Trailing mean','mean_s(t) = sum from j=0 to W-1 of s(t-W+1+j) / W'],
 ['Centered positions','q_j = j - (W-1)/2'],
 ['Trailing slope','slope_s(t) = sum(q_j s(t-W+1+j)) / sum(q_j squared)'],
 ['Feature vector','x(t) = [age, 6 current values, 6 means, 6 slopes]']
], [108,377])
p('Centered positions reduce cancellation relative to fitting against large absolute cycle numbers. Mean units equal sensor units; slopes use sensor-unit/cycle. Training uses every fifth supported window plus each engine endpoint. Inverse engine-window counts, normalized to mean one, give each engine equal total fitting weight. Both sensor models use this identical sampling rule.')
heading('3.2 Random forest and numerical implementation', 2)
p('The proposed nonlinear method averages 200 bootstrapped regression trees [4]. Each tree selects feature thresholds that reduce weighted squared error. Depth (8 or 12), minimum leaf size (5 or 10) and feature fraction (0.7 or 1.0) limit complexity. The forest captures nonlinear age/sensor interactions without requiring a large neural-network training pipeline. Seed 3001 and one computation thread control randomization and timing.')
p('A tree returns a weighted mean of observed training targets in its leaf. The forest therefore interpolates among learned responses and cannot reliably extrapolate to unseen life ranges. More trees reduce Monte Carlo variability but do not remove model bias. Tree stopping constraints are the relevant termination conditions; there is no iterative numerical residual to converge.')
p('The implementation separates parsing, feature construction, fitting, metrics and command-line orchestration in src/jet_rul. NumPy vectorization replaces a reference loop for the same feature formulas. Public functions contain docstrings and generate browsable pdoc HTML. The methods and data contracts are also documented in docs/METHOD.md.')
page()

heading('4. Alternative Solutions')
p('A sensor-ridge model is a credible linear alternative [3]. Its pipeline standardizes predictors using only fitting-engine rows, then solves a weighted penalized least-squares problem. The direct SVD solver handles correlated sensor/current/mean features more stably than explicitly inverting normal equations. The scaler is unweighted; the regression uses equal-engine weights.')
p('<b>Objective:</b> minimize sum_j w_j(y_j - b - z_j^T beta)^2 + alpha ||beta||^2. The intercept b is unpenalized. Alpha candidates are 0.1, 1, 10 and 100. A separate age-only ridge baseline measures how much predictive information exists without any sensor channel.')
rows = []
for k in ['age','ridge','forest']:
    best = cv.loc[cv.kind == k].sort_values('rmse').iloc[0]
    held = selection['holdout_metrics'][k]
    spec = selection['specs'][k]
    setting = f"W={spec['window']}, alpha={spec['alpha']:g}" if k != 'forest' else f"W={spec['window']}, depth={spec['depth']}, leaf={spec['leaf']}, fraction={spec['features']}"
    rows.append([k,setting,f'{best.rmse:.2f}',f"{held['rmse']:.2f}",f"{held['nasa_score']:.1f}"])
table(['Model', 'Selected CV settings', 'CV RMSE', 'Holdout RMSE', 'Holdout score'], rows, [55,203,72,78,77])
p('Table 2. Errors are in cycles. Cross-validation pools 80 independent engine endpoints; holdout results use 20 different engines. NASA score is dimensionless and sample-count dependent. All models use the same eligible endpoints.', 'CaptionReport')
p(f"<b>Frozen decision.</b> The forest had to reduce internal holdout RMSE by at least 5% relative to ridge without increasing NASA score. Its measured relative RMSE improvement was {100*selection['holdout_relative_rmse_improvement']:.2f}%. The rule selected <b>{chosen}</b>. This decision preceded official test evaluation, so a later test ranking cannot change it.")
table(['Criterion', 'Age-only', 'Sensor ridge', 'Random forest'], [
 ['Engineering information','Age only','Linear sensor/age effects','Nonlinear interactions'],
 ['Interpretability','Simple age trend','Coefficient signs require care with collinearity','Many local decision rules'],
 ['Extrapolation','Linear and possibly negative','Linear and possibly negative','Bounded by learned leaf responses'],
 ['Numerical handling','Scaling + SVD + penalty','Scaling + SVD + penalty','Depth/leaf bounds + ensemble averaging'],
 ['Implementation burden','Lowest','Low','Higher fitting/storage cost']
], [105,112,134,134])
p('Deep recurrent models and capped-RUL training are plausible future alternatives, but each would add tuning decisions or change the target. They are excluded to preserve a tractable, interpretable study. Published leaderboard scores are not claimed as directly comparable because capping, sensor selection and endpoint conventions often differ. The implemented alternatives provide a controlled local benchmark.')
page()

heading('5. Validation')
heading('5.1 Engine separation and verification', 2)
p('The 100 training engines are split into 80 development and 20 internal holdout engines. Five-fold cross-validation on the 80 development engines uses 64 fitting and 16 validation engines per fold. Each validation engine contributes one saved pseudo-endpoint chosen uniformly from cycle 30 through T_i-1. Its features use only observations available at that time. Fixed IDs and endpoint cycles are recorded in splits.json.')
p('The search contains 12 ridge and 24 forest configurations, plus four age-only alternatives. Validation RMSE pools all 80 endpoint errors. The internal holdout chooses between the CV winners, so it is a selection set. After that choice is saved, all models refit using all 100 complete training engines. The official evaluation then uses one endpoint from every one of the 100 separate test engines. No difficult engine is excluded.')
p('The preliminary data audit inspected source dimensions and test-label summaries. Model-selection code never parses official test observations or labels; it only verifies their file hashes. A synthetic integration test deliberately supplies nonnumeric test files and still completes model selection. Configuration changes after selection cause final evaluation to refuse execution.')
table(['Verification group', 'Evidence / expected property'], [
 ['Source and labels','26 columns, finite numbers, correct domains, no gaps/duplicates, known RUL sequence [3,2,1,0]'],
 ['Causal features','Analytic constant/linear signals, independent engine histories, future-data mutation invariance'],
 ['Numerical comparison','Vectorized and loop features agree to floating-point tolerance; collinear ridge fit remains finite'],
 ['Validation boundary','Disjoint/reproducible engine IDs, fit-only scaler, official-target parsing prevented'],
 ['Outputs and metrics','Known MAE/RMSE/bias, late penalty > early penalty, explicit overflow, persistence and clipping'],
 ['Execution',f'{test_count} pytest checks passed; machine-readable JUnit report and CI workflow supplied']
], [112,373])
heading('5.2 Metrics and acceptance criteria', 2)
p('For error e = predicted RUL - true RUL, MAE = mean(|e|), RMSE = sqrt(mean(e^2)) and bias = mean(e). Positive errors overestimate life. The NASA-style score sums exp(-e/13)-1 for e&lt;0 and exp(e/10)-1 otherwise. It penalizes late predictions more strongly: an error of +10 cycles scores 1.718, versus 1.158 for -10. The implementation uses expm1 for accuracy near zero.')
p('The supplied local paper prints constants inconsistent with its prose; the NASA NTRS version supports the explicit early=13, late=10 convention [2]. Stating and testing the formula avoids ambiguity. MAE &lt;=20 and RMSE &lt;=25 cycles were proposed engineering targets, not regulatory acceptance limits. Physical plausibility, support warnings and error bands remain relevant even when an average target is met.')
p('Paired bootstrap resampling uses 2,000 samples of the 100 engine endpoints. Its percentile intervals quantify metric uncertainty conditional on these fitted models. They are not individual-engine prediction bounds and exclude uncertainty from collecting another training fleet.')
page()

heading('5. Validation - official results')
rows = [[k, f'{metrics.loc[k,"mae"]:.2f}', f'{metrics.loc[k,"rmse"]:.2f}', f'{metrics.loc[k,"bias"]:.2f}', f'{metrics.loc[k,"nasa_score"]:.1f}', int(metrics.loc[k,'clipped_count'])] for k in ['age','ridge','forest']]
table(['Model', 'MAE', 'RMSE', 'Bias', 'NASA score', 'Clipped'], rows, [78,65,65,65,125,87])
p('Table 3. Official FD001 test endpoints (n=100). MAE/RMSE/bias are in cycles. Metrics use nonnegative output predictions; raw-prediction metrics are also saved. Lower error and score are better. Positive bias means optimistic lifespan estimates.', 'CaptionReport')
figure('predictions', 'Figure 2. Each dot represents a different official test engine. The dashed line is perfect endpoint prediction. No test outcomes were used to select model settings.')
key = 'a' if chosen == 'forest' else 'b'
# Bootstrap implementation stores model A as forest and model B as ridge.
ci_model = {'mae': intervals[f'{key}_mae'], 'rmse': intervals[f'{key}_rmse']}
p(f"The selected {chosen} model has MAE {m.mae:.2f} cycles with bootstrap 95% interval {ci(ci_model['mae'])}, and RMSE {m.rmse:.2f} cycles with interval {ci(ci_model['rmse'])}. The paired forest-minus-ridge RMSE difference interval is {ci(intervals['rmse_difference'])} cycles. An interval spanning zero does not establish a consistent advantage for either model on this test population.")
p(f"The selected model {'meets' if m.mae <=20 else 'misses'} the MAE target of 20 cycles and {'meets' if m.rmse <=25 else 'misses'} the RMSE target of 25 cycles. Relative to age-only, its test RMSE changes by {100*(1-m.rmse/metrics.loc['age','rmse']):.1f}%. The result demonstrates the value and limits of sensor information; it does not justify treating an individual forecast as a guaranteed safe life.")
p(f"<b>Safety-sensitive tradeoff.</b> Forest test NASA score is {metrics.loc['forest','nasa_score']:.1f}, worse than ridge's {metrics.loc['ridge','nasa_score']:.1f}, despite lower MAE and RMSE. The holdout score advantage did not generalize. The selected forest is retained to avoid test-driven reselection, but its optimistic tail errors weaken any maintenance-use claim.")
strata = pd.read_csv(A/'error_by_rul.csv')
rows = [[r.rul_band,int(r.n),f'{r.mae:.2f}',f'{r.rmse:.2f}',f'{r.bias:.2f}'] for r in strata.loc[strata.model == chosen].itertuples()]
table(['True-RUL band', 'Engines', 'MAE', 'RMSE', 'Bias'], rows, [140,80,85,90,90])
p('Table 4. Selected-model performance by actual remaining life. This retrospective stratification reveals where errors occur; true RUL is unavailable during normal inference and is never a predictor.', 'CaptionReport')
page()

heading('5. Validation - robustness and limitations')
figure('robustness', 'Figure 3. Fixed models on 20 internal holdout engines. Noise amplitudes are percentages of development sensor SD. A +1 SD T30 bias represents an illustrative calibration error, not a documented sensor specification.')
p('Short zero-mean perturbations and systematic bias test different failure mechanisms. Averaging reduces some measurement noise, while a persistent bias shifts the whole feature history. These studies characterize already frozen models and do not retune them. The small internal holdout limits the precision of sensitivity conclusions.')
worst = pred.assign(abs_error=(pred[chosen]-pred.true_rul).abs()).nlargest(3,'abs_error')
table(['Engine', 'True RUL', 'Predicted RUL', 'Signed error'], [[int(r.engine_id),f'{r.true_rul:.0f}',f'{getattr(r,chosen):.2f}',f'{getattr(r,chosen)-r.true_rul:+.2f}'] for r in worst.itertuples()], [100,115,135,135])
p('Table 5. Three largest absolute endpoint errors for the selected model, in cycles. These are disclosed failure examples, not discarded outliers.', 'CaptionReport')
p(f"The input-support audit flags {support['engines_with_warnings']} of 100 test engines for sensor-envelope excursions and/or clipping. A warning marks limited support, not proof of a failed sensor. All flagged engines remain in the metrics. The system returns a point estimate without calibrated predictive uncertainty.")
p('<b>Limits of engineering credibility.</b> FD001 is a simulated, restricted regime. Initial condition and wear variation can yield similar measured histories with different remaining lives. A point regressor cannot uniquely recover latent damage. Long-horizon uncertainty, nonmonotone successive forecasts, unmodeled fault modes and operation outside the training envelope are expected failure modes. No monotonicity or conservation constraint is imposed, and the available selected channels do not establish a full thermodynamic balance.')
p('The internal pseudo-endpoint distribution differs from the official censoring process. Bootstrap intervals assume the test engines are representative independent units. The single fixed train/holdout split and small fleet limit generality. The NASA score can be dominated by a few optimistic errors, so it accompanies MAE, RMSE, bias and case-level evidence.')
p('<b>Next experiment.</b> A separate future study should compare uncertainty-calibrated estimates and regime-aware features on a newly reserved validation set. Any adaptation after inspecting this test set requires new untouched evaluation data. Real deployment would additionally require field measurements, sensor calibration and an approved maintenance decision process.')
page()

heading('6. Optimization & Performance')
p(f"Measurements use Python {env['python']} on {env['platform']}, one thread and warm in-memory models. Prediction times exclude file/model loading. Thirty repetitions report median and interquartile range (IQR); three repetitions measure refitting. The machine processor identifier is {escape(env['processor'])}. Timings are hardware-dependent.")
rows=[]
for k in ['age','ridge','forest']:
    v=perf['models'][k]; t=v['model_one']; h=v['history_to_prediction']
    rows.append([k,f"{v['fit_median_seconds']:.4f}",f"{t['median_seconds']*1000:.3f}\n[{t['q25_seconds']*1000:.3f},{t['q75_seconds']*1000:.3f}]",f"{h['median_seconds']*1000:.2f}",f"{v['serialized_bytes']/1024:.1f}"])
table(['Model','Fit (s)','Model only (ms), IQR','History + model (ms)','File (KiB)'],rows,[60,78,146,112,89])
p('Table 6. One-engine prediction; history timing includes validation, trailing features and warnings. Batch-100 timings and all individual fit timings are saved in performance.json.', 'CaptionReport')
p(f"<b>Vectorization.</b> The explicit feature loop takes {perf['feature_reference']['median_seconds']:.4f} s versus {perf['feature_vectorized']['median_seconds']:.4f} s for the vectorized version on 20,631 rows: {perf['feature_speedup']:.2f}x speedup. Their maximum absolute element difference is {perf['feature_max_absolute_difference']:.2e}. This preserves the formula and predictions within numerical tolerance. cProfile records the remaining cumulative-time costs.")
rows=[[int(r.stride),'19 with trends' if r.trends else '7 current only',int(r.training_windows),f'{r.rmse:.2f}',f'{r.fit_seconds:.3f}'] for r in ablation.itertuples()]
table(['Stride','Feature set','Fit windows','RMSE (cycles)','Fit (s)'], rows, [55,140,100,105,85])
p('Table 7. Fixed-parameter forest ablation on the first 64/16-engine development fold. Changing sampling or feature count trades computation against accuracy. This diagnostic does not override frozen selection.', 'CaptionReport')
p('At stride 5, current-only features improve this fold RMSE from 29.30 to 27.83 cycles. Thus trends are not uniformly beneficial. Stride 5 reduces full-feature fit time from 12.25 to 2.41 s here. The small diagnostic fold motivates a future feature study, not a post-test change to the reported model.', 'CaptionReport')
v=perf['models']['ridge']; f=perf['models']['forest']
p(f"<b>Arithmetic and memory.</b> Ridge uses d=19: approximately {v['dot_and_intercept_flops']} FLOPs for its dot product/intercept plus {v['scaling_flops']} for separate scaling. A hypothetical fused cold calculation reads about {v['fused_cold_bytes']} bytes, or {v['fused_flops_per_byte']:.3f} FLOP/byte. This excludes Python, input checking and cache effects; sklearn is not claimed to implement that fused kernel. Forest endpoints average {f['mean_comparisons']:.1f} threshold comparisons and {f['averaging_flops']} aggregation FLOPs. Comparisons are not counted as FLOPs.")
p(f"The forest contains {f['total_nodes']:,} nodes and {f['node_storage_bytes']/1e6:.2f} MB of node/value arrays. A no-cache-reuse traversal estimate is {f['estimated_traversal_bytes']/1024:.1f} KiB per endpoint. Process current RSS is {perf['process_current_rss_mb']:.1f} MB and reported peak is {perf['process_peak_rss_mb']:.1f} MB ({perf['peak_memory_scope']}); these include the interpreter, libraries and data, rather than isolated model memory.")
p(f"Feature construction is O(N*6*W); balanced-tree fitting is approximately O(B*m*N*log N), and prediction O(B*depth). Low arithmetic intensity and irregular tree access explain why FLOPs alone predict runtime poorly. Tree counts 50-400 and three seeds are recorded. Across seeds 3001-3003 the development-fold forest RMSE ranges {seeds.rmse.min():.2f}-{seeds.rmse.max():.2f} cycles. Further optimization should target measured overhead, without sacrificing validation or expanding test-driven tuning.")
page()

heading('7. AI Use & Reflection')
p('<b>Tool and extent.</b> OpenAI Codex provided substantial assistance across experiment design, Python implementation, pytest tests, documentation, report writing, figures, presentation materials and repository preparation. Most code and prose in this prepared version were AI-generated. The student selected the jet-engine lifespan topic, supplied the NASA files and assessment brief, and specified the submission identity and GitHub repository. This is not presented as independently student-written work.')
p('<b>Purpose and checking.</b> AI assistance accelerated implementation and helped connect numerical analysis, machine learning and software reproducibility. Generated work was checked with analytic constant/linear signals, engine-separation and future-leakage tests, parser boundary cases, known metric values, independent official endpoint evaluation and measured profiling. Reported numbers are read from saved experiment artifacts. Those automated checks support correctness, but do not establish student understanding.')
p('<b>Concrete corrections.</b> The first invalid-value tests attempted to insert infinity or a fractional ID into integer-typed pandas fixtures. Pandas rejected the test setup before the project validator ran. Explicitly floating-point corrupt fixtures fixed the test itself. The supplied paper also contained contradictory scoring constants; the project retained an explicit formula supported by the NASA NTRS version and tested that optimistic errors receive the larger penalty. These cases show why plausible generated code or formulas require evidence.')
p('<b>Technical lessons supported by the evidence.</b> Engine-level validation addresses dependence that a random row split would hide. The censored test endpoint requires an external RUL label. Linear regression with regularization is a serious alternative when data are limited and features are strongly related. A forest adds computational cost and must justify that cost quantitatively. A measured model-selection result takes precedence over an initial preference for a more complex method.')
p('<b>Student responsibility.</b> The student must personally verify the source, understand the mathematics and code, judge whether the scope meets teaching-team expectations, and accept responsibility for the submitted claims. This generated report cannot demonstrate individual learning or oral-defense competence on its own. The remaining personal reflection must describe what the student actually learned, checked and changed after running and reviewing the project. No first-person learning experience is fabricated here.')
table(['Before assessment', 'Student-owned evidence to add'], [
 ['Reproduce','Record a personal run, environment and comparison with the supplied metrics'],
 ['Explain','Derive one trailing slope and one ridge/forest prediction mechanism without AI'],
 ['Critique','Choose an actual large-error engine and discuss what the model cannot infer'],
 ['Reflect','Add genuine examples of an AI suggestion accepted, corrected or rejected'],
 ['Rehearse','Deliver the 5-minute talk and answer the 3-minute defense without AI assistance']
], [105,380])
p('The repository includes docs/AI_USE.md with an assistance log and reflection prompts, and presentation/DEFENSE.md with technical practice questions. Commit history records genuine implementation stages rather than fabricated semester-long activity. Neither disclosure nor automated tests replace the individual learning requirement.')
page()

heading('References')
p('References are outside the nine-page main report.', 'CaptionReport')
refs=[
 ('[1] NASA, C-MAPSS Jet Engine Simulated Data, official catalog.', 'https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data'),
 ('[2] A. Saxena, K. Goebel, D. Simon and N. Eklund, Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation, 2008 International Conference on Prognostics and Health Management. NASA NTRS record 20090029214.', 'https://ntrs.nasa.gov/citations/20090029214'),
 ('[3] Scikit-learn documentation, Ridge regression and SVD solver.', 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html'),
 ('[4] Scikit-learn documentation, RandomForestRegressor.', 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html'),
 ('[5] Scikit-learn documentation, Common pitfalls and recommended practices: data leakage.', 'https://scikit-learn.org/stable/common_pitfalls.html'),
 ('[6] pdoc documentation, HTML API documentation generation.', 'https://pdoc.dev/docs/pdoc.html'),
 ('[7] pytest documentation, test reporting.', 'https://pytest.org/en/stable/how-to/output.html'),
]
for text,url in refs:
    p(text)
    p(f'<link href="{url}" color="#2463a6">{url}</link>', 'CaptionReport')
p('[8] MMA3001_2026_Project_Brief.pdf and MMA3001_Project_Brief_Student_Key_Points.pdf, supplied unit assessment documents, 2026. These define report, repository, oral presentation and reflection requirements.')
p('Source checks and preparation: October 2026. Source files remain attributable to their respective owners. Public accessibility is not treated as an explicit data redistribution license.')
heading('Reproducibility evidence',2)
p('Repository: https://github.com/Shanwei0111/MMA3001Project. The README provides environment setup and end-to-end commands. artifacts/ contains saved partitions, configuration, cross-validation records, frozen selection, official predictions, bootstrap intervals, robustness results, profiling and the JUnit test report. docs/html/ contains generated code documentation. scripts/build_report.py regenerates this report from the evidence.')
p(f"Configuration SHA-256: {selection['config_sha256']}", 'CaptionReport')


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#cdd8e2'))
    canvas.line(55,44,A4[0]-55,44)
    canvas.setFont('Helvetica',8)
    canvas.setFillColor(colors.HexColor('#46566c'))
    canvas.drawString(55,31,'Chua Shanwei  |  35393343  |  MMA3001')
    canvas.drawRightString(A4[0]-55,31,str(doc.page))


doc = SimpleDocTemplate(str(OUT/'Jet_Engine_RUL_Report.pdf'), pagesize=A4,
                        leftMargin=55,rightMargin=55,topMargin=43,bottomMargin=57,
                        title='Turbofan remaining useful life from causal sensor histories',
                        author='Chua Shanwei')
doc.build(story, onFirstPage=footer, onLaterPages=footer)
(OUT/'Jet_Engine_RUL_Report.md').write_text('\n\n'.join(md),encoding='utf-8')
print(OUT/'Jet_Engine_RUL_Report.pdf')
