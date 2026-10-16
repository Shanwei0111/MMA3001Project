/** Editable presentation authoring with the Codex bundled Artifact Tool.
 * Set ARTIFACT_SKILL_DIR and RUNTIME_PYTHON before running. Quantitative inputs
 * come from the same saved experiment artifacts as the Python report.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {Presentation, PresentationFile, FileBlob} from '@oai/artifact-tool';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const skill = process.env.ARTIFACT_SKILL_DIR;
if (!skill || !process.env.RUNTIME_PYTHON) throw new Error('Set ARTIFACT_SKILL_DIR and RUNTIME_PYTHON');
const {finalizePresentation, applyPresentationChartFont} = await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')).href);
const load = async n => JSON.parse(await fs.readFile(path.join(root,'artifacts',n),'utf8'));
const selection = await load('selection.json');
const perf = await load('performance.json');
const intervals = await load('bootstrap_intervals.json');
const csv = (await fs.readFile(path.join(root,'artifacts/test_metrics.csv'),'utf8')).trim().split(/\r?\n/);
const head = csv.shift().split(',');
const rows = csv.map(r => Object.fromEntries(r.split(',').map((v,i)=>[head[i],v])));
const metric = k => rows.find(r=>r.model===k);
const navy='#12334D', teal='#168575', blue='#2463A6', ink='#223345', muted='#54677C';
const p = Presentation.create({slideSize:{width:1280,height:720}});
p.theme.defaultFont='Arial';
const notes=[];
function text(s, value, x,y,w,h,size=28,color=ink,bold=false){
  const sh=s.shapes.add({name:value.slice(0,40),geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  sh.text=value;
  sh.text.style={typeface:'Arial',fontSize:size,color,bold};
  return sh;
}
function slide(title, subtitle=''){
  const s=p.slides.add();s.background.fill='#FFFFFF';
  text(s,title,64,46,1152,108,44,navy,true);
  if(subtitle)text(s,subtitle,64,148,1152,55,24,muted);
  text(s,'MMA3001  /  Chua Shanwei  /  35393343',64,665,1030,25,17,muted);
  text(s,String(p.slides.items.length),1160,665,56,25,17,muted);
  return s;
}
function note(s,seconds,script,sources){
  s.speakerNotes.text=`Timing: ${seconds} seconds.\n\n${script}\n\nSources: ${sources}`;
  notes.push({slide:p.slides.items.length,seconds,script,sources});
}
function chart(s,title,categories,values,color,x,y,w,h,unit){
  const c=s.charts.add('bar',{
    position:{left:x,top:y,width:w,height:h},title,
    titleTextStyle:{fontSize:24,bold:true,fill:ink},categories,
    series:[{name:unit,values:values.map(v=>Number(v.toFixed(5))),fill:color}],barOptions:{direction:'column',grouping:'clustered',gapWidth:90},
    hasLegend:false,chartFill:'#FFFFFF',plotAreaFill:'#FFFFFF',
    xAxis:{textStyle:{fontSize:21,fill:ink},majorGridlines:null},
    yAxis:{min:0,title:{text:unit,textStyle:{fontSize:19}},textStyle:{fontSize:18,fill:muted},majorGridlines:{fill:'#DDE5EC',width:1}},
    dataLabels:{showValue:true,position:'outEnd',numberFormatCode:'0.00',textStyle:{fontSize:22,bold:true,fill:ink}}
  });
  applyPresentationChartFont(c,{fontFamily:'Arial'});
  return c;
}

let s=slide('Jet-engine remaining useful life');
text(s,'NASA C-MAPSS FD001',64,220,1152,65,38,teal,true);
text(s,'A controlled comparison of sensor regression methods',64,320,1100,100,32,ink);
text(s,'Chua Shanwei\n35393343',64,490,1000,90,27,muted);
note(s,20,'This project estimates the remaining operational cycles of simulated turbofan engines. I compare a random forest with sensor-based ridge regression and an age-only baseline. The objective is a reproducible engineering comparison, including errors and computational cost. The estimates support a maintenance review in the simulated FD001 setting.','NASA catalog: https://data.nasa.gov/dataset/cmapss-jet-engine-simulated-data');

s=slide('Engineering problem and data','One operating condition and one HPC degradation mode');
text(s,'100 training engines\n20,631 observed cycles',64,230,535,110,35,teal,true);
text(s,'100 separate test engines\nCensored before failure',680,230,535,110,35,blue,true);
text(s,'Input: age plus six temperature, pressure and speed channels.\nAt least 30 contiguous observations from one engine.',64,405,1152,105,28);
text(s,'Output: remaining cycles, raw estimate and input warnings.\nOperational use: maintenance review within the simulated regime.',64,535,1152,100,28);
note(s,40,'FD001 contains one operating condition and HPC degradation. Each engine has a different initial condition and degradation history. Training histories reach failure, so remaining life is the failure cycle minus the current cycle. Test histories stop early, and their separate RUL file gives the correct endpoint truth. The six channels describe compressor and turbine temperatures, pressures and core speed. I retain the native units and absolute cycle number. A prediction needs at least thirty contiguous observations. Missing data and invalid histories raise errors. The result is remaining cycles, not flight hours or a certified replacement decision.','NASA C-MAPSS supplied README and data manifest; Saxena et al. 2008, https://ntrs.nasa.gov/citations/20090029214');

s=slide('Causal features and model comparison','No future measurements enter a prediction');
text(s,'19 predictors',64,225,445,65,42,teal,true);
text(s,'Age\n6 current values\n6 trailing means\n6 trailing slopes',64,315,445,240,30);
text(s,'Random forest',600,225,610,60,34,ink,true);
text(s,'200 trees, depth 8, minimum leaf 10\n30-cycle window, feature fraction 0.7',600,290,610,110,27);
text(s,'Credible baselines',600,440,610,55,34,ink,true);
text(s,'Standardized ridge, alpha = 100\nAge-only ridge',600,505,610,100,27);
note(s,55,'For each of six sensors, I use the latest value, trailing mean and least-squares slope, plus age. The slope uses centered local cycle positions, which improves numerical conditioning. Each feature uses only past and present observations. Training samples every fifth cycle and the final cycle, with equal total fitting weight per engine. Five-fold engine cross-validation chooses the window and model settings. The winning forest averages two hundred trees with depth eight and minimum leaf ten. Ridge is an important alternative because these features are correlated and the fleet is small. It standardizes using fitting rows only and solves regularized least squares with SVD. The age-only baseline tests whether the sensors add useful information. The target remains uncapped RUL.','artifacts/selection.json; docs/METHOD.md; sklearn Ridge and RandomForestRegressor documentation');

s=slide('Validation and reproducibility','Separate engines protect against leakage between adjacent cycles');
text(s,'80 development engines',64,230,1130,55,35,teal,true);
text(s,'Five folds select settings using one saved endpoint per engine.',64,292,1152,60,28);
text(s,'20 internal holdout engines',64,390,1130,55,35,blue,true);
text(s,'Forest needs at least 5% lower RMSE and no worse NASA score.',64,452,1152,60,28);
text(s,'37 passing tests   /   pdoc HTML   /   pinned Python environment',64,555,1152,50,26,ink,true);
text(s,'GitHub: Shanwei0111/MMA3001Project   |   src, tests, docs, artifacts',64,610,1152,38,22,muted);
note(s,45,'The eighty development engines support tuning. Twenty separate internal holdout engines apply the predeclared selection rule. The forest reduced holdout RMSE by ten point six percent and had a lower NASA score, so selection froze before official testing. All models then refit on the hundred training engines. Thirty-seven pytest checks cover analytic signals, labels, invalid inputs, engine separation, future-data leakage and persistence. An integration test prevents official target parsing during selection. The GitHub repository separates source, tests, data provenance, documentation and saved results. Its README supplies reproduction commands, and pdoc HTML explains the functions. Bootstrap uncertainty resamples entire test engines.','artifacts/splits.json, selection.json and test-results.xml; https://github.com/Shanwei0111/MMA3001Project');

s=slide('Official test results','100 engine endpoints, evaluated after model selection');
chart(s,'Endpoint RMSE',['Age-only','Ridge','Forest'],['age','ridge','forest'].map(k=>Number(metric(k).rmse)),teal,55,210,770,400,'Cycles');
text(s,`${Number(metric('forest').mae).toFixed(2)} cycles`,870,245,350,65,40,teal,true);
text(s,'Forest MAE',870,318,350,45,26,ink);
text(s,'NASA score\nForest 2,174.6\nRidge 1,670.4',870,420,350,140,27,ink,true);
text(s,'Lower average error, but a worse penalty for optimistic tail errors.',64,615,1152,42,25,ink,true);
note(s,65,'On the official test set, forest MAE is fourteen point six nine cycles and RMSE is twenty point zero four. Ridge RMSE is twenty-two point two nine, and age-only is thirty-four point eight nine. The selected forest meets the proposed MAE twenty and RMSE twenty-five targets. Its bootstrap RMSE interval is sixteen point three three to twenty-three point five six cycles. However, the paired forest-minus-ridge RMSE interval includes zero, so this fleet does not establish a robust RMSE advantage. There is also a crucial tradeoff: the forest NASA score is worse than ridge, despite lower average error. This score penalizes overestimating life more heavily. The holdout score advantage did not generalize. I retain the frozen choice and disclose the optimistic tail errors instead of choosing again on test outcomes. Metric intervals are not individual-engine confidence bounds.','artifacts/test_metrics.csv and bootstrap_intervals.json; scoring convention: https://ntrs.nasa.gov/api/citations/20090029214/downloads/20090029214.pdf');

s=slide('Performance and optimization','Measured on one machine with one computation thread');
chart(s,'Full training-set feature computation',['Reference loop','Vectorized'],[perf.feature_reference.median_seconds,perf.feature_vectorized.median_seconds],blue,55,210,750,400,'Seconds');
text(s,`${perf.feature_speedup.toFixed(2)}x speedup`,850,245,365,70,40,teal,true);
text(s,`Maximum feature difference\n${perf.feature_max_absolute_difference.toExponential(2)}`,850,340,365,100,25);
text(s,`Forest history-to-output\n${(perf.models.forest.history_to_prediction.median_seconds*1000).toFixed(2)} ms median`,850,485,365,100,25);
text(s,'Ridge: 38 dot-product FLOPs + 38 scaling FLOPs. Tree comparisons differ from FLOPs.',64,615,1152,43,23,ink);
note(s,40,'Vectorization reduces full-training-set feature time from zero point two nine one to zero point one eight three seconds, a one point five nine times speedup. The maximum difference from the reference formula is about one point nine times ten to the minus thirteen. The forest takes about twenty milliseconds from a thirty-observation history to its output on this machine. Ridge has far less arithmetic and storage cost. Its nineteen-feature dot product and scaling each need about thirty-eight FLOPs. Forest traversal instead uses roughly fifteen hundred threshold comparisons, plus two hundred aggregation FLOPs. Those comparisons and irregular memory accesses explain why a FLOP count alone is misleading.','artifacts/performance.json and profiles/features.txt; timings include warm call overhead and exclude loading');

s=slide('Limitations and AI reflection');
text(s,'Engineering limits',64,220,1130,60,35,teal,true);
text(s,'Simulated FD001 only. No real-fleet validation or calibrated RUL interval.\nOptimistic errors remain important even when average targets are met.',64,292,1152,110,28);
text(s,'Substantial AI assistance',64,445,1130,60,35,blue,true);
text(s,'Codex assisted with design, code, tests and written materials.\nThe student owns verification, interpretation and the oral defense.',64,517,1152,105,28);
note(s,35,'The model supports only the simulated FD001 regime. It does not enforce thermodynamic conservation or provide calibrated uncertainty for an individual engine. Some large optimistic errors remain, and systematic sensor bias can change forecasts. Codex provided substantial assistance in design, code, testing and these materials. Validation identified and corrected an invalid-value test-fixture problem and resolved a conflicting score formula against the NASA source. I must personally review and reproduce the work, add my genuine learning reflection and explain it without AI during the defense.','docs/AI_USE.md; supplied MMA3001 project brief, AI use and individual-defense requirements');

await fs.mkdir(path.join(root,'.private/slides'),{recursive:true});
await fs.mkdir(path.join(root,'presentation'),{recursive:true});
await fs.writeFile(path.join(root,'presentation/slide_content.json'),JSON.stringify(notes,null,2));
await fs.writeFile(path.join(root,'presentation/SPEAKER_NOTES.md'),'# Five-minute presentation\n\nTimings total 300 seconds. Rehearse and revise in your own words.\n\n'+notes.map(n=>`## Slide ${n.slide} (${n.seconds} seconds)\n\n${n.script}\n\nSources: ${n.sources}`).join('\n\n'));
const candidate=path.join(root,'.private/slides/candidate.pptx');
await(await PresentationFile.exportPptx(p)).save(candidate);
const finalPath=path.join(root,'presentation/Jet_Engine_RUL_Presentation.pptx');
const result=await finalizePresentation({workspaceDir:root,candidatePath:candidate,finalPath,
  pythonExecutable:process.env.RUNTIME_PYTHON,
  integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit'],
  explicitTotalSlideCount:7,requiredNativeChartOwnerSlides:[5,6],
  materializeLiteralChartWorkbooks:true,
  fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,
  receiptPath:path.join(root,'.private/slides/validation.json')});
console.log(JSON.stringify(result));
const final = await PresentationFile.importPptx(await FileBlob.load(finalPath));
for(const [index,sl] of final.slides.items.entries()){
  const blob=await final.export({slide:sl,format:'png',scale:1});
  await fs.writeFile(path.join(root,'.private/slides',`slide-${index+1}.png`),new Uint8Array(await blob.arrayBuffer()));
}
