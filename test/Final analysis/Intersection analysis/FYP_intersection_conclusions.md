# Intersection scenario — conclusions (from `Intersection_analysis_compilation.xlsx`)

*Figures are **percentages of each policy’s own test runs**, so different run counts (e.g. 50 vs 39) do not bias the comparison. Trip times are simulation seconds among **successful** runs with a **recorded** finish time.*

**Sample sizes in this workbook:** IDM *n* = 50, Expert *n* = 39, IDM trajectory *n* = 40.

---

## Q1. Which policy reaches the destination most often?

**Conclusion:** **IDM** is the most reliable for task completion: **100%** of runs succeeded (`Pass = YES`). **IDM trajectory** is second (**90%** success). **Expert** is the weakest on this metric (**66.7%** success), meaning about **one in three** runs did not reach the goal under the same intersection scenario logging rules.

---

## Q2. Which policy fails most often?

**Conclusion:** **Expert** has the highest failure rate (**33.3%** of runs did not pass). **IDM trajectory** fails in **10%** of runs. **IDM** recorded **0%** failures in this dataset.

---

## Q3. Which policy gets stuck / times out most (too cautious or blocked)?

**Conclusion:** **Expert** shows the only substantial **timeout** rate in this compilation: **23.1%** of runs were flagged `Timeout = YES`, consistent with vehicles remaining nearly stopped for too long and not finishing. **IDM** and **IDM trajectory** show **0%** timeouts in these logs. So, for *this* scenario and metric, stuck behaviour is mainly an **Expert** issue, not IDM or trajectory IDM.

---

## Q4. Which policy is most “aggressive” (more crashes = worse in your framing)?

**Conclusion:** By **average crash count per run**, **IDM trajectory** is the most aggressive (**5.10** crashes per run on average). **IDM** is next (**2.82** mean). **Expert** has the **lowest** mean crashes (**1.28**).

If you use **“% of runs with at least one crash”** instead: **IDM** is highest (**66%** of runs had ≥1 crash), then **IDM trajectory** (**62.5%**), then **Expert** (**48.7%**). So Expert tends to have **fewer contact events** on average and a **smaller** share of runs with any crash, but it compensates with **lower success** and **more timeouts** (Q1–Q3).

---

## Q5. When a policy succeeds and a finish time exists, which is fastest?

**Conclusion:** Among successful runs with a recorded time, **IDM** is the fastest on average (**~50.6 s** simulation time; median **50.6 s**). **IDM trajectory** is next (**~53.9 s** mean, **52.8 s** median). **Expert** is the slowest when it does finish (**~59.9 s** mean, **57.8 s** median). *Note:* Expert has fewer timed successes (*n* = 26 in this pull) than IDM (*n* = 50), so compare time **together** with success and timeout results.

---

## Q6. Overall trade-off — what can you say for the FYP?

**Conclusion (one paragraph you can adapt):**  
On the **fixed intersection → roundabout** route with **multi-agent** traffic in MetaDrive, **IDM** maximises **reliability** (100% success, no timeouts) and **shortest** successful trip times, but a **large majority of runs** still record **at least one collision** (66%), so “finishing” does not mean “collision-free.” **Expert** is **more conservative in contact rate** (lower mean crashes and fewer runs with any crash) but **fails more often** and **timeouts** explain part of the reliability gap. **IDM trajectory** sits **between** IDM and Expert on **success** (90%) with **high** crash counts on average (most aggressive by mean crashes). For your report, state clearly that conclusions are **scenario-specific** (this map, traffic setup, and episode counts) and that **pass**, **timeout**, **crash count**, and **time** measure **different** things—no single policy wins on every axis.

---

## If your examiner asks: “What is the main finding?”

**Short answer:** Under your logged tests, **IDM** best **completes the task** and is **fastest** when timed; **Expert** is **gentler on crashes** but **worse at finishing** and **prone to timeouts** here; **IDM trajectory** is **intermediate** on success but **heaviest** on **mean** crashes.

---

*Regenerate or edit this file after you update the Excel compilation; numbers above match the workbook as loaded on the machine that produced this file.*
