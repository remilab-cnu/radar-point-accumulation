import os, base64
HERE = os.path.dirname(os.path.abspath(__file__)); DOCS = os.path.join(HERE, "..", "docs")
def b64(p): return base64.b64encode(open(os.path.join(DOCS, p), "rb").read()).decode()

html = f"""<h1>Why velocity accumulation loses to geometry on whole-body HAR</h1>
<p class="lead">Cross-subject: velocity beats geometry &amp; intensity on hand gestures (mHomeGes) but loses on
whole-body HAR (MM-Fi). This analyzes <b>why</b>, with evidence — including <b>refuting</b> our first hypothesis and
<b>correcting</b> the separability metric after an internal audit.</p>

<h2>1. The intuitive "motion-granularity" hypothesis is REFUTED</h2>
<p>Hypothesis (from the design): velocity should help for <i>localized</i> (fine) motion and fail for <i>gross</i>
whole-body motion. We tested it <b>within MM-Fi on the same sensor</b> (removing the 60 GHz / class-count confound):
data-drivenly split the 27 actions by vertical motion spread into FINE vs GROSS and re-ran cross-subject.</p>
<ul>
<li>FINE (localized): velocity 73.2 vs geometry 78.3 &rarr; <b>velocity&minus;geometry = &minus;5.1</b></li>
<li>GROSS (whole-body): velocity 84.0 vs geometry 85.6 &rarr; <b>velocity&minus;geometry = &minus;1.6</b></li>
<li>Per-class correlation (vertical motion spread vs velocity&rsquo;s recall edge): <b>r = 0.05 (none)</b></li>
</ul>
<p>Velocity loses to geometry on <b>both</b> subsets, and if anything the gap is <i>smaller</i> on gross motion —
the opposite of the hypothesis. So the boundary is not within-task motion granularity.</p>
<figure><img src="data:image/png;base64,{b64('mmfi_why.png')}"/>
<figcaption>Left: velocity loses to geometry on both FINE and GROSS MM-Fi subsets. Right: no correlation (r=0.05).</figcaption></figure>

<h2>2. The real driver: which arm is class-separable (audit-corrected)</h2>
<p><b>Audit note.</b> An earlier version claimed a Fisher inter/intra separability metric was the mechanism; the audit
showed it ties velocity=geometry (0.171=0.171) on gestures and thus fails to reproduce the +13 CNN win — so we drop it.
The honest, classifier-grounded metric is a parameter-free <b>nearest-class-mean (NCM) cross-subject accuracy</b> on the
raw maps, which <b>does reproduce the CNN direction on both datasets</b>:</p>
<table>
<tr><th>Cross-subject accuracy</th><th>Velocity</th><th>Geometry</th><th>Intensity</th></tr>
<tr><td>mHomeGes — deep CNN</td><td><b>67.5</b></td><td>54.4</td><td>44.6</td></tr>
<tr><td>mHomeGes — NCM (simple)</td><td><b>43.8</b></td><td>31.7</td><td>31.8</td></tr>
<tr><td>MM-Fi — deep CNN</td><td>70.4</td><td><b>77.5</b></td><td>77.0</td></tr>
<tr><td>MM-Fi — NCM (simple)</td><td>24.1</td><td>41.7</td><td><b>44.0</b></td></tr>
</table>
<p>Even a nearest-class-mean classifier prefers <b>velocity</b> on gestures (43.8 vs ~31.7) and <b>geometry/intensity</b>
on whole-body activity (velocity collapses to 24.1) — the reversal is not an artifact of the deep model.</p>
<p><b>Why velocity separates co-located gestures (descriptive, gesture side):</b> the class-mean maps' cosine overlap is
<b>0.100 for velocity vs 0.917 for geometry</b> on mHomeGes — velocity maps are nearly orthogonal across gesture classes
while geometry maps are almost identical (all a band at hand height). (This overlap metric explains the gesture side only;
it does not by itself capture the HAR reversal, so we rely on NCM there.)</p>
<figure><img src="data:image/png;base64,{b64('mmfi_why_mechanism.png')}"/>
<figcaption>Class-mean height-occupancy (geometry). Hand-gesture classes (top) look alike — all a band at hand height,
so geometry cannot separate them. Whole-body classes (bottom) have distinct height signatures — geometry separates them.</figcaption></figure>

<h2>3. Mechanism (honest, evidence-based)</h2>
<p><b>Velocity accumulation wins when discriminative information lives in MOTION rather than spatial/postural
configuration.</b> Hand gestures are performed in the same small volume, so where the reflections sit (occupancy /
intensity) cannot tell swipe-left from swipe-right — only the motion pattern (velocity) can, and even a simple NCM
classifier confirms velocity is the separable arm there. Whole-body activities instead produce distinct body-shape /
height footprints that geometry captures directly, so geometry overtakes velocity (NCM agrees). It is <i>not</i> that
velocity "fails" on HAR in absolute terms; rather geometry becomes the more separable cue and surpasses it.</p>
<p><b>Caveats stated honestly:</b> (i) MM-Fi uses a 60 GHz IWR6843 with coarser Doppler quantization (~0.6 vs 0.356 m/s)
which blunts velocity somewhat; (ii) no single hand-crafted scalar perfectly predicts the deep CNN — we therefore lead
with NCM (which tracks the direction) plus the class-mean visual, and avoid over-precise "separability-scaling" claims.</p>

<h2>4. Implication for the paper</h2>
<p>The scope boundary is a <b>property of the task, not of motion size</b>: velocity accumulation is advantageous for
<b>spatially co-located, motion-defined tasks</b> (hand gestures), where geometry is ambiguous — exactly the in-cabin
hand-gesture regime. We scope the paper's positive claim to gesture; whole-body HAR (MM-Fi) is reported as an explicit
boundary. This strengthens the positive claim: velocity is <i>needed</i> for gestures precisely because they are
geometrically ambiguous.</p>

<style>
:root{{color-scheme:light dark}} body{{max-width:60rem}}
.lead{{font-size:1.05rem;line-height:1.5;color:#444}} @media(prefers-color-scheme:dark){{.lead{{color:#bbb}}}}
h2{{margin-top:1.8rem}} figure{{margin:1rem 0}} img{{width:100%;border:1px solid #ccc;border-radius:6px}}
figcaption{{font-size:.85rem;color:#666;margin-top:.3rem}}
table{{border-collapse:collapse;margin:1rem 0}} th,td{{border:1px solid #bbb;padding:.35rem .7rem;text-align:center}}
</style>"""
open(os.path.join(DOCS, "why_har_reversal.html"), "w").write(html)
print("wrote", os.path.getsize(os.path.join(DOCS, "why_har_reversal.html")), "bytes")
