"use strict";
/* GPUs and TPUs on Google Cloud: data and figures.
   No dependencies. Every chart is inline SVG drawn at the real CSS width of
   its container, so chart text stays at its stated pixel size on phones. */

/* =========================================================================
   DATA: one record per accelerator model on Google Cloud, 11 September 2026
   ========================================================================= */
const FAM = {
  blackwell: { name: "NVIDIA Blackwell", kind: "gpu", blurb: "2024–26. Blackwell GPUs add FP4 Tensor Core modes and up to 1.8 TB/s of bidirectional NVLink bandwidth per GPU." },
  hopper:    { name: "NVIDIA Hopper", kind: "gpu", blurb: "2022–24. The H100 introduced FP8 and the Transformer Engine; the H200 keeps the same die with 141 GB of HBM3e." },
  ampere:    { name: "NVIDIA Ampere", kind: "gpu", blurb: "2020. The A100 supports TF32, BF16, FP16 and INT8 Tensor Core modes, but not FP8." },
  pro:       { name: "NVIDIA graphics-class", kind: "gpu", blurb: "The L4 and RTX PRO 6000 use GDDR memory and PCIe. Google Cloud provides no NVLink between these GPUs." },
  legacy:    { name: "NVIDIA legacy, on N1 VMs", kind: "gpu", blurb: "Older accelerators attached to N1 VMs. P100 support ends in September 2026." },
  tpu:       { name: "Google TPU", kind: "tpu", blurb: "Google's own systolic-array chips, rented as VMs or pod slices. Programmed through JAX and PyTorch/XLA; no CUDA." },
};

// mem = GB per chip · bw = GB/s · fp8 = dense low-precision peak TFLOPS at the precision named in peakBasis (FP8, INT8, FP16 or BF16)
// bf16 = dense BF16/FP16 TFLOPS (bf16Basis names another precision where the chip has no fast 16-bit mode)
// vcpu / ram / net = largest documented VM shape: vCPUs, host RAM GB, host NIC Gbps (TPU: per-VM NIC, not per-chip DCN); null = not published
// price = $/chip-hr list in us-central1 (null = not public) · onDemand = sold on demand at that price; otherwise the price is indicative and priceBasis says how it is sold
// domain = chips in one fast fabric · perVM = chips-per-VM options. All throughput and bandwidth figures are vendor peaks and approximate.
const CHIPS = [
  { id:"gb300", name:"GB300 (B300)", short:"GB300", fam:"blackwell", series:"A4X Max · a4x-maxgpu-4g-metal", mem:279, memType:"HBM3e", bw:8000, fp8:5000, bf16:2500, price:null, onDemand:false, priceBasis:"reservation only", peakBasis:"FP8", domain:72, domainName:"NVL72 rack", perVM:"4 (bare metal)", vcpu:144, ram:960, net:3600, tag:"new", year:2026,
    notes:"Grace Blackwell Ultra superchips, bare metal only. 18 instances form one 72-GPU NVLink domain with about 20 TB of HBM. Reservation only." },
  // NVIDIA lists GB200 NVL72 at 720 PFLOPS FP8 and 360 PFLOPS FP16/BF16 with sparsity: 5,000 and 2,500 TFLOPS dense per GPU.
  { id:"gb200", name:"GB200", short:"GB200", fam:"blackwell", series:"A4X · a4x-highgpu-4g", mem:186, memType:"HBM3e", bw:8000, fp8:5000, bf16:2500, price:null, onDemand:false, priceBasis:"reservation or DWS", peakBasis:"FP8", domain:72, domainName:"NVL72 rack", perVM:"4", vcpu:140, ram:884, net:2000, tag:"new", year:2025,
    notes:"Two Grace CPUs (Arm) plus four Blackwell GPUs per VM, joined by NVLink-C2C. NVLink spans 72 GPUs across VMs. Sold through reservations and DWS; no on-demand list price." },
  { id:"b200", name:"B200", short:"B200", fam:"blackwell", series:"A4 · a4-highgpu-8g", mem:180, memType:"HBM3e", bw:8000, fp8:4500, bf16:2250, price:16.11, onDemand:false, priceBasis:"indicative: Spot, Flex-start or reservation", peakBasis:"FP8", domain:8, domainName:"8-GPU NVSwitch", perVM:"8", vcpu:224, ram:3968, net:3600, tag:"", year:2025,
    notes:"Eight B200 GPUs, an x86 host with 3.9 TB of RAM, and up to 3.6 Tbps of GPUDirect RDMA network bandwidth." },
  { id:"h200", name:"H200", short:"H200", fam:"hopper", series:"A3 Ultra · a3-ultragpu-8g", mem:141, memType:"HBM3e", bw:4800, fp8:1979, bf16:989, price:10.85, onDemand:false, priceBasis:"indicative: Spot, Flex-start or reservation", peakBasis:"FP8", domain:8, domainName:"8-GPU NVSwitch", perVM:"8", vcpu:224, ram:2952, net:3600, tag:"", year:2024,
    notes:"The Hopper architecture with 141 GB of HBM3e and 4.8 TB/s of peak memory bandwidth per GPU: more capacity and bandwidth than the H100." },
  { id:"h100m", name:"H100 (Mega)", short:"H100 Mega", fam:"hopper", series:"A3 Mega · a3-megagpu-8g", mem:80, memType:"HBM3", bw:3350, fp8:1979, bf16:989, price:11.0, onDemand:false, priceBasis:"indicative: Spot, Flex-start or reservation", peakBasis:"FP8", domain:8, domainName:"8-GPU NVSwitch", perVM:"8", vcpu:208, ram:1872, net:1800, tag:"", year:2024,
    notes:"H100 with 1.8 Tbps of GPUDirect-TCPXO between VMs, built for multi-node training. Sold through Spot, Flex-start and reservations." },
  { id:"h100", name:"H100 (High)", short:"H100", fam:"hopper", series:"A3 High · a3-highgpu-1g…8g", mem:80, memType:"HBM3", bw:3350, fp8:1979, bf16:989, price:10.98, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:8, domainName:"8-GPU NVSwitch", perVM:"1, 2, 4 (Spot/Flex) or 8", vcpu:208, ram:1872, net:1000, tag:"", year:2023,
    notes:"The only 8 × H100 shape with a plain on-demand price. Smaller 1-, 2- and 4-GPU shapes exist for Spot and Flex-start." },
  { id:"h100e", name:"H100 (Edge)", short:"H100 Edge", fam:"hopper", series:"A3 Edge · a3-edgegpu-8g", mem:80, memType:"HBM3", bw:3350, fp8:1979, bf16:989, price:10.98, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:8, domainName:"8-GPU NVSwitch", perVM:"8", vcpu:208, ram:1872, net:400, tag:"", year:2024,
    notes:"H100s with a smaller network (400 Gbps, 600 in two regions), placed in more regions for latency-sensitive inference." },
  { id:"a100_80", name:"A100 80 GB", short:"A100 80", fam:"ampere", series:"A2 Ultra · a2-ultragpu-1g…8g", mem:80, memType:"HBM2e", bw:2039, fp8:624, bf16:312, price:5.03, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:8, domainName:"8-GPU NVSwitch", perVM:"1, 2, 4, 8", vcpu:96, ram:1360, net:960, tag:"", year:2021,
    notes:"Supports INT8 Tensor Core throughput but not FP8. An 80 GB card holds roughly 80 billion one-byte weights before runtime and cache overhead." },
  { id:"a100_40", name:"A100 40 GB", short:"A100 40", fam:"ampere", series:"A2 Standard · a2-highgpu-1g…16g", mem:40, memType:"HBM2", bw:1555, fp8:624, bf16:312, price:3.67, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:16, domainName:"16-GPU NVSwitch (a2-megagpu)", perVM:"1, 2, 4, 8, 16", vcpu:96, ram:1360, net:384, tag:"", year:2020,
    notes:"The a2-megagpu-16g is the only 16-GPU NVLink shape on Google Cloud. Broadly available, on demand." },
  { id:"rtx6000", name:"RTX PRO 6000", short:"RTX PRO 6000", fam:"pro", series:"G4 · g4-standard-48…384", mem:96, memType:"GDDR7", bw:1600, fp8:1000, bf16:500, price:4.50, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:8, domainName:"PCIe (no NVLink)", perVM:"1, 2, 4, 8", vcpu:384, ram:1440, net:400, tag:"new", year:2025,
    notes:"Workstation-class Blackwell GPU with 96 GB of GDDR7, FP4 support and ray-tracing hardware. Multiple cards communicate over PCIe, without NVLink." },
  { id:"l4", name:"L4", short:"L4", fam:"pro", series:"G2 · g2-standard-4…96", mem:24, memType:"GDDR6", bw:300, fp8:242, bf16:121, price:0.70, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:8, domainName:"PCIe (no NVLink)", perVM:"1, 2, 4, 8", vcpu:96, ram:384, net:100, tag:"", year:2023,
    notes:"Ada Lovelace GPU with 24 GB of GDDR6 and a 72 W power limit, designed for inference, graphics and video." },
  { id:"t4", name:"T4", short:"T4", fam:"legacy", series:"N1 + nvidia-tesla-t4", mem:16, memType:"GDDR6", bw:320, fp8:130, bf16:65, price:0.35, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:4, domainName:"PCIe", perVM:"1, 2, 4", vcpu:96, ram:624, net:100, tag:"", year:2018,
    notes:"Turing GPU with 16 GB of GDDR6. The lowest on-demand GPU price in this dated table, and broad regional availability." },
  { id:"v100", name:"V100", short:"V100", fam:"legacy", series:"N1 + nvidia-tesla-v100", mem:16, memType:"HBM2", bw:900, fp8:125, bf16:125, price:2.48, onDemand:true, priceBasis:"on demand", peakBasis:"FP16", domain:8, domainName:"NVLink (P2P)", perVM:"1, 2, 4, 8", vcpu:96, ram:624, net:100, tag:"", year:2017,
    notes:"Volta was NVIDIA's first Tensor Core architecture. Current accelerators provide more memory and newer low-precision modes." },
  { id:"p100", name:"P100", short:"P100", fam:"legacy", series:"N1 + nvidia-tesla-p100", mem:16, memType:"HBM2", bw:732, fp8:21, bf16:21, price:1.46, onDemand:true, priceBasis:"on demand", peakBasis:"FP16", domain:4, domainName:"PCIe", perVM:"1, 2, 4", vcpu:96, ram:624, net:100, tag:"sunset", year:2016,
    notes:"Pascal, with no Tensor Cores. Reaches end of support in September 2026; do not start new work on it." },
  // The P4 has no fast FP16 path; its 5.5 TFLOPS figure is FP32.
  { id:"p4", name:"P4", short:"P4", fam:"legacy", series:"N1 + nvidia-tesla-p4", mem:8, memType:"GDDR5", bw:192, fp8:22, bf16:5.5, bf16Basis:"FP32", price:0.60, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:4, domainName:"PCIe", perVM:"1, 2, 4", vcpu:96, ram:624, net:100, tag:"", year:2016,
    notes:"Pascal inference card with INT8 as its fastest mode. Mostly used as a virtual-workstation GPU today." },

  { id:"v2", name:"TPU v2", short:"v2", fam:"tpu", series:"v2-8 … v2-512", mem:16, memType:"HBM", bw:700, fp8:45, bf16:45, price:1.50, onDemand:true, priceBasis:"on demand (pod rate)", peakBasis:"BF16", domain:512, domainName:"512-chip pod, 2D torus", perVM:"4 (v2-8) or slices", vcpu:null, ram:null, net:null, tag:"legacy", year:2017,
    notes:"The 2017 generation has two TensorCores and 16 GB of HBM per chip. A v2-8 device contains eight TensorCores across four chips. The Cloud TPU API is no longer under active development." },
  { id:"v3", name:"TPU v3", short:"v3", fam:"tpu", series:"v3-8 … v3-2048", mem:32, memType:"HBM2", bw:900, fp8:123, bf16:123, price:2.00, onDemand:true, priceBasis:"on demand (pod rate)", peakBasis:"BF16", domain:1024, domainName:"1,024-chip pod, 2D torus", perVM:"4 (v3-8) or slices", vcpu:null, ram:null, net:null, tag:"legacy", year:2018,
    notes:"Liquid-cooled successor to v2 with twice the HBM. Still rentable in us-central1 and europe-west4 for legacy JAX and TensorFlow jobs." },
  { id:"v4", name:"TPU v4", short:"v4", fam:"tpu", series:"v4-8 … v4-4096", mem:32, memType:"HBM2", bw:1200, fp8:275, bf16:275, price:3.22, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:4096, domainName:"4,096-chip pod, 3D mesh (torus per qualifying slice)", perVM:"4", vcpu:null, ram:null, net:null, tag:"", year:2022,
    notes:"The first TPU with optical circuit switches: the pod is a 3D mesh, and slices of qualifying shapes can be wired as a 3D torus or twisted torus. Two TensorCores with four MXUs each. Only in us-central2." },
  { id:"v5e", name:"TPU v5e", short:"v5e", fam:"tpu", series:"ct5lp-hightpu-1t…8t", mem:16, memType:"HBM2", bw:819, fp8:393, bf16:197, price:1.20, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:256, domainName:"256-chip pod, 2D torus", perVM:"1, 4, 8", vcpu:224, ram:384, net:200, tag:"", year:2023,
    notes:"One TensorCore per chip, 400 GB/s of bidirectional ICI, and configurations from one to 256 chips." },
  { id:"v5p", name:"TPU v5p", short:"v5p", fam:"tpu", series:"ct5p-hightpu-4t", mem:95, memType:"HBM2e", bw:2765, fp8:459, bf16:459, price:4.20, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:8960, domainName:"8,960-chip pod, 3D torus", perVM:"4", vcpu:208, ram:448, net:200, tag:"", year:2023,
    notes:"Two TensorCores, four SparseCores, and 1.2 TB/s of bidirectional ICI per chip. The largest schedulable job is 6,144 chips." },
  { id:"v6e", name:"TPU v6e Trillium", short:"Trillium", fam:"tpu", series:"ct6e-standard-1t…8t", mem:32, memType:"HBM3", bw:1638, fp8:1836, bf16:918, price:2.70, onDemand:true, priceBasis:"on demand", peakBasis:"INT8", domain:256, domainName:"256-chip pod, 2D torus", perVM:"1, 4, 8", vcpu:360, ram:1440, net:200, tag:"", year:2024,
    notes:"One TensorCore with 256 × 256 MXUs, and two SparseCores. The 256-chip pod uses a 2D torus." },
  { id:"tpu7x", name:"TPU7x Ironwood", short:"Ironwood", fam:"tpu", series:"tpu7x-standard-4t", mem:192, memType:"HBM3e", bw:7380, fp8:4614, bf16:2307, price:12.00, onDemand:true, priceBasis:"on demand", peakBasis:"FP8", domain:9216, domainName:"9,216-chip pod, 3D torus", perVM:"4", vcpu:224, ram:960, net:400, tag:"new", year:2026,
    notes:"Generally available since 31 March 2026. Two TensorCores plus four SparseCores; the first TPU with native FP8. A full pod holds 1.77 PB of HBM in one fabric." },
];

/* =========================================================================
   helpers
   ========================================================================= */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NS = "http://www.w3.org/2000/svg";
/** Create an SVG element, set its attributes, append it to parent. */
function S(parent, tag, attrs = {}, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text != null) e.textContent = text;
  if (parent) parent.appendChild(e);
  return e;
}
function logScale(d0, d1, r0, r1) {
  const a = Math.log10(d0), b = Math.log10(d1);
  return (v) => r0 + (Math.log10(v) - a) / (b - a) * (r1 - r0);
}
const kindOf = (c) => FAM[c.fam].kind;
const nf = (n) => n.toLocaleString("en-US");
const fmtBW = (gbs) => (gbs / 1000).toFixed(2);                      // GB/s -> "4.80" TB/s
const fmtMem = (v) => v >= 1000 ? (v / 1000).toFixed(1) + " TB" : v + " GB";
const fmt$ = (v) => v == null ? "n/a" : "$" + v.toFixed(2);
/** Call draw(width) now and whenever the element's width changes. */
function onWidth(el, draw) {
  let last = -1;
  const run = () => { const w = Math.round(el.clientWidth); if (w > 0 && w !== last) { last = w; draw(w); } };
  if ("ResizeObserver" in window) new ResizeObserver(run).observe(el); else addEventListener("resize", run);
  run();
}
/** A segmented control: exactly one button pressed. */
function seg(group, onPick) {
  $$("button", group).forEach((b) => b.addEventListener("click", () => {
    if (b.disabled) return;
    $$("button", group).forEach((o) => o.setAttribute("aria-pressed", String(o === b)));
    onPick(b);
  }));
}

/* =========================================================================
   Figure 1: GPU and TPU schematics (static)
   ========================================================================= */
(function anatomy() {
  const g = $("#anat-gpu");
  S(g, "text", { class: "label muted", x: 0, y: 16 }, "132 SMs");
  for (let i = 0; i < 132; i++) {
    const c = i % 12, r = Math.floor(i / 12), hi = r === 4 && c === 11;
    S(g, "rect", { x: c * 11, y: 26 + r * 11, width: 9, height: 9, fill: hi ? "var(--gpu)" : "var(--rule)" });
  }
  S(g, "line", { x1: 65.5, x2: 65.5, y1: 148, y2: 178, stroke: "var(--rule-2)" });
  S(g, "rect", { x: 0.5, y: 178.5, width: 130, height: 24, fill: "none", stroke: "var(--ink-3)" });
  S(g, "text", { class: "label", x: 65, y: 195, "text-anchor": "middle" }, "HBM");
  S(g, "text", { class: "label muted", x: 0, y: 222 }, "80 GB, 3.35 TB/s");
  // zoom lines from the highlighted SM to its enlargement
  S(g, "line", { x1: 130, y1: 70, x2: 170, y2: 26.5, stroke: "var(--rule-2)" });
  S(g, "line", { x1: 130, y1: 79, x2: 170, y2: 255.5, stroke: "var(--rule-2)" });
  S(g, "rect", { x: 170.5, y: 26.5, width: 229, height: 229, fill: "none", stroke: "var(--gpu)" });
  S(g, "text", { class: "label", x: 182, y: 47, "font-weight": 600 }, "One SM");
  S(g, "text", { class: "label muted", x: 182, y: 70 }, "Warps, 32 threads each");
  for (let i = 0; i < 8; i++) S(g, "rect", { x: 182 + i * 26, y: 78, width: 20, height: 13, fill: i === 1 || i === 5 ? "var(--gpu)" : "var(--rule)" });
  S(g, "text", { class: "label muted", x: 182, y: 110 }, "2 issuing, 6 waiting for data");
  const rows = ["Registers", "Arithmetic units", "Tensor Cores", "Shared memory"];
  rows.forEach((t, i) => {
    const y = 124 + i * 31;
    S(g, "line", { x1: 182, x2: 388, y1: y + 0.5, y2: y + 0.5, stroke: "var(--rule)" });
    S(g, "text", { class: "label", x: 182, y: y + 21, "font-weight": t === "Tensor Cores" ? 600 : 400 }, t === "Tensor Cores" ? "Tensor Cores: matrix units" : t);
  });

  const t = $("#anat-tpu");
  S(t, "text", { class: "label muted", x: 0, y: 16 }, "A v4 chip has two TensorCores");
  S(t, "rect", { x: 0.5, y: 26.5, width: 268, height: 229, fill: "none", stroke: "var(--tpu)" });
  S(t, "text", { class: "label", x: 12, y: 47, "font-weight": 600 }, "One TensorCore");
  for (let i = 0; i < 4; i++) {
    const x0 = 12 + i * 64, y0 = 58, s = 54;
    for (let k = 1; k < 8; k++) {
      const d = k * s / 8;
      S(t, "line", { x1: x0 + d, x2: x0 + d, y1: y0, y2: y0 + s, stroke: "var(--tpu)", "stroke-opacity": 0.35 });
      S(t, "line", { x1: x0, x2: x0 + s, y1: y0 + d, y2: y0 + d, stroke: "var(--tpu)", "stroke-opacity": 0.35 });
    }
    S(t, "rect", { x: x0 + 0.5, y: y0 + 0.5, width: s - 1, height: s - 1, fill: "none", stroke: "var(--tpu)" });
  }
  S(t, "text", { class: "label muted", x: 12, y: 134 }, "4 MXUs, each 128 × 128 cells");
  const units = [["Vector unit", "activations, softmax"], ["Scalar unit", "control flow, addresses"]];
  units.forEach(([a, b], i) => {
    const y = 148 + i * 34;
    S(t, "line", { x1: 12, x2: 257, y1: y + 0.5, y2: y + 0.5, stroke: "var(--rule)" });
    const tx = S(t, "text", { class: "label", x: 12, y: y + 22 });
    S(tx, "tspan", {}, a);
    S(tx, "tspan", { class: "label muted", dx: 6 }, "· " + b);
  });
  S(t, "line", { x1: 12, x2: 257, y1: 216.5, y2: 216.5, stroke: "var(--rule)" });
  S(t, "text", { class: "label muted", x: 12, y: 239 }, "BF16 multiplies, FP32 sums");
  S(t, "line", { x1: 269, x2: 310, y1: 120.5, y2: 120.5, stroke: "var(--rule-2)" });
  S(t, "rect", { x: 310.5, y: 100.5, width: 89, height: 40, fill: "none", stroke: "var(--ink-3)" });
  S(t, "text", { class: "label", x: 355, y: 125, "text-anchor": "middle" }, "HBM");
  S(t, "text", { class: "label muted", x: 355, y: 162, "text-anchor": "middle" }, "32 GB per chip");
  S(t, "text", { class: "label muted", x: 355, y: 180, "text-anchor": "middle" }, "1.2 TB/s");
})();

/* =========================================================================
   Figure 2: systolic array filling, as small multiples
   ========================================================================= */
(function systolic() {
  const host = $("#systolic");
  const N = 6, p = 17, cs = 15, x0 = 98, y0 = 6;
  const frames = [
    [1, "Cycle 1", "1 of 36 cells busy. The last row starts 5 cycles later."],
    [4, "Cycle 4", "10 of 36 busy: the array fills along a diagonal."],
    [11, "Cycle 11 onward", "36 of 36 busy, one multiply-accumulate each per cycle."],
  ];
  frames.forEach(([t, head, sub]) => {
    const d = document.createElement("div");
    d.dataset.cycle = t;
    const h = document.createElement("h4"); h.textContent = head; d.appendChild(h);
    const svg = S(d, "svg", { viewBox: "0 0 200 112", role: "img", "aria-label": `${head}: ${sub}` });
    let busy = 0;
    for (let i = 0; i < N; i++) {
      const front = t - 1 - i;                 // column the newest operand of row i has reached
      for (let j = 0; j < N; j++) {
        const on = j <= front;
        if (on) busy++;
        S(svg, "rect", on
          ? { class: "cell busy", x: x0 + j * p, y: y0 + i * p, width: cs, height: cs, fill: "var(--tpu)" }
          : { class: "cell", x: x0 + j * p + 0.5, y: y0 + i * p + 0.5, width: cs - 1, height: cs - 1, fill: "none", stroke: "var(--rule-2)" });
      }
      if (front < 0) {                          // operand still queued outside the array
        const cx = x0 + front * p + cs / 2, cy = y0 + i * p + cs / 2;
        S(svg, "line", { x1: cx + 4, x2: x0 - 3, y1: cy, y2: cy, stroke: "var(--rule-2)" });
        S(svg, "circle", { cx, cy, r: 3.5, fill: "var(--ink-3)" });
      }
    }
    d.dataset.busy = busy;
    const ps = document.createElement("p"); ps.className = "sub"; ps.textContent = sub; d.appendChild(ps);
    host.appendChild(d);
  });
})();

/* =========================================================================
   Figure 3: the lineup table with expandable rows
   ========================================================================= */
(function lineup() {
  const tb = $("#lineup-table tbody");
  const maxMem = Math.max(...CHIPS.map((c) => c.mem));
  const order = ["blackwell", "hopper", "ampere", "pro", "legacy", "tpu"];
  const host = (c) => c.vcpu == null ? "not published for this generation"
    : `${c.vcpu} vCPU · ${fmtMem(c.ram)} RAM · ${c.net >= 1000 ? c.net / 1000 + " Tbps" : c.net + " Gbps"} host network`;
  const row = (c) => {
    const tag = c.tag === "sunset" ? `<span class="aside">support ends Sep 2026</span>` : c.tag === "legacy" ? `<span class="aside">legacy</span>` : "";
    const price = c.price == null ? `<span class="aside">not public</span><span class="ast"></span>` : fmt$(c.price) + `<span class="ast">${c.onDemand ? "" : "*"}</span>`;
    return `<tr class="chip-row" data-id="${c.id}">
      <th scope="row"><button type="button" class="namebtn" aria-expanded="false" aria-controls="d-${c.id}"><span class="nm">${c.name}</span></button>${tag}</th>
      <td class="n">${c.mem}</td>
      <td class="bar"><span style="--c:var(--${kindOf(c)});width:${(c.mem / maxMem * 100).toFixed(1)}%"></span></td>
      <td class="n">${fmtBW(c.bw)}</td>
      <td class="n">${nf(c.fp8)}<span class="basis">${c.peakBasis}</span></td>
      <td class="n">${nf(c.bf16)}<span class="basis">${c.bf16Basis || ""}</span></td>
      <td class="n">${nf(c.domain)}</td>
      <td class="n">${price}</td>
      <td class="series">${c.series.split(" · ")[0]}</td>
    </tr>
    <tr class="detail" id="d-${c.id}" hidden><td colspan="9"><div class="sticky-cell">
      <p>${c.notes}</p>
      <dl class="kv">
        <dt>Machine type</dt><dd>${c.series}</dd>
        <dt>Memory</dt><dd>${c.mem} GB ${c.memType} at ${fmtBW(c.bw)} TB/s</dd>
        <dt>Chips per VM</dt><dd>${c.perVM}</dd>
        <dt>Fast domain</dt><dd>${nf(c.domain)} chips · ${c.domainName}</dd>
        <dt>Host, largest shape</dt><dd>${host(c)}</dd>
        <dt>Pricing</dt><dd>${c.price == null ? "no public price; " : fmt$(c.price) + " per chip-hour, "}${c.priceBasis}</dd>
        <dt>First on Google Cloud</dt><dd>${c.year}</dd>
      </dl></div></td></tr>`;
  };
  tb.innerHTML = order.map((fk) => {
    const f = FAM[fk];
    return `<tr class="group"><th colspan="9" scope="colgroup"><span class="sticky-cell"><span class="fam">${f.name}</span> <span class="blurb">${f.blurb}</span></span></th></tr>`
      + CHIPS.filter((c) => c.fam === fk).map(row).join("");
  }).join("");
  tb.addEventListener("click", (e) => {
    const b = e.target.closest(".namebtn"); if (!b) return;
    const open = b.getAttribute("aria-expanded") !== "true";
    b.setAttribute("aria-expanded", String(open));
    document.getElementById(b.getAttribute("aria-controls")).hidden = !open;
  });
})();

/* =========================================================================
   Figure 4: the roofline for one chip
   ========================================================================= */
(function roofline() {
  const sel = $("#roof-chip");
  [["NVIDIA GPUs", "gpu"], ["Google TPUs", "tpu"]].forEach(([label, k]) => {
    const og = document.createElement("optgroup"); og.label = label;
    CHIPS.filter((c) => kindOf(c) === k).forEach((c) => { const o = document.createElement("option"); o.value = c.id; o.textContent = c.name; og.appendChild(o); });
    sel.appendChild(og);
  });
  sel.value = "h200";
  const ai = $("#roof-ai"), out = $("#roof-ai-out"), verdict = $("#roof-verdict"), presets = $("#roof-presets");
  const host = $("#roof-host"), svg = $("#roofsvg");
  let I = 128, W = 592, preset = 128;   // preset: the intensity of the pressed preset button, or null after a slider move
  const fmtI = (v) => v >= 100 ? nf(Math.round(v)) : String(Math.round(v * 10) / 10);
  const fmtT = (v) => v >= 10 ? nf(Math.round(v)) : v.toFixed(1);

  function draw() {
    const c = CHIPS.find((d) => d.id === sel.value), k = kindOf(c);
    const peak = c.fp8, bwTB = c.bw / 1000, rp = peak / bwTB;
    const H = W < 480 ? 320 : 360, m = { l: 50, r: 12, t: 30, b: 44 };
    const x = logScale(1, 10000, m.l, W - m.r), y = logScale(0.1, 10000, H - m.b, m.t);
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H);
    svg.replaceChildren();
    const grid = S(svg, "g", { class: "grid" });
    [0.1, 1, 10, 100, 1000, 10000].forEach((v) => S(grid, "line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
    const ax = S(svg, "g", { class: "axis" });
    S(ax, "line", { x1: m.l, x2: W - m.r, y1: H - m.b + 0.5, y2: H - m.b + 0.5 });
    [0.1, 1, 10, 100, 1000, 10000].forEach((v) => S(ax, "text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, v < 1 ? "0.1" : nf(v)));
    // x ticks; the ridge point gets its own tick, and plain ticks that would collide with it are dropped
    [1, 10, 100, 1000, 10000].forEach((v) => {
      if (Math.abs(x(v) - x(rp)) < 46) return;
      S(ax, "text", { x: x(v), y: H - m.b + 16, "text-anchor": v === 1 ? "start" : v === 10000 ? "end" : "middle" }, nf(v));
    });
    S(svg, "line", { class: "ridge-line", x1: x(rp), x2: x(rp), y1: y(peak), y2: H - m.b + 4 });
    S(svg, "text", { class: "annot", x: x(rp), y: H - m.b + 17, "text-anchor": "middle", "font-size": 13, fill: "var(--ink)", id: "roof-ridge" }, `ridge ${nf(Math.round(rp))}`);
    S(svg, "text", { class: "axis-title", x: W - m.r, y: H - 6, "text-anchor": "end" }, "Arithmetic intensity, FLOP per byte (log scale)");
    S(svg, "text", { class: "axis-title", x: 0, y: 13 }, "Attainable TFLOPS (log scale)");

    // the roof: the bandwidth slope, then the flat compute peak, each labeled with its regime
    S(svg, "path", { class: "roof-line", stroke: `var(--${k})`, d: `M${x(1)},${y(bwTB)}L${x(rp)},${y(peak)}L${x(10000)},${y(peak)}` });
    S(svg, "text", { class: "label", x: W - m.r, y: y(peak) - 7, "text-anchor": "end" }, `compute-bound · ${c.peakBasis} peak ≈ ${nf(peak)} TFLOPS`);
    const att = Math.min(peak, I * bwTB), bound = I < rp ? "memory" : "compute";
    const share = att / peak * 100, pct = share < 1 ? "under 1%" : Math.round(share) + "%";
    const slopeTxt = `memory-bound · ${fmtBW(c.bw)} TB/s × intensity`;
    const dx = x(rp) - x(1), dy = y(peak) - y(bwTB), len = Math.hypot(dx, dy), need = slopeTxt.length * 6.3 + 16;
    if (len > need) {
      // along the slope, on the stretch away from the workload dot
      const f = Math.log10(I) / Math.log10(rp), half = need / 2 / len;
      const want = bound === "memory" && f < 0.5 ? 0.72 : 0.3;
      const at = Math.min(1 - half - 0.03, Math.max(half + 0.03, want));
      const iv = 10 ** (at * Math.log10(rp)), ang = Math.atan2(dy, dx) * 180 / Math.PI;
      const gl = S(svg, "g", { transform: `translate(${x(iv)},${y(iv * bwTB)}) rotate(${ang})` });
      S(gl, "text", { class: "label muted", x: 0, y: 15, "text-anchor": "middle" }, slopeTxt);
    } else {
      S(svg, "text", { class: "label muted", x: m.l + 6, y: m.t + 14 }, slopeTxt);
    }
    // the workload
    const px = x(I), py = y(Math.max(0.1, att));
    S(svg, "line", { class: "drop-line", x1: px, x2: px, y1: py, y2: H - m.b });
    S(svg, "circle", { cx: px, cy: py, r: 5, fill: "var(--ink)", stroke: "var(--paper)", "stroke-width": 1.5, id: "roof-dot" });
    if (bound === "memory") {
      // right of the dot sits under the slope; if there is no room, go above-left of the dot, over the slope
      const lbl = `${fmtT(att)} TFLOPS, ${pct} of peak`, room = W - m.r - px > lbl.length * 6.9 + 12;
      S(svg, "text", { class: "label halo", "font-weight": 600, x: room ? px + 10 : px - 8, y: room ? py + 16 : py - 10, "text-anchor": room ? "start" : "end" }, lbl);
    } else {
      // on the flat roof: centered under the dot, kept right of the ridge
      const lbl = `${pct} of peak`, w = lbl.length * 6.9;
      const cx = Math.max(x(rp) + 6 + w / 2, Math.min(px, W - m.r - w / 2));
      S(svg, "text", { class: "label halo", "font-weight": 600, x: cx, y: py + 20, "text-anchor": "middle" }, lbl);
    }

    out.textContent = fmtI(I);
    $$("button", presets).forEach((b) => b.setAttribute("aria-pressed", String(preset !== null && +b.dataset.ai === preset)));
    verdict.innerHTML = `<b>${c.name}</b>: dense ${c.peakBasis} peak ≈ ${nf(peak)} TFLOPS ÷ ${fmtBW(c.bw)} TB/s puts the ridge at about ${nf(Math.round(rp))} FLOP/byte. At ${fmtI(I)} FLOP/byte the workload is <b>${bound}-bound</b> and can use at most ${pct} of peak.`
      + (bound === "memory" ? " Bigger batches, quantized weights or a chip with more bandwidth per FLOP would help." : " Higher compute throughput can raise this bound.");
    svg.setAttribute("aria-label", `Roofline for ${c.name}: ridge at ${Math.round(rp)} FLOP per byte; workload at ${fmtI(I)} FLOP per byte is ${bound}-bound at ${pct} of peak.`);
  }
  sel.addEventListener("change", draw);
  ai.addEventListener("input", () => { I = 10 ** +ai.value; preset = null; draw(); });
  $$("button", presets).forEach((b) => b.addEventListener("click", () => { I = preset = +b.dataset.ai; ai.value = Math.log10(I); draw(); }));
  ai.value = Math.log10(I);
  onWidth(host, (w) => { W = w; draw(); });
})();

/* =========================================================================
   Figure 5: peak compute against bandwidth for every chip
   ========================================================================= */
(function scatter() {
  const host = $("#scatter-host"), svg = $("#scatter");
  // label text, dx, dy, anchor; null = the label is carried by a coincident point
  const LAB = {
    gb300: ["GB300 · GB200", -9, -9, "end"], gb200: null, b200: ["B200", 8, 15, "start"], tpu7x: ["TPU7x Ironwood", -10, 6, "end"],
    h200: ["H200", 8, 4], h100: ["H100 High · Mega · Edge", 0, -11, "middle"], h100m: null, h100e: null,
    rtx6000: ["RTX PRO 6000", 8, 4], v6e: ["TPU v6e", -9, 4, "end"], a100_80: ["A100 80 GB", 8, 4], a100_40: ["A100 40 GB", -9, 4, "end"],
    v5p: ["TPU v5p", 8, 4], v5e: ["TPU v5e", -9, 4, "end"], v4: ["TPU v4", 8, 4], v100: ["V100 · TPU v3", 8, 4], v3: null,
    l4: ["L4", 8, 4], t4: ["T4", 8, 4], v2: ["TPU v2", 8, 4], p100: ["P100", 8, 4], p4: ["P4", 8, 4],
  };
  function draw(cw) {
    const W = Math.max(cw, 576), H = 440, m = { l: 52, r: 48, t: 30, b: 44 };
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", W); svg.setAttribute("height", H); svg.style.width = W + "px";
    svg.replaceChildren();
    const X0 = 150, X1 = 10000, Y0 = 10, Y1 = 10000;
    const x = logScale(X0, X1, m.l, W - m.r), y = logScale(Y0, Y1, H - m.b, m.t);
    const grid = S(svg, "g", { class: "grid" });
    [10, 100, 1000, 10000].forEach((v) => S(grid, "line", { x1: m.l, x2: W - m.r, y1: y(v), y2: y(v) }));
    const ax = S(svg, "g", { class: "axis" });
    S(ax, "line", { x1: m.l, x2: W - m.r, y1: H - m.b + 0.5, y2: H - m.b + 0.5 });
    [200, 500, 1000, 2000, 5000, 10000].forEach((v) => S(ax, "text", { x: x(v), y: H - m.b + 16, "text-anchor": "middle" }, nf(v)));
    [10, 100, 1000, 10000].forEach((v) => S(ax, "text", { x: m.l - 6, y: y(v) + 4, "text-anchor": "end" }, nf(v)));
    S(svg, "text", { class: "axis-title", x: W - m.r, y: H - 6, "text-anchor": "end" }, "Memory bandwidth, GB/s (log scale)");
    S(svg, "text", { class: "axis-title", x: 0, y: 13 }, "Dense peak TFLOPS (log scale)");
    // lines of equal ridge point: TFLOPS = r × TB/s
    const iso = S(svg, "g", { class: "iso" });
    [[100, 1900, 14], [300, 3200, 14], [1000, 400, -5]].forEach(([r, at, off]) => {
      const xa = Math.max(X0, Y0 * 1000 / r), xb = Math.min(X1, Y1 * 1000 / r);
      S(iso, "line", { x1: x(xa), y1: y(r * xa / 1000), x2: x(xb), y2: y(r * xb / 1000) });
      const ang = Math.atan2(y(r * xb / 1000) - y(r * xa / 1000), x(xb) - x(xa)) * 180 / Math.PI;
      const gl = S(svg, "g", { transform: `translate(${x(at)},${y(r * at / 1000)}) rotate(${ang})` });
      S(gl, "text", { class: "annot", x: 0, y: off }, `${nf(r)} FLOP/byte`);
    });
    CHIPS.forEach((c) => S(svg, "circle", { cx: x(c.bw), cy: y(c.fp8), r: 5, class: `dot-${kindOf(c)}`, stroke: "var(--paper)", "stroke-width": 1, "data-id": c.id }));
    CHIPS.forEach((c) => {
      const L = LAB[c.id]; if (!L) return;
      S(svg, "text", { class: "label", x: x(c.bw) + L[1], y: y(c.fp8) + L[2], "text-anchor": L[3] || "start" }, L[0]);
    });
  }
  onWidth(host, draw);
})();

/* =========================================================================
   Figure 6: interconnect topologies, as small multiples (static)
   ========================================================================= */
(function topology() {
  const [a, b, c, d] = $$("#topos svg");
  const LINK = "var(--ink-3)";
  // 8 GPUs on one NVSwitch
  S(a, "rect", { x: 10.5, y: 14.5, width: 199, height: 28, fill: "none", stroke: "var(--ink-2)" });
  S(a, "text", { class: "label topo", x: 110, y: 34, "text-anchor": "middle" }, "NVSwitch");
  for (let i = 0; i < 8; i++) {
    const cx = 22 + i * 25.14;
    S(a, "line", { x1: cx, x2: cx, y1: 43, y2: 90, stroke: LINK });
    S(a, "rect", { x: cx - 8, y: 90, width: 16, height: 16, class: "gpu-fill" });
  }
  S(a, "text", { class: "label muted topo", x: 110, y: 128, "text-anchor": "middle" }, "one VM");
  // NVL72: 18 VMs × 4 GPUs on the rack's NVLink switches
  S(b, "rect", { x: 6.5, y: 8.5, width: 207, height: 28, fill: "none", stroke: "var(--ink-2)" });
  S(b, "text", { class: "label topo", x: 110, y: 28, "text-anchor": "middle" }, "NVLink switches");
  for (let col = 0; col < 18; col++) {
    const cx = 6 + col * 11.5 + 5.75;
    S(b, "line", { x1: cx, x2: cx, y1: 37, y2: 56, stroke: LINK });
    for (let r = 0; r < 4; r++) S(b, "rect", { x: cx - 4, y: 56 + r * 11, width: 8, height: 8, class: "gpu-fill" });
  }
  S(b, "path", { d: "M7.5,103v5h10v-5", fill: "none", stroke: "var(--ink-3)" });
  S(b, "text", { class: "label muted topo", x: 6, y: 128 }, "1 VM = 4 GPUs");
  // 2D torus, 4 × 4
  const p = 26, ox = 71, oy = 22, n = 4;
  for (let i = 0; i < n; i++) {
    const yy = oy + i * p, xx = ox + i * p;
    S(c, "line", { x1: ox, x2: ox + (n - 1) * p, y1: yy, y2: yy, stroke: LINK });
    S(c, "line", { x1: xx, x2: xx, y1: oy, y2: oy + (n - 1) * p, stroke: LINK });
    const e = (n - 1) * p;
    S(c, "path", { d: `M${ox},${yy}C${ox + 20},${yy - 16} ${ox + e - 20},${yy - 16} ${ox + e},${yy}`, fill: "none", stroke: "var(--tpu)", "stroke-opacity": 0.7 });
    S(c, "path", { d: `M${xx},${oy}C${xx - 16},${oy + 20} ${xx - 16},${oy + e - 20} ${xx},${oy + e}`, fill: "none", stroke: "var(--tpu)", "stroke-opacity": 0.7 });
  }
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) S(c, "circle", { cx: ox + j * p, cy: oy + i * p, r: 4.5, class: "tpu-fill" });
  S(c, "text", { class: "label muted topo", x: 110, y: 128, "text-anchor": "middle" }, "arcs: wraparound links");
  // 3D torus: one 4 × 4 × 4 cube, oblique projection
  const P = (i, j, k) => [59 + i * 22 + k * 12, 38 + j * 19 - k * 10];
  const L = (p1, p2) => S(d, "line", { x1: p1[0], y1: p1[1], x2: p2[0], y2: p2[1], stroke: "var(--rule-2)" });
  for (let k = n - 1; k >= 0; k--) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    if (i < n - 1) L(P(i, j, k), P(i + 1, j, k));
    if (j < n - 1) L(P(i, j, k), P(i, j + 1, k));
    if (k < n - 1) L(P(i, j, k), P(i, j, k + 1));
  }
  for (let k = n - 1; k >= 0; k--) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const q = P(i, j, k);
    S(d, "circle", { cx: q[0], cy: q[1], r: 2.8, class: "tpu-fill", "fill-opacity": k === 0 ? 1 : 0.45 + 0.15 * (n - 1 - k) });
  }
  const s0 = P(0, n - 1, 0), s1 = P(n - 1, n - 1, 0);
  S(d, "path", { d: `M${s1[0]},${s1[1]}C${s1[0] - 14},${s1[1] + 18} ${s0[0] + 14},${s0[1] + 18} ${s0[0]},${s0[1]}`, fill: "none", stroke: "var(--tpu)", "stroke-width": 1.5 });
  S(d, "text", { class: "label muted topo", x: 110, y: 128, "text-anchor": "middle" }, "arc: one wraparound link");
})();

/* =========================================================================
   Figure 7: how many chips a model needs
   ========================================================================= */
(function fit() {
  const P = $("#fit-params"), Po = $("#fit-params-out"), kv = $("#fit-kv"), kvo = $("#fit-kv-out"), tb = $("#fit-table tbody");
  let bytes = 1, mode = "infer";
  seg($("#fit-prec"), (b) => { bytes = +b.dataset.bytes; render(); });
  seg($("#fit-mode"), (b) => { mode = b.dataset.mode; $$("#fit-prec button").forEach((x) => { x.disabled = mode === "train"; }); render(); });
  function render() {
    const params = 10 ** +P.value;                               // billions
    Po.textContent = params >= 10 ? Math.round(params) : params.toFixed(1);
    kvo.textContent = "+" + kv.value + "%";
    const perParam = mode === "train" ? 16 : bytes;
    const need = params * perParam * (1 + kv.value / 100);         // GB
    $("#fit-need").textContent = need >= 1000 ? (need / 1000).toFixed(2) + " TB" : Math.round(need) + " GB";
    const rows = CHIPS.map((c) => { const n = Math.ceil(need / (c.mem * 0.9)); return { c, n, fits: n <= c.domain, cost: c.price == null ? null : n * c.price }; });
    const priced = rows.filter((r) => r.cost != null && r.fits).sort((a, b) => a.cost - b.cost);
    const cheap = priced.find((r) => r.c.onDemand);
    const fewest = rows.filter((r) => r.fits).sort((a, b) => a.n - b.n)[0];
    $("#fit-cheap").textContent = cheap ? `${cheap.c.name} × ${nf(cheap.n)}, ${fmt$(cheap.cost)} per hour` : "nothing in one fast domain";
    $("#fit-fewest").textContent = fewest ? `${fewest.c.name} × ${nf(fewest.n)}` : "none in one fast domain";
    const maxCost = Math.max(1, ...priced.map((r) => r.cost));
    tb.innerHTML = rows.sort((a, b) => (a.cost ?? 1e9) - (b.cost ?? 1e9)).map((r) => {
      const fitsTxt = r.fits ? (r.n === 1 ? "yes, one chip" : "yes, " + r.c.domainName) : `no: needs ${nf(r.n)} > ${nf(r.c.domain)} in ${r.c.domainName}`;
      const bar = r.cost == null || !r.fits ? "" : `<span style="--c:var(--${kindOf(r.c)});width:${Math.max(2, r.cost / maxCost * 100).toFixed(1)}%"></span>`;
      const cost = r.cost == null ? `n/a<span class="ast"></span>` : fmt$(r.cost) + `<span class="ast">${r.c.onDemand ? "" : "*"}</span>`;
      return `<tr class="${r.fits ? "" : "no"}"><th scope="row">${r.c.name}</th><td class="n">${nf(r.n)}</td><td class="n">${cost}</td><td class="cost-bar">${bar}</td><td class="n">${fmtMem(r.c.mem)}</td><td>${fitsTxt}</td></tr>`;
    }).join("");
  }
  [P, kv].forEach((el) => el.addEventListener("input", render));
  render();
})();

/* =========================================================================
   top bar: reading progress and the current section
   ========================================================================= */
(function topbar() {
  const bar = $("#progress"), nav = $(".topbar nav"), links = $$(".topbar nav a");
  const secs = links.map((a) => document.getElementById(a.getAttribute("href").slice(1)));
  let cur = -1;
  function update() {
    const h = document.documentElement.scrollHeight - innerHeight;
    bar.style.width = (h > 0 ? Math.min(100, scrollY / h * 100) : 0) + "%";
    let now = -1;
    secs.forEach((s, i) => { if (s.getBoundingClientRect().top < innerHeight * 0.3) now = i; });
    if (now === cur) return;
    cur = now;
    links.forEach((a, i) => { if (i === cur) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); });
    if (cur >= 0 && nav.scrollWidth > nav.clientWidth) nav.scrollLeft = links[cur].offsetLeft - nav.offsetLeft - 16;
  }
  addEventListener("scroll", update, { passive: true });
  addEventListener("resize", update);
  update();
})();
