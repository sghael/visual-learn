/* Browser-only teaching models. Copies live beside each page so file:// works. */
(() => {
  'use strict';
  const escape = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const color = {subject:'var(--c-blue)', active:'var(--c-green)', context:'var(--c-grey)', negative:'var(--c-red)', latent:'var(--c-purple)'};
  const fmt = (n, digits=2) => Number(n.toFixed(digits)).toLocaleString('en-US');
  const text = (x,y,s,attrs='') => `<text x="${x}" y="${y}" ${attrs}>${escape(s)}</text>`;
  const rect = (x,y,w,h,fill,attrs='') => `<rect x="${x}" y="${y}" width="${Math.max(0,w)}" height="${h}" fill="${fill}" ${attrs}/>`;
  const line = (x1,y1,x2,y2,stroke='var(--rule-2)') => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${stroke}"/>`;
  const svg = (body,height,label,width=592) => `<div class="scroll-x graphic" tabindex="0" role="region" aria-label="Scrollable figure"><svg viewBox="0 0 ${width} ${height}" width="${width}" height="${height}" role="img" aria-label="${escape(label)}">${body}</svg></div>`;
  const bars = (rows,unit,max) => {
    max = max || Math.max(1,...rows.map(r=>r.value));
    return svg(rows.map((r,i)=>text(0,i*65+17,r.label)+text(582,i*65+17,`${fmt(r.value)} ${unit}`,'text-anchor="end"')+rect(0,i*65+28,r.value/max*580,16,r.color||color.subject)).join(''),rows.length*65,'Directly labeled bars, same linear scale');
  };
  const button = (label,action) => {const b=document.createElement('button');b.type='button';b.className='btn';b.textContent=label;b.addEventListener('click',action);return b;};
  document.querySelectorAll('[data-widget]').forEach((el,widgetIndex) => {
    const kind=el.dataset.widget;
    const c=JSON.parse(el.querySelector('script').textContent);
    const controls=document.createElement('div');controls.className='controls';
    const graphic=document.createElement('div');graphic.className='widget-graphic';
    const result=document.createElement('p');result.className='widget-result';result.setAttribute('aria-live','polite');result.setAttribute('aria-atomic','true');
    el.append(controls,graphic,result);
    const slider=(label,min,max,step,value,update)=>{
      const wrap=document.createElement('label');wrap.className='control';
      const name=document.createElement('span');name.textContent=label;
      const input=document.createElement('input');input.type='range';input.min=min;input.max=max;input.step=step;input.value=value;
      input.id=`w${widgetIndex}-${controls.children.length}`;
      const out=document.createElement('output');out.htmlFor=input.id;out.textContent=value;
      input.addEventListener('input',()=>{out.textContent=input.value;update(+input.value);});
      wrap.append(name,input,out);controls.append(wrap);return input;
    };
    const presets=(labels,initial,update,groupLabel='Figure examples')=>{
      const group=document.createElement('span');group.className='seg';group.setAttribute('role','group');group.setAttribute('aria-label',groupLabel);
      labels.forEach((label,i)=>{const b=button(label,()=>{[...group.children].forEach(x=>x.setAttribute('aria-pressed',String(x===b)));update(i);});b.setAttribute('aria-pressed',String(i===initial));group.append(b);});controls.append(group);return group;
    };
    if(kind==='flow') {
      graphic.innerHTML=`<ol class="flow">${c.steps.map(s=>`<li><strong>${escape(s.label)}</strong><span>${escape(s.detail)}</span></li>`).join('')}</ol>`;
      result.remove();controls.remove();
    } else if(kind==='bars') {
      graphic.innerHTML=bars(c.bars,c.unit);result.textContent=c.note||'';controls.remove();
    } else if(kind==='kv') {
      let tokens=c.tokens,heads=c.kvHeads;
      const memory=h=>2*c.layers*tokens*h*c.headDim*2/2**30;
      const draw=()=>{graphic.innerHTML=bars([{label:`MHA · ${c.heads} KV heads`,value:memory(c.heads),color:color.context},{label:`Selected · ${heads} KV heads`,value:memory(heads)}],'GiB');result.textContent=`${fmt(tokens)} tokens × ${c.layers} layers × ${heads} KV heads × ${c.headDim} values × 2 (K and V) × 2 bytes = ${fmt(memory(heads),3)} GiB. KV cache only, one sequence in bf16.`;el.dataset.value=memory(heads);};
      slider('Context tokens',1024,131072,1024,tokens,v=>{tokens=v;draw();});presets(['MHA','GQA','MQA'],1,i=>{heads=[c.heads,c.kvHeads,1][i];draw();});draw();
    } else if(kind==='moe') {
      let active=c.active,example=0;
      const n=Math.min(16,c.experts),shared=c.shared||0;
      const draw=()=>{
        const selected=Array.from({length:active},(_,i)=>(example*3+i*3)%n);
        let body=text(0,18,`Token ${['A','B','C'][example]} → router → ${active} selected experts`);
        for(let i=0;i<n;i++){const x=(i%4)*148,y=42+Math.floor(i/4)*69,on=selected.includes(i);body+=rect(x,y,133,52,'none',`stroke="${on?color.active:'var(--rule-2)'}" stroke-width="${on?2:1}"`)+text(x+8,y+21,`Expert ${i+1}`)+text(x+8,y+41,on?'Selected':'Not selected',`class="${on?'selected':'muted'}"`);}
        if(shared){const y=42+Math.ceil(n/4)*69;body+=rect(0,y,577,52,'none',`stroke="${color.active}" stroke-width="2"`)+text(8,y+21,`${shared} shared expert${shared>1?'s':''} · always active`)+text(8,y+41,'Runs for every token, alongside the routed experts.');}
        graphic.innerHTML=svg(body,50+Math.ceil(n/4)*69+(shared?69:0),'Toy router with selected and always-active shared experts labeled');
        result.textContent=`${active} / ${n} routed experts active (${fmt(active/n*100)}%). ${shared?`${shared} shared expert${shared>1?'s':''} always active. `:''}Unselected weights still exist. This fraction is not a speedup.`;el.dataset.value=active;
      };
      presets(['Token A','Token B','Token C'],0,i=>{example=i;draw();});slider('Experts per token',1,Math.min(4,n),1,active,v=>{active=v;draw();});draw();
    } else if(kind==='attention') {
      const n=c.tokens||16;let query=n-1,mode=c.mode||'local';
      const modes=['full','local','hybrid'];
      const draw=()=>{let body=text(0,16,'Rows: query token · columns: earlier keys'),count=0;
        const cell=24,x0=45,y0=42;
        for(let i=0;i<n;i++){body+=text(x0+i*cell+10,35,i+1,'text-anchor="middle"');body+=text(27,y0+i*cell+15,i+1,'text-anchor="end"');for(let j=0;j<n;j++){const allowed=j<=i&&(mode==='full'||j>i-c.window||(mode==='hybrid'&&j%4===0));if(i===query&&allowed)count++;body+=rect(x0+j*cell,y0+i*cell,21,21,allowed?(i===query?color.subject:color.context):'none',`stroke="var(--rule)"`);}}
        body+=text(460,65,`Query ${query+1}`)+text(460,87,`${count} keys`)+text(460,125,'Filled: visible')+text(460,147,'Empty: masked');
        graphic.innerHTML=svg(body,y0+n*cell+8,'Causal attention mask with the selected query row highlighted');
        result.textContent=`Query ${query+1} can read ${count} earlier-or-current keys. ${mode==='hybrid'?'Toy hybrid adds every fourth earlier key; this is not an actual layer layout.':mode==='local'?`The local window includes at most ${c.window} tokens, including the query.`:'Full causal attention can read every preceding token and itself.'}`;el.dataset.value=count;
      };
      slider('Query token',1,n,1,query+1,v=>{query=v-1;draw();});presets(['Full','Local','Toy hybrid'],modes.indexOf(mode),i=>{mode=modes[i];draw();});draw();
    } else if(kind==='sparse') {
      const n=c.tokens||16;let k=c.selected||4,example=0;
      const base=c.scores||Array.from({length:n},(_,i)=>((i*7+3)%n+1)/n);
      const draw=()=>{const scores=Array.from({length:n},(_,i)=>base[(i+example*5)%n]);const chosen=scores.map((v,i)=>({v,i})).sort((a,b)=>b.v-a.v||a.i-b.i).slice(0,k).map(x=>x.i);let body=text(0,17,'Score every candidate record; costly attention reads the selected set.');
        scores.forEach((v,i)=>{const y=32+i*27,on=chosen.includes(i);body+=text(0,y+14,`Key ${i+1}`)+rect(64,y,v/Math.max(...scores)*380,17,on?color.active:color.context)+text(455,y+14,`${fmt(v)} · ${on?'selected':'skipped'}`);});
        graphic.innerHTML=svg(body,40+n*27,'Toy sparse attention indexer scores with top-k keys selected');result.textContent=`The indexer considers all ${n} keys. Expensive attention uses ${k} (${fmt(k/n*100)}%). Scores are illustrative; selecting a useful subset is learned.`;el.dataset.value=k;
      };
      presets(['Query A','Query B','Query C'],0,i=>{example=i;draw();});slider('Keys selected',1,n,1,k,v=>{k=v;draw();});draw();
    } else if(kind==='grpo') {
      let rewards=c.rewards||[0,0,1,1];
      const draw=()=>{const mean=rewards.reduce((a,b)=>a+b,0)/rewards.length;const std=Math.sqrt(rewards.reduce((s,r)=>s+(r-mean)**2,0)/rewards.length);const adv=rewards.map(r=>std?(r-mean)/std:0);let body=text(0,16,'Normalized advantage: (reward − group mean) / standard deviation');
        body+=line(295,32,295,40+rewards.length*52);
        adv.forEach((a,i)=>{const y=40+i*52,w=Math.abs(a)*100;body+=text(0,y+16,`Answer ${i+1} · reward ${rewards[i]}`)+rect(a<0?295-w:295,y,w,22,a<0?color.negative:color.active)+text(585,y+16,`${a>0?'+':''}${fmt(a,3)}`,'text-anchor="end"');});
        graphic.innerHTML=svg(body,48+rewards.length*52,'Signed group-relative advantages with zero marked');result.textContent=`Mean ${fmt(mean,3)} · standard deviation ${fmt(std,3)}. ${std?'Above-average answers are reinforced relative to the group.':'All rewards are equal: no within-group learning signal.'}`;el.dataset.value=adv.join(',');
      };
      presets(['Mixed results','All equal','One succeeds'],0,i=>{rewards=i===0?c.rewards||[0,0,1,1]:i===1?[1,1,1,1]:[0,0,0,1];draw();});draw();
    } else if(kind==='precision') {
      let bits=4;
      const draw=()=>{graphic.innerHTML=bars([16,8,4].map(b=>({label:`${b}-bit weights`,value:c.billions*b/8,color:b===bits?color.subject:color.context})),'GB',c.billions*2);result.textContent=`${fmt(c.billions)} billion weights × ${bits} / 8 = ${fmt(c.billions*bits/8)} decimal GB. Ideal packed weights only; no scales, KV cache, activations or runtime.`;el.dataset.value=c.billions*bits/8;};
      presets(['4 bits','8 bits','16 bits'],0,i=>{bits=[4,8,16][i];draw();});draw();
    } else if(kind==='scaling') {
      let params=c.baselineParams;const budget=c.baselineParams*c.baselineTokens;
      const draw=()=>{const data=budget/params,maxParams=Math.max(70,c.baselineParams*2),x=n=>55+(n-1)/(maxParams-1)*510,y=d=>215-d/budget*160;
        const curve=Array.from({length:160},(_,i)=>{const n=1+i/159*(maxParams-1);return `${i?'L':'M'}${x(n)},${y(budget/n)}`;}).join(' ');
        let body=text(0,18,'Training tokens D (trillions) · fixed approximate compute 6ND')+line(55,45,55,215)+line(55,215,575,215)+text(45,60,fmt(budget),'text-anchor="end"')+text(45,219,'0','text-anchor="end"');
        body+=`<path d="${curve}" fill="none" stroke="${color.subject}" stroke-width="2"/>`+line(x(params),y(data),x(params),215)+`<circle cx="${x(params)}" cy="${y(data)}" r="5" fill="${color.subject}"/>`;
        [1,c.baselineParams,maxParams].forEach(n=>{body+=text(x(n),236,n,'text-anchor="middle"');});
        body+=text(315,262,'Parameters N (billions)','text-anchor="middle"')+text(250,71,`Selected: ${fmt(params)}B weights, ${fmt(data,3)}T tokens`)+text(250,95,'Every point has the same N × D.');
        graphic.innerHTML=svg(body,282,'Inverse training-compute tradeoff on linear axes');result.textContent=`6ND = ${fmt(6*budget)} × 10²¹ approximate training FLOPs. ${fmt(params)}B parameters permit ${fmt(data,3)}T training tokens; dense per-token parameter work is ${fmt(params/c.baselineParams)}× baseline. This predicts a budget, not quality.`;el.dataset.value=data;};
      slider('Parameters (B)',1,Math.max(70,c.baselineParams*2),0.5,params,v=>{params=v;draw();});draw();
    } else if(kind==='latent') {
      let size=c.latent;
      const draw=()=>{graphic.innerHTML=bars([{label:'Uncompressed representation',value:c.original,color:color.context},{label:'Latent vector',value:size,color:color.latent}],'scalars',c.original);result.textContent=`${size} / ${c.original} = ${fmt(size/c.original*100)}% as many cached scalars in this dimensionality example. Separate positional state and reconstruction costs are excluded.`;el.dataset.value=size;};
      slider('Latent width',Math.min(16,c.latent),c.original,1,size,v=>{size=v;draw();});draw();
    } else if(kind==='distill') {
      let temp=1;
      const draw=()=>{const m=Math.max(...c.logits);const z=c.logits.map(x=>Math.exp((x-m)/temp));const sum=z.reduce((a,b)=>a+b,0);const probs=z.map(x=>x/sum);graphic.innerHTML=bars(c.labels.map((label,i)=>({label:`Token ${label}`,value:probs[i]*100,color:i===c.logits.indexOf(m)?color.subject:color.context})),'%',100);result.textContent=`At temperature ${temp.toFixed(1)}, the leading token gets ${fmt(Math.max(...probs)*100)}%. A hard target gives it 100%. The other probabilities preserve relative preferences; this is a toy, not the release's training temperature.`;el.dataset.value=probs.join(',');};
      slider('Temperature',0.2,4,0.1,temp,v=>{temp=v;draw();});draw();
    } else if(kind==='parallel') {
      let workers=2;
      const draw=()=>{const ends=Array(workers).fill(0),tasks=c.tasks.map((duration,i)=>{const worker=ends.indexOf(Math.min(...ends)),start=ends[worker];ends[worker]+=duration;return {i,worker,start,duration};});const total=c.tasks.reduce((a,b)=>a+b,0);let body=text(0,17,'Independent tasks · same time scale for every worker');
        for(let w=0;w<workers;w++)body+=text(0,55+w*58,`Worker ${w+1}`)+line(90,67+w*58,580,67+w*58);
        tasks.forEach(t=>{const x=90+t.start/total*480,y=32+t.worker*58,width=t.duration/total*480;body+=rect(x,y,width-2,30,t.i%2?color.context:color.subject)+text(x+5,y+20,`${t.i+1}`,`class="task-label"`);});
        body+=text(90,workers*58+34,'0')+text(580,workers*58+34,`${total} time units`,'text-anchor="end"');
        graphic.innerHTML=svg(body,workers*58+50,'Greedy scheduling of independent tasks across workers');result.textContent=`Completion time: ${Math.max(...ends)} units with ${workers} workers; ${total} units sequentially. Tasks ${c.tasks.map((x,i)=>`${i+1}: ${x}`).join(', ')}. Dependencies and coordination overhead are excluded.`;el.dataset.value=Math.max(...ends);
      };
      slider('Workers',1,4,1,workers,v=>{workers=v;draw();});draw();
    } else if(kind==='state') {
      let tokens=c.tokens||128;const state=c.stateScalars||256,kv=c.kvScalarsPerToken||64;
      const draw=()=>{
        const maxTokens=2048,maxScalars=Math.max(maxTokens*kv,state)*1.1;
        const x=t=>65+t/maxTokens*480,y=s=>230-s/maxScalars*180;
        let body=text(0,18,'Cached scalars · both axes stay fixed as history changes');
        body+=line(65,40,65,230)+line(65,230,552,230);
        [0,maxTokens*kv].forEach(s=>{body+=text(55,y(s)+4,fmt(s),'text-anchor="end"');});
        [0,1024,2048].forEach(t=>{body+=text(x(t),253,t,'text-anchor="middle"');});
        body+=line(x(0),y(0),x(maxTokens),y(maxTokens*kv),color.subject);
        body+=line(x(0),y(state),x(maxTokens),y(state),color.latent);
        body+=line(x(tokens),45,x(tokens),230);
        for(const [s,fill] of [[tokens*kv,color.subject],[state,color.latent]])body+=`<circle cx="${x(tokens)}" cy="${y(s)}" r="4" fill="${fill}"/>`;
        body+=text(90,65,`Full-attention KV: ${kv} scalars per token`)+text(305,211,`Fixed state: ${fmt(state)} scalars`)+text(310,282,'History length (tokens)','text-anchor="middle"');
        graphic.innerHTML=svg(body,302,'Linear KV growth and constant recurrent state on fixed axes');
        result.textContent=`${tokens} tokens: full KV = ${tokens} × ${kv} = ${fmt(tokens*kv)} scalars; recurrent state = ${fmt(state)} scalars. An illustrative single-layer comparison. Hybrid models also retain their attention layers' cache.`;el.dataset.value=tokens*kv;
      };
      slider('History tokens',16,2048,16,tokens,v=>{tokens=v;draw();});draw();
    } else if(kind==='sink') {
      let sinkLogit=0,enabled=true;
      const draw=()=>{
        const weights=c.logits.map(v=>Math.exp(v)),sink=enabled?Math.exp(sinkLogit):0,total=weights.reduce((a,b)=>a+b,0)+sink;
        const probabilities=weights.map(v=>v/total),unused=sink/total,output=probabilities.reduce((s,p,i)=>s+p*c.values[i],0);
        const rows=probabilities.map((p,i)=>({label:`Token ${i+1} · value ${c.values[i]}`,value:p*100}));
        rows.push({label:'Sink · contributes zero',value:unused*100,color:color.latent});
        graphic.innerHTML=bars(rows,'%',100);
        result.textContent=`Token mass = ${fmt((1-unused)*100,3)}%; sink mass = ${fmt(unused*100,3)}%. Weighted output = ${fmt(output,3)}. The sink participates in normalization, then contributes no value vector. These logits and scalar values are invented.`;
        el.dataset.value=output;
      };
      presets(['With sink','Without sink'],0,i=>{enabled=i===0;draw();},'Attention normalization');
      slider('Hypothetical sink logit',-4,4,0.5,sinkLogit,v=>{sinkLogit=v;draw();});draw();
    } else if(kind==='verifier') {
      let example=0,strict=false;
      const candidates=['6','The answer is 6.','3 × 2 is 5. Final answer: 6.','7'];
      const draw=()=>{const response=candidates[example],last=response.match(/\d+(?=\D*$)/)?.[0],reward=(strict?response:last)==='6'?1:0;graphic.innerHTML=svg(text(0,24,'Prompt: What is 3 × 2?')+line(0,42,580,42)+text(0,78,`Response: ${response}`)+text(0,125,`Checker: ${strict?'exact text equals “6”':'last number equals 6'}`)+text(0,177,`Reward: ${reward}`,`fill="${reward?color.active:color.negative}"`),200,'A toy deterministic checker scores different responses');result.textContent=`${reward?'Accepted':'Rejected'}. ${example===2&&!strict?'The answer checker misses the incorrect intermediate arithmetic.':example===1&&strict?'Exact string matching rejects a correct answer because of formatting.':'The reward depends on the checker, not just whether the response sounds convincing.'} This is not Tülu’s production verifier.`;el.dataset.value=reward;};
      presets(['Correct','Formatted','Flawed reasoning','Wrong'],0,i=>{example=i;draw();},'Response example');presets(['Final number','Exact string'],0,i=>{strict=i===1;draw();},'Checking rule');draw();
    } else if(kind==='mixture') {
      let targeted=30;
      const draw=()=>{const a=c.tokens*targeted/100,b=c.tokens-a;graphic.innerHTML=bars([{label:'Targeted midtraining data',value:a},{label:'Broad replay data',value:b,color:color.context}],'B tokens',c.tokens);result.textContent=`Fixed budget: ${c.tokens}B tokens. ${fmt(a)}B targeted + ${fmt(b)}B replay. Changing the mixture holds token count fixed, but cannot by itself predict skill or forgetting.`;el.dataset.value=a;};
      slider('Targeted share (%)',0,100,5,targeted,v=>{targeted=v;draw();});draw();
    } else if(kind==='cache') {
      let share=c.sharedLayers||4,bits=c.bits||8;
      const draw=()=>{const raw=c.tokens*c.layers*c.width*bits/8/2**20,compact=c.tokens*Math.ceil(c.layers/share)*c.width*bits/8/2**20;graphic.innerHTML=bars([{label:'Separate state at every layer',value:raw,color:color.context},{label:`One cache per ${share} layers`,value:compact}],'MiB',c.tokens*c.layers*c.width*2/2**20);result.textContent=`${c.tokens} tokens × ${Math.ceil(c.layers/share)} distinct caches × ${c.width} scalars × ${bits}/8 bytes = ${fmt(compact)} MiB. Toy reuse across depth, with no scale metadata or extra state.`;el.dataset.value=compact;};
      slider('Layers per shared cache',1,Math.min(8,c.layers),1,share,v=>{share=v;draw();});presets(['4 bits','8 bits','16 bits'],[4,8,16].indexOf(bits),i=>{bits=[4,8,16][i];draw();});draw();
    } else if(kind==='patches') {
      let patch=c.patch||2;
      const draw=()=>{const cols=Math.ceil(c.width/patch),rows=Math.ceil(c.height/patch),n=cols*rows,height=384*c.height/c.width;let body=text(0,18,'An illustrative image, partitioned into patch tokens');
        body+=rect(0,35,384,height,'none','stroke="var(--ink-2)"');
        for(let i=0;i<5;i++)body+=line(24,70+i*32,270-(i%2)*60,70+i*32,'var(--c-grey)');
        body+=rect(300,60,55,55,'none','stroke="var(--c-orange)"');
        for(let i=1;i<cols;i++)body+=line(i*patch/c.width*384,35,i*patch/c.width*384,35+height,color.subject);
        for(let i=1;i<rows;i++)body+=line(0,35+i*patch/c.height*height,384,35+i*patch/c.height*height,color.subject);
        body+=text(410,80,`${cols} columns`)+text(410,106,`${rows} rows`)+text(410,155,`${n} patch tokens`)+text(410,196,`${fmt(n*n)} pairs`);
        graphic.innerHTML=svg(body,55+height,'Image patch boundaries and the resulting token count');result.textContent=`${cols} × ${rows} = ${n} patches; dense image self-attention has ${fmt(n*n)} query-key pairs. Abstract image units, not a release's tokenizer. Patches are learned representations, not averaged pixels.`;el.dataset.value=n;};
      slider('Patch side (units)',1,4,1,patch,v=>{patch=v;draw();});draw();
    } else if(kind==='diffusion') {
      let step=0,order=0;const words=['Today,','the','small','robot','reads','a','book','.'];
      const schedules=[[2,5,0,6,3,1,4,7],[0,1,2,3,4,5,6,7]];
      const draw=()=>{const revealed=schedules[order].slice(0,step*2);let body=text(0,20,'A scripted fill order, not actual model inference');
        words.forEach((word,i)=>{const x=(i%4)*148,y=45+Math.floor(i/4)*70,on=revealed.includes(i);body+=rect(x,y,136,45,'none',`stroke="${on?color.subject:'var(--rule-2)'}"`)+text(x+68,y+28,on?word:'[MASK]','text-anchor="middle"');});
        graphic.innerHTML=svg(body,195,'A sentence gradually unmasked at multiple positions');result.textContent=`Round ${step} / 4: ${revealed.length} of 8 positions committed. ${order?'Left-to-right comparison, two positions per round.':'The illustrative confidence order fills separated positions in one round.'} Real LLaDA predicts distributions for all masked positions, then remasks a subset.`;next.disabled=step===4;el.dataset.value=revealed.length;};
      presets(['Scattered order','Left-to-right order'],0,i=>{order=i;step=0;draw();});const next=button('Denoise one round',()=>{step=Math.min(4,step+1);draw();});controls.append(next,button('Reset',()=>{step=0;draw();}));draw();
    } else if(kind==='ternary') {
      let weight=0.65;const scale=0.5;
      const draw=()=>{const q=Math.max(-1,Math.min(1,Math.sign(weight)*Math.round(Math.abs(weight/scale)))),reconstructed=q*scale;let body=text(0,18,'Three forward-pass weight values, with scale s = 0.5');body+=line(40,80,552,80);
        [-1,0,1].forEach(v=>{const x=296+v*scale*200;body+=`<circle cx="${x}" cy="80" r="5" fill="${color.subject}"/>`+text(x,112,`${v} × s`,'text-anchor="middle"');});
        const x=296+weight*200;body+=line(x,40,x,72,color.negative)+text(x,31,`Latent ${fmt(weight)}`,'text-anchor="middle"')+text(0,158,`Forward: ${q} × 0.5 = ${reconstructed}`)+text(0,187,`Rounding error: ${fmt(Math.abs(weight-reconstructed))}`);
        graphic.innerHTML=svg(body,213,'A real-valued latent weight mapped to a ternary forward-pass value');result.textContent=`round(${fmt(weight)} / 0.5), clipped to −1, 0, +1, gives ${q}. This toy rounds exact half-way ties away from zero and fixes the scale. Training adjusts higher-precision latent weights through a surrogate gradient.`;el.dataset.value=q;};
      slider('Latent weight',-1,1,0.05,weight,v=>{weight=v;draw();});draw();
    } else if(kind==='recurrence') {
      let step=0,selective=true;const values=[1,0,0,0,2,0,0,0];
      const draw=()=>{let h=0;const hist=[];for(let i=0;i<step;i++){const gate=selective?(values[i]===0?0:1):0.5;h=(1-gate)*h+gate*values[i];hist.push(h);}
        let body=text(0,20,'h(new) = (1 − gate) × h(old) + gate × input');
        values.forEach((v,i)=>{const x=i*72;body+=rect(x,45,62,40,'none',`stroke="${i<step?color.subject:'var(--rule-2)'}"`)+text(x+31,70,`${v}`,'text-anchor="middle"')+text(x+31,112,i<step?fmt(hist[i],3):'—','text-anchor="middle"');});body+=text(0,148,'Top: input · bottom: state after that input');
        graphic.innerHTML=svg(body,172,'Step through a selective scalar recurrence');result.textContent=`${step} / ${values.length} inputs processed. State = ${fmt(h,4)}. ${selective?'Zero-valued distractors leave the state unchanged.':'A fixed gate blends in each distractor and decays the stored signal.'}`;next.disabled=step===values.length;el.dataset.value=h;};
      presets(['Selective gate','Fixed gate'],0,i=>{selective=i===0;step=0;draw();});const next=button('Step',()=>{step=Math.min(values.length,step+1);draw();});controls.append(next,button('Reset',()=>{step=0;draw();}));draw();
    } else {
      throw new Error(`Unknown figure kind: ${kind}`);
    }
  });
  const links=[...document.querySelectorAll('.topbar nav a[href^="#"]')];
  const updateProgress=()=>{const max=document.documentElement.scrollHeight-innerHeight;const progress=document.querySelector('.progress');if(progress)progress.style.width=`${max>0?Math.min(100,scrollY/max*100):0}%`;let active=links[0];for(const a of links){const section=document.querySelector(a.getAttribute('href'));if(section&&section.getBoundingClientRect().top<180)active=a;}links.forEach(a=>{if(a===active)a.setAttribute('aria-current','true');else a.removeAttribute('aria-current');});};
  addEventListener('scroll',updateProgress,{passive:true});updateProgress();
})();
